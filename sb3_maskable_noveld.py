"""
Maskable NovelD-PPO built on sb3_contrib's MaskablePPO (SB3 2.7.x) with RND-based intrinsic rewards.
This mirrors the NovelD logic from the legacy stable_baselines3 NovelD_PPO while using mask-aware sampling.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np
import torch as th
from gym import spaces
from torch import nn
from torch.nn import functional as F

from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.buffers import MaskableDictRolloutBuffer, MaskableRolloutBuffer
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.maskable.utils import get_action_masks, is_masking_supported
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv


class RunningMeanStd:
    """
    Welford's algorithm for running mean and variance.
    """

    def __init__(self, shape: Tuple[int, ...] = ()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, arr: np.ndarray) -> None:
        if arr.size == 0:
            return
        arr = np.asarray(arr)
        batch_count = arr.shape[0]
        batch_mean = arr.mean(axis=0)
        batch_var = arr.var(axis=0)
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        self.mean = new_mean
        self.var = np.maximum(new_var, 1e-8)
        self.count = tot_count


class RNDModel(nn.Module):
    """
    Simple Random Network Distillation module with independent target and predictor networks.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 256):
        super().__init__()
        self.target = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        target_feature = self.target(obs)
        predict_feature = self.predictor(obs)
        return predict_feature, target_feature


class MaskableNovelD_PPO(MaskablePPO):
    """
    Maskable PPO variant that augments extrinsic reward with NovelD intrinsic rewards computed via RND.
    """

    def __init__(
        self,
        policy: Union[str, Type[MaskableActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: Optional[int] = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        *,
        int_coef: float = 1.0,
        ext_coef: float = 1.0,
        int_gamma: float = 0.99,
        update_proportion: float = 0.25,
        rnd_hidden_dim: int = 256,
        rnd_output_dim: int = 256,
        rnd_learning_rate: Optional[float] = None,
        normalize_intrinsic_rewards: bool = True,
        intrinsic_reward_clamp: float = 10.0,
        use_masking: bool = True,
    ):
        self.int_coef = float(int_coef)
        self.ext_coef = float(ext_coef)
        self.int_gamma = float(int_gamma)
        self.update_proportion = float(np.clip(update_proportion, 0.0, 1.0))
        self.rnd_hidden_dim = int(rnd_hidden_dim)
        self.rnd_output_dim = int(rnd_output_dim)
        self.rnd_learning_rate = rnd_learning_rate
        self.normalize_intrinsic_rewards = bool(normalize_intrinsic_rewards)
        self.intrinsic_reward_clamp = float(intrinsic_reward_clamp)
        self._intrinsic_enabled = self.int_coef > 0.0
        self.use_masking = use_masking
        self._mask_warning_emitted = False

        self._obs_rms: Optional[RunningMeanStd] = None
        self._reward_rms: Optional[RunningMeanStd] = None
        self._rnd: Optional[RNDModel] = None
        self._rnd_optimizer: Optional[th.optim.Optimizer] = None
        self._obs_flat_dim: Optional[int] = None
        self._last_rollout_intrinsic_stats: Optional[Dict[str, float]] = None

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            target_kl=target_kl,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )

    def _setup_model(self) -> None:
        super()._setup_model()
        self._setup_noveld_modules()

    def _setup_noveld_modules(self) -> None:
        if self.observation_space is None:
            return
        if isinstance(self.observation_space, spaces.Dict):
            raise NotImplementedError("MaskableNovelD_PPO currently does not support Dict observation spaces.")
        obs_shape = self.observation_space.shape
        if obs_shape is None:
            raise ValueError("Observation space must define a shape.")
        self._obs_flat_dim = int(np.prod(obs_shape))
        self._obs_rms = RunningMeanStd(shape=(self._obs_flat_dim,))
        self._reward_rms = RunningMeanStd(shape=())
        if self._intrinsic_enabled:
            self._rnd = RNDModel(self._obs_flat_dim, hidden_dim=self.rnd_hidden_dim, output_dim=self.rnd_output_dim).to(
                self.device
            )
            lr = self.rnd_learning_rate
            if lr is None:
                lr = float(self.lr_schedule(1.0))
            self._rnd_optimizer = th.optim.Adam(self._rnd.predictor.parameters(), lr=lr)
        else:
            self._rnd = None
            self._rnd_optimizer = None

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: Union[MaskableRolloutBuffer, MaskableDictRolloutBuffer],
        n_rollout_steps: int,
        use_masking: Optional[bool] = None,
    ) -> bool:
        assert isinstance(
            rollout_buffer, (MaskableRolloutBuffer, MaskableDictRolloutBuffer)
        ), "RolloutBuffer doesn't support action masking"
        assert self._last_obs is not None, "No previous observation was provided"

        self.policy.set_training_mode(False)
        n_steps = 0
        action_masks = None
        rollout_buffer.reset()

        use_mask = self.use_masking if use_masking is None else use_masking
        if use_mask and not is_masking_supported(env):
            raise ValueError("Environment does not support action masking. Consider using ActionMasker wrapper.")

        callback.on_rollout_start()
        intrinsic_records = []

        while n_steps < n_rollout_steps:
            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)  # type: ignore[arg-type]
                if use_mask:
                    action_masks = get_action_masks(env)
                    if action_masks is None:
                        # Hard fail to surface mask wiring issues immediately
                        raise RuntimeError(
                            "MaskableNovelD_PPO collected rollouts with use_masking=True but env returned no action masks."
                        )
                actions, values, log_probs = self.policy(obs_tensor, action_masks=action_masks)

            actions = actions.cpu().numpy()
            new_obs, rewards, dones, infos = env.step(actions)
            rewards = rewards.astype(np.float32)

            self.num_timesteps += env.num_envs
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

            augmented_rewards = rewards
            intrinsic_reward = None
            if self._intrinsic_enabled and self._rnd is not None:
                intrinsic_reward = self._compute_intrinsic_reward(new_obs)
                intrinsic_records.append(intrinsic_reward.copy())
                augmented_rewards = self.ext_coef * rewards + self.int_coef * intrinsic_reward

            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    augmented_rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,
                actions,
                augmented_rewards,
                self._last_episode_starts,
                values,
                log_probs,
                action_masks=action_masks,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        callback.on_rollout_end()

        if intrinsic_records:
            flat_intrinsic = np.concatenate(intrinsic_records)
            self._last_rollout_intrinsic_stats = {
                "mean": float(np.mean(flat_intrinsic)),
                "std": float(np.std(flat_intrinsic)),
                "max": float(np.max(flat_intrinsic)),
            }
        else:
            self._last_rollout_intrinsic_stats = None

        return True

    def _compute_intrinsic_reward(self, obs: np.ndarray) -> np.ndarray:
        assert self._rnd is not None and self._obs_rms is not None and self._reward_rms is not None
        obs_tensor = th.as_tensor(obs, device=self.device, dtype=th.float32)
        obs_tensor = obs_tensor.view(obs_tensor.shape[0], -1)
        self._obs_rms.update(obs_tensor.cpu().numpy())
        mean_tensor = th.as_tensor(self._obs_rms.mean, device=self.device, dtype=th.float32)
        var_tensor = th.as_tensor(self._obs_rms.var, device=self.device, dtype=th.float32)
        normalized_obs = (obs_tensor - mean_tensor) / th.sqrt(var_tensor + 1e-8)
        with th.no_grad():
            predict_feature, target_feature = self._rnd(normalized_obs)
            intrinsic = F.mse_loss(predict_feature, target_feature, reduction="none").mean(dim=1)
        intrinsic = intrinsic.clamp(min=0.0)

        if self.normalize_intrinsic_rewards:
            self._reward_rms.update(intrinsic.cpu().numpy().reshape(-1))
            intrinsic = intrinsic / math.sqrt(float(self._reward_rms.var))

        if self.intrinsic_reward_clamp > 0:
            intrinsic = intrinsic.clamp(max=self.intrinsic_reward_clamp)

        return intrinsic.detach().cpu().numpy().astype(np.float32)

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        clip_range_vf = self.clip_range_vf(self._current_progress_remaining) if self.clip_range_vf is not None else None

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        approx_kl_divs = []
        rnd_losses = []
        continue_training = True

        for epoch in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions, action_masks=rollout_data.action_masks
                )
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )

                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                entropy_loss = -th.mean(entropy) if entropy is not None else -th.mean(-log_prob)
                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

                if self._intrinsic_enabled and self._rnd is not None and self._rnd_optimizer is not None:
                    rnd_loss = self._update_rnd(rollout_data.observations)
                    rnd_losses.append(rnd_loss)

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if rnd_losses:
            self.logger.record("train/rnd_loss", float(np.mean(rnd_losses)))
        if self._intrinsic_enabled and self._last_rollout_intrinsic_stats:
            self.logger.record("rollout/intrinsic_reward_mean", self._last_rollout_intrinsic_stats["mean"])
            self.logger.record("rollout/intrinsic_reward_std", self._last_rollout_intrinsic_stats["std"])
            self.logger.record("rollout/intrinsic_reward_max", self._last_rollout_intrinsic_stats["max"])
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def _update_rnd(self, observations: th.Tensor) -> float:
        assert self._rnd is not None and self._obs_rms is not None and self._rnd_optimizer is not None
        obs = observations.float()
        if obs.ndim > 2:
            obs = obs.view(obs.shape[0], -1)
        mean_tensor = th.as_tensor(self._obs_rms.mean, device=obs.device, dtype=th.float32)
        var_tensor = th.as_tensor(self._obs_rms.var, device=obs.device, dtype=th.float32)
        normalized_obs = (obs - mean_tensor) / th.sqrt(var_tensor + 1e-8)

        predict_feature, target_feature = self._rnd(normalized_obs.detach())
        forward_loss = F.mse_loss(predict_feature, target_feature.detach(), reduction="none").mean(dim=1)

        if self.update_proportion < 1.0:
            mask = (th.rand_like(forward_loss) < self.update_proportion).float()
            denom = mask.sum().clamp(min=1.0)
            forward_loss = (forward_loss * mask).sum() / denom
        else:
            forward_loss = forward_loss.mean()

        self._rnd_optimizer.zero_grad()
        forward_loss.backward()
        th.nn.utils.clip_grad_norm_(self._rnd.predictor.parameters(), self.max_grad_norm)
        self._rnd_optimizer.step()
        return float(forward_loss.item())
