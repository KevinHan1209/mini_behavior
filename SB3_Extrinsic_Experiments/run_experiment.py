#!/usr/bin/env python3
"""Maskable-only SB3 NovelD experiment runner using gymnasium."""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import imageio
import numpy as np
import torch
from stable_baselines3.common import utils
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.logger import KVWriter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.maskable.utils import get_action_masks

from sb3_maskable_noveld import MaskableNovelD_PPO

from env_wrapper import CustomObservationWrapper
from mini_behavior.register import env_list, register

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None
    print("Warning: wandb not available. Proceeding without wandb logging.")


class WandbOutputFormat(KVWriter):
    """Forward SB3 logger scalars to an active Weights & Biases run."""

    def write(
        self,
        key_values: Dict[str, Any],
        key_excluded: Dict[str, Union[str, Tuple[str, ...]]],
        step: int = 0,
    ) -> None:
        if not WANDB_AVAILABLE or wandb.run is None:
            return
        payload: Dict[str, float] = {}
        for key, value in key_values.items():
            exclude = key_excluded.get(key)
            if exclude is not None and ("wandb" in exclude or "json" in exclude):
                continue
            if isinstance(value, (float, int, np.floating, np.integer)):
                payload[key] = float(value)
            elif isinstance(value, np.ndarray) and value.size == 1:
                payload[key] = float(value.squeeze())
            elif hasattr(value, "item"):
                try:
                    payload[key] = float(value.item())
                except Exception:
                    continue
        if payload:
            wandb.log(payload, step=step)

    def close(self) -> None:
        return


def flatten_action_mask(mask_dict: Optional[Dict[str, np.ndarray]], action_space: gym.Space) -> Optional[np.ndarray]:
    """Convert dict masks into a flat boolean mask matching MultiDiscrete.nvec."""
    if mask_dict is None or not hasattr(action_space, "nvec"):
        return None
    mask_parts = []
    for key, dim in zip(("left", "right", "loco"), action_space.nvec.tolist()):
        branch = mask_dict.get(key)
        if branch is None:
            mask_parts.append(np.ones(int(dim), dtype=np.bool_))
        else:
            arr = np.asarray(branch, dtype=np.bool_).reshape(-1)
            if arr.size < dim:
                arr = np.pad(arr, (0, dim - arr.size), constant_values=False)
            mask_parts.append(arr[:dim])
    return np.concatenate(mask_parts, axis=0).astype(np.bool_)


def mask_fn(env) -> np.ndarray:
    mask_dict = getattr(env, "get_action_mask", lambda: None)()
    flat = flatten_action_mask(mask_dict, env.action_space)
    if flat is None:
        raise RuntimeError("Action mask missing or None; aborting to avoid unmasked training.")
    return flat


class ActionMaskWrapper(gym.Wrapper):
    """Expose action_masks() for sb3 maskable algorithms."""

    def action_masks(self) -> np.ndarray:
        return np.asarray(mask_fn(self.env), dtype=np.bool_)

    def __getattr__(self, name):
        return getattr(self.env, name)


def create_base_env(env_id: str, env_kwargs: Dict[str, object]):
    """Bare env constructor kept for legacy helpers/tests."""
    return gym.make(env_id, **env_kwargs)


def create_env(env_id: str, env_kwargs: Dict[str, object], seed: int) -> gym.Env:
    env = create_base_env(env_id, env_kwargs)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env = CustomObservationWrapper(env)
    env = ActionMaskWrapper(env)
    env = Monitor(env)
    return env


def make_vec_env(env_id: str, env_kwargs: Dict[str, object], seed: int, num_envs: int) -> DummyVecEnv:
    def make_thunk(rank: int):
        def _init():
            return create_env(env_id, env_kwargs, seed=seed + rank)

        return _init

    return DummyVecEnv([make_thunk(i) for i in range(num_envs)])


class TrueEpisodeRewardCallback(BaseCallback):
    """Log Monitor-style episode rewards from the latest rollout."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._episode_rewards: List[float] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if not infos:
            return True
        for info in infos:
            episode = info.get("episode") if isinstance(info, dict) else None
            if isinstance(episode, dict) and "r" in episode:
                self._episode_rewards.append(float(episode["r"]))
        return True

    def _on_rollout_end(self) -> None:
        if not self._episode_rewards:
            return
        mean_reward = float(np.mean(self._episode_rewards))
        self.logger.record("rollout/true_ep_rew_mean", mean_reward)
        self.logger.record("rollout/true_ep_rew_count", len(self._episode_rewards))
        self._episode_rewards.clear()


class EvalEpRewMeanCallback(BaseCallback):
    """Periodic masked evaluation mirroring ep_rew_mean."""

    def __init__(
        self,
        eval_env,
        n_eval_episodes: int,
        eval_freq: int,
        use_masks: bool = True,
        deterministic: bool = False,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.n_eval_episodes = max(int(n_eval_episodes), 1)
        self.eval_freq = max(int(eval_freq), 1)
        self.use_masks = use_masks
        self.deterministic = deterministic

    def _run_eval(self) -> float:
        obs = self.eval_env.reset()
        episode_returns: List[float] = []
        running_returns = np.zeros(self.eval_env.num_envs, dtype=np.float32)
        while len(episode_returns) < self.n_eval_episodes:
            action_masks = get_action_masks(self.eval_env) if self.use_masks else None
            actions, _ = self.model.predict(
                obs,
                deterministic=self.deterministic,
                action_masks=action_masks,
            )
            obs, rewards, dones, _infos = self.eval_env.step(actions)
            running_returns += np.asarray(rewards, dtype=np.float32)
            for idx, done in enumerate(dones):
                if done:
                    episode_returns.append(float(running_returns[idx]))
                    running_returns[idx] = 0.0
                    if len(episode_returns) >= self.n_eval_episodes:
                        break
        return float(np.mean(episode_returns))

    def _on_step(self) -> bool:
        if self.num_timesteps == 0 or (self.num_timesteps % self.eval_freq) != 0:
            return True
        mean_reward = self._run_eval()
        self.logger.record("eval/ep_rew_mean_masked", mean_reward)
        self.logger.record("eval/n_episodes", self.n_eval_episodes, exclude="tensorboard")
        if self.verbose:
            print(f"[EvalEpRewMeanCallback] step {self.num_timesteps}: eval/ep_rew_mean_masked={mean_reward:.3f}")
        return True


class GifLoggerCallback(BaseCallback):
    """Render a short masked rollout to GIF and log to wandb."""

    def __init__(
        self,
        env_id: str,
        env_kwargs: Dict[str, object],
        log_interval: int,
        gif_steps: int,
        seed: int,
        deterministic: bool,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.env_id = env_id
        self.env_kwargs = env_kwargs
        self.log_interval = max(int(log_interval), 1)
        self.gif_steps = gif_steps
        self.seed = seed
        self.deterministic = deterministic
        self._next_log = self.log_interval

    def _rollout_frames(self) -> Optional[Tuple[List[np.ndarray], float]]:
        env = create_env(self.env_id, self.env_kwargs, seed=self.seed)
        try:
            reset_out = env.reset()
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            frames: List[np.ndarray] = []
            total_reward = 0.0
            done = False
            steps = 0
            while steps < self.gif_steps and not done:
                action_masks = get_action_masks(env)
                action, _ = self.model.predict(
                    obs,
                    deterministic=self.deterministic,
                    action_masks=action_masks,
                )
                step_out = env.step(action)
                if len(step_out) == 5:
                    obs, reward, terminated, truncated, _info = step_out
                    done = bool(terminated or truncated)
                else:
                    obs, reward, done, _info = step_out
                total_reward += float(np.asarray(reward).sum())
                frame = env.render()
                if frame is not None:
                    frames.append(np.asarray(frame, dtype=np.uint8))
                steps += 1
            return frames, total_reward
        finally:
            env.close()

    def _maybe_log(self) -> None:
        if not WANDB_AVAILABLE or wandb.run is None:
            return
        try:
            rollout = self._rollout_frames()
        except Exception as exc:
            if self.verbose:
                print(f"[GifLogger] Skipping GIF logging: {exc}")
            return
        if rollout is None:
            return
        frames, total_reward = rollout
        if not frames:
            return
        tmp: Optional[tempfile.NamedTemporaryFile] = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
                imageio.mimsave(tmp.name, frames, fps=10)
                video = wandb.Video(tmp.name, fps=10, format="gif")
            wandb.log(
                {
                    f"rollout/gif_{self.num_timesteps}": video,
                    f"rollout/gif_reward_{self.num_timesteps}": total_reward,
                },
                step=self.num_timesteps,
            )
            if self.verbose:
                print(f"[GifLogger] Logged GIF at step {self.num_timesteps} (reward {total_reward:.3f})")
        finally:
            if tmp is not None:
                try:
                    os.remove(tmp.name)
                except Exception:
                    pass

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_log:
            self._maybe_log()
            self._next_log += self.log_interval
        return True


def run_experiment(config_path: str, output_dir: str = "results", seed: int = 1):
    """Run a single experiment based on config file."""

    with open(config_path, "r") as f:
        config = json.load(f)

    exp_name = config["name"]
    extrinsic_rewards = config["extrinsic_rewards"]
    hyperparams = config["hyperparameters"]
    env_config = config["env_config"]
    wandb_config = config.get("wandb_config", {})

    exp_output_dir = Path(output_dir) / exp_name
    exp_output_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_output_dir / "experiment_config.json", "w") as f:
        json.dump(config, f, indent=2)

    checkpoint_dir = exp_output_dir / "checkpoints"
    tensorboard_log_dir = exp_output_dir / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    env_id = env_config["env_id"]
    task = env_config["task"]
    env_kwargs = {
        "room_size": env_config["room_size"],
        "max_steps": env_config["max_steps"],
        "extrinsic_rewards": extrinsic_rewards,
    }
    if env_id not in env_list:
        register(id=env_id, entry_point=f"mini_behavior.envs:{task}Env", kwargs=env_kwargs)

    utils.set_random_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_envs = int(hyperparams["num_envs"])
    vec_env = make_vec_env(env_id, env_kwargs, seed=seed, num_envs=num_envs)
    eval_num_envs = int(hyperparams.get("eval_num_envs", num_envs))
    eval_env = make_vec_env(env_id, env_kwargs, seed=seed, num_envs=eval_num_envs)

    model = MaskableNovelD_PPO(
        policy=MaskableActorCriticPolicy,
        env=vec_env,
        learning_rate=hyperparams["learning_rate"],
        n_steps=hyperparams["n_steps"],
        batch_size=hyperparams["batch_size"],
        n_epochs=hyperparams["n_epochs"],
        gamma=hyperparams["gamma"],
        gae_lambda=hyperparams["gae_lambda"],
        clip_range=hyperparams["clip_range"],
        ent_coef=hyperparams["ent_coef"],
        vf_coef=hyperparams["vf_coef"],
        max_grad_norm=hyperparams["max_grad_norm"],
        int_coef=hyperparams.get("int_coef", 1.0),
        ext_coef=hyperparams.get("ext_coef", 1.0),
        int_gamma=hyperparams.get("int_gamma", 0.99),
        update_proportion=hyperparams.get("update_proportion", 0.25),
        tensorboard_log=str(tensorboard_log_dir),
        device=device,
        verbose=1,
        use_masking=True,
    )

    logger = utils.configure_logger(
        model.verbose,
        model.tensorboard_log,
        tb_log_name=exp_name,
        reset_num_timesteps=True,
    )
    model.set_logger(logger)

    wandb_run = None
    if WANDB_AVAILABLE:
        full_wandb_config = {
            "experiment_name": exp_name,
            "extrinsic_rewards": extrinsic_rewards,
            **hyperparams,
            **env_config,
        }
        wandb_run = wandb.init(
            project=wandb_config.get("project", "sb3-noveld-extrinsic"),
            name=exp_name,
            group=wandb_config.get("group", "extrinsic_only"),
            tags=wandb_config.get("tags", ["sb3", "noveld", "extrinsic"]),
            config=full_wandb_config,
        )
        if not any(isinstance(fmt, WandbOutputFormat) for fmt in model.logger.output_formats):
            model.logger.output_formats.append(WandbOutputFormat())

    checkpoint_interval = int(os.environ.get("CHECKPOINT_INTERVAL", 500_000))
    gif_interval = int(os.environ.get("GIF_LOG_INTERVAL", checkpoint_interval))
    callbacks: List[BaseCallback] = []
    if checkpoint_interval > 0:
        callbacks.append(
            CheckpointCallback(save_freq=checkpoint_interval, save_path=str(checkpoint_dir), name_prefix="checkpoint")
        )
    callbacks.append(TrueEpisodeRewardCallback())
    callbacks.append(
        EvalEpRewMeanCallback(
            eval_env=eval_env,
            n_eval_episodes=eval_num_envs,
            eval_freq=int(hyperparams.get("eval_freq", 10_000)),
            deterministic=False,
            verbose=1,
        )
    )
    if WANDB_AVAILABLE and wandb_run is not None and gif_interval > 0:
        callbacks.append(
            GifLoggerCallback(
                env_id=env_id,
                env_kwargs=env_kwargs,
                log_interval=gif_interval,
                gif_steps=int(os.environ.get("GIF_STEPS", 200)),
                seed=seed,
                deterministic=False,
                verbose=1,
            )
        )

    callback_arg: Optional[BaseCallback] = None
    if callbacks:
        callback_arg = callbacks[0] if len(callbacks) == 1 else CallbackList(callbacks)

    model.learn(total_timesteps=hyperparams["total_timesteps"], callback=callback_arg)

    final_model_path = exp_output_dir / "final_model"
    model.save(str(final_model_path))

    with open(exp_output_dir / "SUCCESS", "w") as f:
        f.write(f"Experiment {exp_name} completed successfully\n")

    vec_env.close()
    eval_env.close()
    if WANDB_AVAILABLE and wandb_run is not None:
        wandb.finish()

    return exp_output_dir


def main():
    parser = argparse.ArgumentParser(description="Run a single SB3 NovelD experiment with extrinsic rewards")
    parser.add_argument("config", help="Path to experiment configuration JSON file")
    parser.add_argument("--output-dir", default="results", help="Directory to save results (default: results)")
    parser.add_argument("--seed", type=int, default=1, help="Random seed (default: 1)")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Config file '{args.config}' not found")
        sys.exit(1)

    try:
        output_dir = run_experiment(args.config, args.output_dir, args.seed)
        print(f"\nExperiment completed successfully!")
        print(f"Results saved to: {output_dir}")
    except Exception as e:
        print(f"\nError running experiment: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
