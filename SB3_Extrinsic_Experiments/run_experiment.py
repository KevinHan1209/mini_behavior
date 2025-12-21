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
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from stable_baselines3.common import utils
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.logger import KVWriter
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.maskable.utils import get_action_masks

from sb3_maskable_noveld import MaskableNovelD_PPO
from PIL import Image, ImageDraw, ImageFont

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
    base_env = unwrap_base_env(env)
    mask_dict = getattr(base_env, "get_action_mask", lambda: None)()
    flat = flatten_action_mask(mask_dict, env.action_space)
    if flat is None:
        raise RuntimeError(
            f"Action mask missing or None for env {base_env.__class__.__name__}; aborting to avoid unmasked training."
        )
    return flat


class ActionMaskWrapper(gym.Wrapper):
    """Expose action_masks() for sb3 maskable algorithms."""

    def action_masks(self) -> np.ndarray:
        return np.asarray(mask_fn(self.env), dtype=np.bool_)

    def __getattr__(self, name):
        return getattr(self.env, name)


def create_base_env(env_id: str, env_kwargs: Dict[str, object], render_mode: Optional[str] = None):
    """Bare env constructor kept for legacy helpers/tests."""
    kwargs = dict(env_kwargs)
    if render_mode is not None and "render_mode" not in kwargs:
        kwargs["render_mode"] = render_mode
    return gym.make(env_id, **kwargs)


def unwrap_base_env(env) -> gym.Env:
    base = env
    while hasattr(base, "env"):
        base = base.env
    return getattr(base, "unwrapped", base)


def expected_action_nvec(base_env) -> np.ndarray:
    manip = getattr(base_env, "manipulation_actions", None)
    loco = getattr(base_env, "locomotion_actions", None)
    if manip is None or loco is None:
        raise ValueError("Environment missing manipulation_actions/locomotion_actions for maskable action space.")
    return np.asarray([len(manip), len(manip), len(loco)], dtype=np.int64)


def validate_action_space(vec_env) -> np.ndarray:
    """Ensure the vec env exposes the same nvec as the base env + noop branch."""
    env_space_nvec = getattr(vec_env.action_space, "nvec", None)
    base_env = unwrap_base_env(vec_env.envs[0])
    base_space_nvec = getattr(base_env.action_space, "nvec", None)
    if env_space_nvec is None or base_space_nvec is None:
        raise ValueError("Action space is missing nvec; maskable policy requires MultiDiscrete with nvec.")

    env_nvec = np.asarray(env_space_nvec, dtype=np.int64)
    base_nvec = np.asarray(base_space_nvec, dtype=np.int64)
    expected = expected_action_nvec(base_env)
    if env_nvec.shape != expected.shape or not np.array_equal(env_nvec, expected):
        raise ValueError(f"Vectorized env nvec {env_nvec.tolist()} does not match expected {expected.tolist()}.")
    if base_nvec.shape != expected.shape or not np.array_equal(base_nvec, expected):
        raise ValueError(f"Base env nvec {base_nvec.tolist()} does not match expected {expected.tolist()}.")
    return expected


def create_env(env_id: str, env_kwargs: Dict[str, object], seed: int, render_mode: Optional[str] = None) -> gym.Env:
    env = create_base_env(env_id, env_kwargs, render_mode=render_mode)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env = CustomObservationWrapper(env)
    env = ActionMaskWrapper(env)
    return env


def make_vec_env(env_id: str, env_kwargs: Dict[str, object], seed: int, num_envs: int) -> DummyVecEnv:
    def make_thunk(rank: int):
        def _init():
            return create_env(env_id, env_kwargs, seed=seed + rank)

        return _init

    return DummyVecEnv([make_thunk(i) for i in range(num_envs)])


class EpisodeReturnLogger(BaseCallback):
    """
    Log per-episode returns without relying on sb3's Monitor.
    Accumulates rewards per env and records stats at rollout end.
    """

    def __init__(self, num_envs: int, verbose: int = 0):
        super().__init__(verbose)
        self.num_envs = int(num_envs)
        self._running_returns = np.zeros(self.num_envs, dtype=np.float64)
        self._episode_returns: List[float] = []
        self._running_lengths = np.zeros(self.num_envs, dtype=np.int64)
        self._episode_lengths: List[int] = []

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")
        if rewards is None or dones is None:
            return True
        rewards_arr = np.asarray(rewards, dtype=np.float64).reshape(-1)
        dones_arr = np.asarray(dones, dtype=np.bool_).reshape(-1)
        if rewards_arr.shape[0] != self.num_envs:
            # If vec env size changes unexpectedly, resize buffers defensively.
            self.num_envs = rewards_arr.shape[0]
            self._running_returns = np.zeros(self.num_envs, dtype=np.float64)
            self._running_lengths = np.zeros(self.num_envs, dtype=np.int64)
        self._running_returns[: rewards_arr.shape[0]] += rewards_arr
        self._running_lengths[: rewards_arr.shape[0]] += 1
        for idx, done in enumerate(dones_arr):
            if done:
                self._episode_returns.append(float(self._running_returns[idx]))
                self._episode_lengths.append(int(self._running_lengths[idx]))
                self._running_returns[idx] = 0.0
                self._running_lengths[idx] = 0
        return True

    def _on_rollout_end(self) -> None:
        if not self._episode_returns:
            return
        mean_reward = float(np.mean(self._episode_returns))
        count = len(self._episode_returns)
        self.logger.record("rollout/ep_rew_mean", mean_reward)
        self.logger.record("rollout/ep_rew_count", count, exclude="tensorboard")
        mean_length = float(np.mean(self._episode_lengths)) if self._episode_lengths else 0.0
        self.logger.record("rollout/ep_len_mean", mean_length)
        self._episode_returns.clear()
        self._episode_lengths.clear()


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
        self._font = ImageFont.load_default()

    def _render_hist_panel(self, branch_counts: Dict[str, np.ndarray], branch_labels: Dict[str, List[str]], target_height: int) -> np.ndarray:
        """Render stacked histograms with labeled axes using matplotlib."""
        branches = ("left", "right", "loco")
        colors = [(66 / 255, 135 / 255, 245 / 255), (245 / 255, 99 / 255, 66 / 255), (66 / 255, 189 / 255, 86 / 255)]
        fig, axes = plt.subplots(len(branches), 1, figsize=(3.2, max(2.4, target_height / 100)), constrained_layout=True)
        if len(branches) == 1:
            axes = [axes]
        for ax, branch, color in zip(axes, branches, colors):
            counts = np.asarray(branch_counts.get(branch), dtype=np.int64)
            labels = branch_labels.get(branch, [])
            ax.bar(range(len(counts)), counts, color=color)
            ax.set_title(branch, fontsize=8)
            ax.set_xticks(range(len(counts)))
            ax.set_xticklabels(labels if labels else [str(i) for i in range(len(counts))], rotation=45, ha="right", fontsize=6)
            ax.tick_params(axis="y", labelsize=6)
            for spine in ax.spines.values():
                spine.set_visible(True)
            ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
        canvas = FigureCanvas(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        out = np.asarray(buf).copy()
        plt.close(fig)
        # Ensure height matches target; preserve aspect ratio
        img = Image.fromarray(out)
        if img.height != target_height:
            new_w = int(img.width * (target_height / img.height))
            img = img.resize((new_w, target_height), resample=Image.BILINEAR)
        return np.array(img.convert("RGB"), dtype=np.uint8)

    def _render_sidebars(
        self,
        frame: np.ndarray,
        branch_counts: Dict[str, np.ndarray],
        branch_labels: Dict[str, List[str]],
        cumulative_reward: float,
        carrying: Dict[str, list],
        step_idx: int,
    ) -> np.ndarray:
        """Compose left sidebar text + right matplotlib histogram panel with axes/labels."""
        h, w, _ = frame.shape
        left_w = 180

        # Left sidebar: status text
        left_img = Image.new("RGB", (left_w, h), color=(255, 255, 255))
        draw = ImageDraw.Draw(left_img)
        lines = [
            f"Step: {step_idx}",
            f"Total reward: {cumulative_reward:.2f}",
            f"Carry L: {', '.join(carrying.get('left', [])) or 'empty'}",
            f"Carry R: {', '.join(carrying.get('right', [])) or 'empty'}",
        ]
        y = 10
        for line in lines:
            draw.text((10, y), line, fill=(0, 0, 0), font=self._font)
            y += 18
        left_arr = np.array(left_img, dtype=np.uint8)

        # Right sidebar via matplotlib
        hist_panel = self._render_hist_panel(branch_counts, branch_labels, target_height=h)

        return np.concatenate([left_arr, frame, hist_panel], axis=1)

    def _rollout_frames(self) -> Optional[Tuple[List[np.ndarray], float, Dict[str, Any]]]:
        env = create_env(self.env_id, self.env_kwargs, seed=self.seed, render_mode="rgb_array")
        try:
            reset_out = env.reset()
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            frames: List[np.ndarray] = []
            total_reward = 0.0
            done = False
            steps = 0
            base_env = unwrap_base_env(env)
            branch_counts = {
                "left": np.zeros(env.action_space.nvec[0], dtype=np.int64),
                "right": np.zeros(env.action_space.nvec[1], dtype=np.int64),
                "loco": np.zeros(env.action_space.nvec[2], dtype=np.int64),
            }
            branch_valid = {
                "left": np.zeros(env.action_space.nvec[0], dtype=np.int64),
                "right": np.zeros(env.action_space.nvec[1], dtype=np.int64),
                "loco": np.zeros(env.action_space.nvec[2], dtype=np.int64),
            }
            branch_labels = {
                "left": [a.name for a in getattr(base_env, "manipulation_actions", [])],
                "right": [a.name for a in getattr(base_env, "manipulation_actions", [])],
                "loco": [a.name for a in getattr(base_env, "locomotion_actions", [])],
            }
            while steps < self.gif_steps and not done:
                mask_dict = getattr(base_env, "get_action_mask", lambda: None)()
                if mask_dict is not None:
                    # Record mask validity counts per branch
                    for key in ("left", "right", "loco"):
                        branch_valid[key] += np.asarray(mask_dict[key], dtype=np.int64)
                action_masks = flatten_action_mask(mask_dict, env.action_space)
                action, _ = self.model.predict(
                    obs,
                    deterministic=self.deterministic,
                    action_masks=action_masks,
                )
                if isinstance(action, np.ndarray):
                    act_arr = action.reshape(-1)
                else:
                    act_arr = np.asarray(action).reshape(-1)
                # Track chosen actions per branch
                for idx, key in enumerate(("left", "right", "loco")):
                    act_idx = int(act_arr[idx])
                    branch_counts[key][act_idx] += 1
                step_out = env.step(action)
                if len(step_out) == 5:
                    obs, reward, terminated, truncated, _info = step_out
                    done = bool(terminated or truncated)
                else:
                    obs, reward, done, _info = step_out
                step_rew = float(np.asarray(reward).sum())
                total_reward += step_rew
                frame = env.render()
                carry_names = {
                    "left": [obj.name for obj in getattr(base_env, "carrying", {}).get("left", [])],
                    "right": [obj.name for obj in getattr(base_env, "carrying", {}).get("right", [])],
                }
                frame = self._render_sidebars(
                    frame,
                    branch_counts=branch_counts,
                    branch_labels=branch_labels,
                    cumulative_reward=total_reward,
                    carrying=carry_names,
                    step_idx=steps,
                )
                if frame is not None:
                    frames.append(np.asarray(frame, dtype=np.uint8))
                steps += 1
            stats = {
                "steps": steps,
                "total_reward": total_reward,
                "branch_counts": branch_counts,
                "branch_valid": branch_valid,
            }
            return frames, total_reward, stats
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
        frames, total_reward, stats = rollout
        if not frames:
            return
        tmp: Optional[tempfile.NamedTemporaryFile] = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
                imageio.mimsave(tmp.name, frames, fps=10)
                video = wandb.Video(tmp.name, fps=10, format="gif")
            gif_metrics = {
                f"rollout/gif_reward_{self.num_timesteps}": total_reward,
            }
            # Log action distributions per branch
            branch_counts = stats.get("branch_counts", {})
            branch_valid = stats.get("branch_valid", {})
            for branch in ("left", "right", "loco"):
                if branch in branch_counts:
                    gif_metrics[f"rollout/{branch}_action_counts_{self.num_timesteps}"] = branch_counts[branch].tolist()
                if branch in branch_valid:
                    gif_metrics[f"rollout/{branch}_mask_counts_{self.num_timesteps}"] = branch_valid[branch].tolist()
            wandb.log(
                {
                    f"rollout/gif_{self.num_timesteps}": video,
                    **gif_metrics,
                },
                step=self.num_timesteps,
            )
            if self.verbose:
                print(
                    f"[GifLogger] Logged GIF at step {self.num_timesteps} (reward {total_reward:.3f}, "
                    f"invalid {stats.get('invalid_choices', 0)}, steps {stats.get('steps', 0)})"
                )
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
    expected_nvec = validate_action_space(vec_env)
    validate_action_space(eval_env)
    for e in vec_env.envs:
        mask_fn(e)
    for e in eval_env.envs:
        mask_fn(e)

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
    callbacks.append(EpisodeReturnLogger(num_envs=num_envs))
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

    model_nvec = np.asarray(getattr(model.action_space, "nvec", None), dtype=np.int64)
    if model_nvec.shape != expected_nvec.shape or not np.array_equal(model_nvec, expected_nvec):
        raise ValueError(
            f"Model action_space.nvec {model_nvec.tolist()} does not match expected {expected_nvec.tolist()}."
        )

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
