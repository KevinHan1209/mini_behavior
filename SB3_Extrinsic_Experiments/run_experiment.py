#!/usr/bin/env python3
"""
Run a single SB3 NovelD experiment with extrinsic rewards only.
Based on test/test_NovelD.py but adapted for configurable extrinsic rewards.
"""
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gym
import numpy as np
import tempfile
import torch
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from gym import Env

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Proceeding without wandb logging.")

from stable_baselines3 import NovelD_PPO
from stable_baselines3.common import utils
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EveryNTimesteps
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import KVWriter

from env_wrapper import CustomObservationWrapper
from mini_behavior.register import env_list, register
from mini_behavior.utils.states_base import RelativeObjectState


class WandbOutputFormat(KVWriter):
    """Forward SB3 logger scalars to an active Weights & Biases run."""

    def write(self, key_values: Dict[str, Any], key_excluded: Dict[str, Union[str, Tuple[str, ...]]], step: int = 0) -> None:
        if not WANDB_AVAILABLE or wandb.run is None:
            return
        log_payload: Dict[str, float] = {}
        for key, value in key_values.items():
            exclude = key_excluded.get(key)
            if exclude is not None and ("wandb" in exclude or "json" in exclude):
                continue
            if isinstance(value, (float, int, np.floating, np.integer)):
                log_payload[key] = float(value)
            elif isinstance(value, np.ndarray) and value.size == 1:
                log_payload[key] = float(value.squeeze())
            elif isinstance(value, torch.Tensor) and value.numel() == 1:
                log_payload[key] = float(value.item())
        if log_payload:
            wandb.log(log_payload, step=step)

    def close(self) -> None:
        return


class ActionDistributionCallback(BaseCallback):
    """Track and log the empirical action distribution during training."""

    def __init__(self, action_dims: List[int], log_interval: int, verbose: int = 0):
        super().__init__(verbose)
        self.action_dims = action_dims
        self.log_interval = max(int(log_interval), 1)
        self._counts = [np.zeros(dim, dtype=np.int64) for dim in action_dims]
        self._steps_accumulated = 0

    def _accumulate(self, actions) -> int:
        if hasattr(actions, "detach"):
            actions = actions.detach().cpu().numpy()
        actions_arr = np.asarray(actions)
        if actions_arr.ndim == 1:
            actions_arr = actions_arr.reshape(1, -1)
        for branch, counts in enumerate(self._counts):
            indices = actions_arr[:, branch].astype(np.int64)
            np.add.at(counts, indices, 1)
        return actions_arr.shape[0]

    def _flush(self) -> None:
        for branch, counts in enumerate(self._counts):
            total = counts.sum()
            if total > 0:
                freqs = counts / total
                for action_idx, freq in enumerate(freqs):
                    self.logger.record(
                        f"action_dist/branch_{branch}/a{action_idx}",
                        float(freq),
                    )
            self._counts[branch].fill(0)
        self._steps_accumulated = 0

    def _on_step(self) -> bool:
        actions = self.locals.get("actions")
        if actions is None:
            return True
        batch_size = self._accumulate(actions)
        self._steps_accumulated += batch_size
        if self._steps_accumulated >= self.log_interval:
            self._flush()
        return True

    def _on_training_end(self) -> None:
       if self._steps_accumulated > 0:
           self._flush()


class GifLoggerCallback(BaseCallback):
    """Render a short rollout to GIF and log to wandb at fixed intervals."""

    def __init__(
        self,
        env_id: str,
        env_kwargs: Dict[str, object],
        log_interval: int,
        gif_steps: int = 200,
        seed: int = 1,
        deterministic: bool = False,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.env_id = env_id
        self.env_kwargs = env_kwargs
        self.log_interval = max(int(log_interval), 1)
        self.gif_steps = gif_steps
        self.seed = seed
        self.deterministic = deterministic
        # Start logging at the configured interval
        self._next_log = self.log_interval

    def _get_action_labels(self, base_env, action_dims: List[int]) -> List[List[str]]:
        """Resolve human-readable labels for each action branch."""
        labels: List[List[str]] = []

        # Manipulation actions are shared for both arms
        manip_enum = getattr(base_env, "manipulation_actions", None)
        manip_labels = [a.name for a in manip_enum] if manip_enum is not None else None

        # Locomotion actions
        loco_enum = getattr(base_env, "locomotion_actions", None)
        loco_labels = [a.name for a in loco_enum] if loco_enum is not None else None

        for branch_idx, dim in enumerate(action_dims):
            if branch_idx in (0, 1) and manip_labels and len(manip_labels) == dim:
                labels.append(manip_labels)
            elif branch_idx == 2 and loco_labels and len(loco_labels) == dim:
                labels.append(loco_labels)
            else:
                labels.append([f"a{i}" for i in range(dim)])
        return labels

    def _summarize_carrying(self, base_env) -> Dict[str, List[str]]:
        """Return a stable, printable snapshot of carried objects per arm."""
        carrying = getattr(base_env, "carrying", {})
        summary: Dict[str, List[str]] = {}
        for arm in ("left", "right"):
            held_items = []
            for obj in carrying.get(arm, []):
                if hasattr(obj, "get_name"):
                    held_items.append(obj.get_name())
                elif hasattr(obj, "name"):
                    held_items.append(str(obj.name))
                elif hasattr(obj, "type"):
                    held_items.append(str(obj.type))
                else:
                    held_items.append(str(obj))
            summary[arm] = sorted(held_items)
        return summary

    def _render_action_panel(
        self,
        action_counts: List[np.ndarray],
        action_labels: List[List[str]],
    ) -> np.ndarray:
        """Render histograms to an RGB array."""
        branches = ["Left Arm", "Right Arm", "Locomotion"]
        fig, axes = plt.subplots(
            nrows=len(action_counts),
            ncols=1,
            figsize=(3.3, 6),
            constrained_layout=True,
        )
        axes = np.atleast_1d(axes)
        for idx, (ax, counts) in enumerate(zip(axes, action_counts)):
            labels = action_labels[idx] if idx < len(action_labels) else [f"a{i}" for i in range(len(counts))]
            total = counts.sum()
            freqs = counts / total if total > 0 else np.zeros_like(counts, dtype=float)
            ax.bar(range(len(counts)), freqs, color="#4c72b0")
            ax.set_ylim(0, 1)
            ax.set_title(f"{branches[idx] if idx < len(branches) else f'Branch {idx}'} ({int(total)} steps)", fontsize=8)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=90, fontsize=6)
            ax.set_yticks([0, 0.5, 1.0])
            ax.grid(True, axis="y", linewidth=0.3)

        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        panel = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)
        plt.close(fig)
        return panel

    def _render_status_panel(
        self,
        carrying: Dict[str, List[str]],
        success_counts: Dict[str, int],
    ) -> np.ndarray:
        """Render success/failure counts and carrying info to an RGB array."""
        fig, ax = plt.subplots(figsize=(2.8, 3.5))
        success = success_counts.get("success", 0)
        failure = success_counts.get("failure", 0)
        total = success + failure
        ax.bar(["Success", "Failure"], [success, failure], color=["#2ca02c", "#d62728"])
        ax.set_title(f"Actions (n={total})", fontsize=10)
        ax.set_ylim(0, max(1, success, failure))
        ax.grid(True, axis="y", linewidth=0.3)

        # Carrying info underneath as text box
        carrying_text = "\n".join(
            f"{arm.title()}: {', '.join(items) if items else 'empty'}" for arm, items in carrying.items()
        )
        fig.text(0.5, 0.02, carrying_text, ha="center", va="bottom", fontsize=9)

        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        panel = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)
        plt.close(fig)
        return panel

    def _combine_with_panels(
        self,
        frame: np.ndarray,
        left_panel: Optional[np.ndarray],
        right_panel: Optional[np.ndarray],
    ) -> np.ndarray:
        """Place optional left/right panels around the environment frame."""
        env_img = Image.fromarray(frame)
        pieces = []

        def _resize_panel(panel_arr: np.ndarray) -> Image.Image:
            img = Image.fromarray(panel_arr)
            new_height = env_img.height
            new_width = max(1, int(img.width * (new_height / img.height)))
            return img.resize((new_width, new_height), resample=Image.BILINEAR)

        if left_panel is not None:
            pieces.append(_resize_panel(left_panel))
        pieces.append(env_img)
        if right_panel is not None:
            pieces.append(_resize_panel(right_panel))

        total_width = sum(p.width for p in pieces)
        combined = Image.new("RGB", (total_width, env_img.height), color=(255, 255, 255))

        offset = 0
        for img in pieces:
            combined.paste(img, (offset, 0))
            offset += img.width

        return np.array(combined)

    def _augment_frame(
        self,
        frame: np.ndarray,
        action_counts: List[np.ndarray],
        action_labels: List[List[str]],
        carrying: Dict[str, List[str]],
        success_counts: Dict[str, int],
    ) -> np.ndarray:
        """Attach live action histograms, carrying info, and success stats to a frame."""
        try:
            right_panel = self._render_action_panel(action_counts, action_labels)
            left_panel = self._render_status_panel(carrying, success_counts)
            return self._combine_with_panels(frame, left_panel=left_panel, right_panel=right_panel)
        except Exception as exc:
            if self.verbose:
                print(f"[GifLogger] Falling back to raw frame (panel render failed: {exc})")
            return frame

    def _maybe_log(self) -> None:
        if not WANDB_AVAILABLE or wandb.run is None:
            return
        import numpy as np

        env = gym.make(self.env_id, **self.env_kwargs)
        env = CustomObservationWrapper(env)
        base_env = env.env if hasattr(env, "env") else env
        try:
            reset_out = env.reset(seed=self.seed)
        except TypeError:
            # Older gym reset signature
            if hasattr(env, "seed"):
                env.seed(self.seed)
            reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        frames = []
        total_reward = 0.0
        done = False
        steps = 0
        action_dims = env.action_space.nvec.tolist() if hasattr(env.action_space, "nvec") else [env.action_space.n]
        action_counts = [np.zeros(dim, dtype=np.int64) for dim in action_dims]
        action_labels = self._get_action_labels(base_env, action_dims)
        success_counts = {"success": 0, "failure": 0}

        # Capture initial state before any action
        initial_carrying = self._summarize_carrying(base_env)
        initial_frame = base_env.render(mode="rgb_array")
        frames.append(self._augment_frame(initial_frame, action_counts, action_labels, initial_carrying, success_counts))

        while steps < self.gif_steps and not done:
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
            obs, reward, done, _info = env.step(action)
            total_reward += float(reward)
            action_arr = np.asarray(action).reshape(-1)
            for branch_idx, action_idx in enumerate(action_arr):
                if branch_idx >= len(action_counts):
                    break
                idx_int = int(action_idx)
                if 0 <= idx_int < len(action_counts[branch_idx]):
                    action_counts[branch_idx][idx_int] += 1
            success_flag = bool(getattr(base_env, "action_done", False))
            if success_flag:
                success_counts["success"] += 1
            else:
                success_counts["failure"] += 1
            carrying_snapshot = self._summarize_carrying(base_env)
            frame = base_env.render(mode="rgb_array")
            frames.append(self._augment_frame(frame, action_counts, action_labels, carrying_snapshot, success_counts))
            steps += 1
        env.close()
        if not frames:
            return
        tmp = None
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
        except Exception as exc:
            print(f"[GifLogger] Skipping GIF logging: {exc}")
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


def make_vec_env(env_id: str, env_kwargs: Optional[Dict[str, object]], seed: int, num_envs: int) -> DummyVecEnv:
    """Create a monitored DummyVecEnv with CustomObservationWrapper for NovelD training."""

    def make_thunk(rank: int):
        def _init():
            env = gym.make(env_id, **(env_kwargs or {}))
            env = CustomObservationWrapper(env)
            env = Monitor(env)
            try:
                env.reset(seed=seed + rank)
            except TypeError:
                if hasattr(env, "seed"):
                    env.seed(seed + rank)
            return env

        return _init

    return DummyVecEnv([make_thunk(i) for i in range(num_envs)])


def build_checkpoint_callbacks(save_dir: Path, interval: int) -> List[BaseCallback]:
    """Create SB3 callbacks that checkpoint the policy every `interval` timesteps."""
    callbacks: List[BaseCallback] = []
    if interval <= 0:
        return callbacks
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_cb = CheckpointCallback(save_freq=1, save_path=str(save_dir), name_prefix="checkpoint")
    callbacks.append(EveryNTimesteps(n_steps=interval, callback=checkpoint_cb))
    return callbacks


def run_experiment(config_path: str, output_dir: str = "results", seed: int = 1):
    """Run a single experiment based on config file."""

    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)

    exp_name = config['name']
    extrinsic_rewards = config['extrinsic_rewards']
    hyperparams = config['hyperparameters']
    env_config = config['env_config']
    wandb_config = config.get('wandb_config', {})

    print(f"\n{'='*60}")
    print(f"Running SB3 Experiment: {exp_name}")
    print(f"Description: {config['description']}")
    print(f"Extrinsic Rewards: {extrinsic_rewards}")
    print(f"{'='*60}\n")

    # Create experiment-specific output directory
    exp_output_dir = Path(output_dir) / exp_name
    exp_output_dir.mkdir(parents=True, exist_ok=True)

    # Save config to output directory
    config_copy_path = exp_output_dir / 'experiment_config.json'
    with open(config_copy_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Setup checkpoint and log directories
    checkpoint_dir = exp_output_dir / "checkpoints"
    tensorboard_log_dir = exp_output_dir / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    # Register environment with extrinsic rewards
    env_id = env_config['env_id']
    task = env_config['task']
    room_size = env_config['room_size']
    max_steps = env_config['max_steps']

    env_kwargs = {
        "room_size": room_size,
        "max_steps": max_steps,
        "extrinsic_rewards": extrinsic_rewards
    }

    if env_id not in env_list:
        register(
            id=env_id,
            entry_point=f'mini_behavior.envs:{task}Env',
            kwargs=env_kwargs
        )

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize wandb if available
    wandb_run = None
    if WANDB_AVAILABLE:
        # Combine all configs for wandb
        full_wandb_config = {
            "experiment_name": exp_name,
            "extrinsic_rewards": extrinsic_rewards,
            **hyperparams,
            **env_config,
        }

        wandb_run = wandb.init(
            project=wandb_config.get('project', 'sb3-noveld-extrinsic'),
            name=exp_name,
            group=wandb_config.get('group', 'extrinsic_only'),
            tags=wandb_config.get('tags', ['sb3', 'noveld', 'extrinsic']),
            config=full_wandb_config,
        )
        print(f"Wandb run initialized: {wandb_run.name}")

    # Create vectorized environments
    num_envs = hyperparams['num_envs']
    vec_env = make_vec_env(env_id, env_kwargs, seed, num_envs)

    # Create NovelD_PPO model
    print("\n=== Creating SB3 NovelD_PPO Model ===")
    model = NovelD_PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=hyperparams['learning_rate'],
        n_steps=hyperparams['n_steps'],
        batch_size=hyperparams['batch_size'],
        n_epochs=hyperparams['n_epochs'],
        gamma=hyperparams['gamma'],
        gae_lambda=hyperparams['gae_lambda'],
        clip_range=hyperparams['clip_range'],
        ent_coef=hyperparams['ent_coef'],
        vf_coef=hyperparams['vf_coef'],
        max_grad_norm=hyperparams['max_grad_norm'],
        int_coef=hyperparams['int_coef'],
        ext_coef=hyperparams['ext_coef'],
        int_gamma=hyperparams['int_gamma'],
        update_proportion=hyperparams['update_proportion'],
        tensorboard_log=str(tensorboard_log_dir),
        device=device,
        verbose=1,
    )

    # Configure logger with wandb
    logger = utils.configure_logger(
        model.verbose,
        model.tensorboard_log,
        tb_log_name=exp_name,
        reset_num_timesteps=True
    )
    model.set_logger(logger)

    if WANDB_AVAILABLE and wandb_run is not None:
        if not any(isinstance(fmt, WandbOutputFormat) for fmt in model.logger.output_formats):
            model.logger.output_formats.append(WandbOutputFormat())

    # Setup callbacks
    checkpoint_interval = int(os.environ.get("GIF_LOG_INTERVAL", 500_000))
    callbacks = build_checkpoint_callbacks(checkpoint_dir, checkpoint_interval)

    # Add action distribution callback
    if hasattr(vec_env.action_space, "nvec"):
        action_dims = vec_env.action_space.nvec.tolist()
        log_interval = num_envs * hyperparams['n_steps']
        callbacks.append(ActionDistributionCallback(action_dims, log_interval=log_interval))
    # Log rollout GIFs to wandb at the same interval as checkpoints
    callbacks.append(
        GifLoggerCallback(
            env_id=env_id,
            env_kwargs=env_kwargs,
            log_interval=checkpoint_interval,
            gif_steps=200,
            seed=seed,
            deterministic=False,
            verbose=1,
        )
    )

    callback_arg: Optional[BaseCallback] = None
    if callbacks:
        callback_arg = callbacks[0] if len(callbacks) == 1 else CallbackList(callbacks)

    # Train the model
    print("\n=== Starting Training ===")
    print(f"Total timesteps: {hyperparams['total_timesteps']:,}")
    print(f"Num envs: {num_envs}")
    print(f"Steps per env: {hyperparams['n_steps']}")
    print(f"Batch size: {hyperparams['batch_size']}")
    print(f"Intrinsic coef: {hyperparams['int_coef']}")
    print(f"Extrinsic coef: {hyperparams['ext_coef']}")
    print(f"Entropy coef: {hyperparams['ent_coef']}")
    print(f"\nExtrinsic rewards active:")
    for reward_type, value in extrinsic_rewards.items():
        print(f"  {reward_type}: {value}")
    print()

    model.learn(total_timesteps=hyperparams['total_timesteps'], callback=callback_arg)

    # Save final model
    final_model_path = exp_output_dir / "final_model"
    model.save(str(final_model_path))
    print(f"\nTraining complete. Final model saved to {final_model_path.with_suffix('.zip')}")

    # Create success marker
    with open(exp_output_dir / 'SUCCESS', 'w') as f:
        f.write(f"Experiment {exp_name} completed successfully\n")

    # Close environment and wandb
    vec_env.close()
    if WANDB_AVAILABLE and wandb_run is not None:
        wandb.finish()

    print(f"\nResults saved to: {exp_output_dir}")
    return exp_output_dir


def main():
    parser = argparse.ArgumentParser(
        description='Run a single SB3 NovelD experiment with extrinsic rewards'
    )
    parser.add_argument(
        'config',
        help='Path to experiment configuration JSON file'
    )
    parser.add_argument(
        '--output-dir',
        default='results',
        help='Directory to save results (default: results)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='Random seed (default: 1)'
    )

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
