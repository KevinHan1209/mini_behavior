# test/test_NovelD.py
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure the repository root is on sys.path for relative imports.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gym
import numpy as np
import torch
from gym import Env

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from array2gif import write_gif  # type: ignore

    GIF_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    GIF_AVAILABLE = False

    def write_gif(*_args, **_kwargs):
        raise RuntimeError("array2gif is not installed. Install with `pip install array2gif` to enable GIF export.")

from stable_baselines3 import NovelD_PPO
from stable_baselines3.common import utils
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EveryNTimesteps
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import KVWriter

from env_wrapper import CustomObservationWrapper
from mini_behavior.register import env_list, register
from mini_behavior.utils.states_base import RelativeObjectState

# Default training configuration aligned with the legacy runner.
TOTAL_TIMESTEPS = 2_500_000
NUM_ENVS = 8
ROLLOUT_STEPS = 125
NUM_MINIBATCHES = 4
BATCH_SIZE = (NUM_ENVS * ROLLOUT_STEPS) // NUM_MINIBATCHES
CHECKPOINT_INTERVAL = 500_000
CHECKPOINT_DIR = Path("checkpoints_sb3")
TENSORBOARD_LOG_DIR = Path("logs") / "noveld_sb3_multitoy"


def count_binary_flags(env: Env) -> int:
    """Count the total number of non-relative (binary) state flags exposed by the environment."""
    num_flags = 0
    for obj_list in env.objs.values():
        for obj in obj_list:
            for state in obj.states.values():
                if not isinstance(state, RelativeObjectState):
                    num_flags += 1
    return num_flags


def extract_binary_flags(obs: np.ndarray, env: Env) -> np.ndarray:
    """
    Extract the binary flag portion of an observation.

    The observation is ordered as:
      - agent_x, agent_y, agent_dir
      - For each object: pos_x, pos_y, followed by binary flags
    """
    flags: List[float] = []
    index = 3  # Skip agent position/direction
    for obj_list in env.objs.values():
        for obj in obj_list:
            index += 2  # Skip stored position
            for state in obj.states.values():
                if not isinstance(state, RelativeObjectState):
                    flags.append(float(obs[index]) if index < len(obs) else 0.0)
                    index += 1
    return np.asarray(flags, dtype=np.float32)


def generate_flag_mapping(env: Env) -> List[Dict[str, object]]:
    """Map each binary flag index back to the underlying object's state metadata."""
    mapping: List[Dict[str, object]] = []
    for obj_type, obj_list in env.objs.items():
        for obj_index, obj in enumerate(obj_list):
            for state_name, state in obj.states.items():
                if not isinstance(state, RelativeObjectState):
                    mapping.append(
                        {
                            "object_type": obj_type,
                            "object_index": obj_index,
                            "state_name": state_name,
                        }
                    )
    return mapping


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


def train_agent(
    env_id: str,
    env_kwargs: Optional[Dict[str, object]],
    device: torch.device,
    total_timesteps: int = TOTAL_TIMESTEPS,
    wandb_run=None,
) -> NovelD_PPO:
    """Train the Stable-Baselines3 NovelD_PPO agent on MultiToy."""

    print("\n=== Starting SB3 NovelD_PPO Training ===")
    print(f"Environment: {env_id}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Vectorised envs: {NUM_ENVS}, rollout steps per env: {ROLLOUT_STEPS}, batch size: {BATCH_SIZE}")

    vec_env = make_vec_env(env_id, env_kwargs, seed=1, num_envs=NUM_ENVS)

    model = NovelD_PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=ROLLOUT_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        int_coef=1.0,
        ext_coef=0.0,
        int_gamma=0.99,
        update_proportion=0.25,
        tensorboard_log=str(TENSORBOARD_LOG_DIR),
        device=device,
        verbose=1,
    )

    logger = utils.configure_logger(model.verbose, model.tensorboard_log, tb_log_name="noveld_sb3", reset_num_timesteps=True)
    model.set_logger(logger)
    if WANDB_AVAILABLE and wandb_run is not None:
        if not any(isinstance(fmt, WandbOutputFormat) for fmt in model.logger.output_formats):
            model.logger.output_formats.append(WandbOutputFormat())

    callbacks = build_checkpoint_callbacks(CHECKPOINT_DIR, CHECKPOINT_INTERVAL)
    if hasattr(vec_env.action_space, "nvec"):
        action_dims = vec_env.action_space.nvec.tolist()
        log_interval = NUM_ENVS * ROLLOUT_STEPS
        callbacks.append(ActionDistributionCallback(action_dims, log_interval=log_interval))

    callback_arg: Optional[BaseCallback] = None
    if callbacks:
        callback_arg = callbacks[0] if len(callbacks) == 1 else CallbackList(callbacks)

    model.learn(total_timesteps=total_timesteps, callback=callback_arg)

    final_path = CHECKPOINT_DIR / "noveld_sb3_multitoy_final"
    model.save(str(final_path))
    print(f"\nTraining complete. Final model saved to {final_path.with_suffix('.zip')}")

    vec_env.close()
    return model


def test_agent(
    env_id: str,
    model: NovelD_PPO,
    device: torch.device,
    env_kwargs: Optional[Dict[str, object]],
    num_episodes: int = 5,
    max_steps_per_episode: int = 200,
    wandb_run=None,
) -> None:
    """Evaluate a trained NovelD_PPO agent and optionally log to Weights & Biases."""
    print(f"\n=== Evaluating Trained Agent ({num_episodes} episodes) ===")

    new_run_created = False
    active_run = wandb_run
    if WANDB_AVAILABLE and active_run is None:
        active_run = wandb.init(
            project="noveld-ppo-sb3-multitoy",
            config={
                "env_id": env_id,
                "num_episodes": num_episodes,
                "max_steps": max_steps_per_episode,
                "int_coef": model.int_coef,
                "ext_coef": model.ext_coef,
                "mode": "evaluation",
            },
        )
        new_run_created = True

    if not WANDB_AVAILABLE:
        print("wandb not available; proceeding without external logging.")

    test_env = gym.make(env_id, **(env_kwargs or {}))
    test_env = CustomObservationWrapper(test_env)

    env_unwrapped = getattr(test_env, "env", test_env)
    num_binary_flags = count_binary_flags(env_unwrapped)
    flag_mapping = generate_flag_mapping(env_unwrapped)

    if WANDB_AVAILABLE and active_run is not None:
        mapping_table = wandb.Table(columns=["flag_id", "object_type", "object_index", "state_name"])
        for idx, mapping in enumerate(flag_mapping):
            mapping_table.add_data(idx, mapping["object_type"], mapping["object_index"], mapping["state_name"])
        wandb.log({"flag_mapping": mapping_table})

    policy_device = torch.device(model.device)

    for episode in range(num_episodes):
        try:
            obs = test_env.reset()
        except TypeError:
            obs, _ = test_env.reset()
        done = False
        steps = 0
        total_reward = 0.0
        novelty_values: List[float] = []
        frames: List[np.ndarray] = []
        activity = [0] * num_binary_flags
        prev_flags: Optional[np.ndarray] = None

        while not done and steps < max_steps_per_episode:
            frame = None
            if hasattr(test_env, "render"):
                try:
                    frame = test_env.render()
                except TypeError:
                    frame = test_env.render(mode="rgb_array")
            if isinstance(frame, np.ndarray):
                frames.append(np.moveaxis(frame, 2, 0))

            action, _ = model.predict(obs, deterministic=False)
            action_to_env = np.asarray(action, dtype=np.int64)
            next_outcome = test_env.step(action_to_env)
            if len(next_outcome) == 5:
                next_obs, reward, terminated, truncated, _ = next_outcome
                done = bool(terminated or truncated)
            else:
                next_obs, reward, done, _ = next_outcome

            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=policy_device).unsqueeze(0)
            with torch.no_grad():
                value_tensor = model.policy.predict_values(obs_tensor)
            value_estimate = float(value_tensor.squeeze().cpu().item())

            novelty_val = None
            if getattr(model, "_intrinsic_enabled", False):
                novelty = model._compute_intrinsic_reward(np.asarray(obs, dtype=np.float32).reshape(1, -1))
                novelty_val = float(novelty.squeeze().item())
                novelty_values.append(novelty_val)

            obs = next_obs
            total_reward += float(reward)
            steps += 1

            current_flags = extract_binary_flags(np.asarray(obs, dtype=np.float32), env_unwrapped)
            if prev_flags is not None and len(current_flags) == len(prev_flags):
                differences = (current_flags != prev_flags).astype(int)
                activity = [a + d for a, d in zip(activity, differences.tolist())]
            prev_flags = current_flags

            if WANDB_AVAILABLE and active_run is not None:
                wandb.log(
                    {
                        "step_reward": reward,
                        "step_value_estimate": value_estimate,
                        "step_novelty": novelty_val,
                        "episode_index": episode,
                        "step_index": steps,
                    }
                )

        if frames and GIF_AVAILABLE:
            gif_path = f"episode_{episode + 1}.gif"
            write_gif(np.array(frames), gif_path, fps=1)
            if WANDB_AVAILABLE and active_run is not None:
                wandb.log({f"episode_{episode + 1}_replay": wandb.Video(gif_path, fps=10, format="gif")})

        if WANDB_AVAILABLE and active_run is not None:
            activity_table = wandb.Table(
                columns=["flag_id", "object_type", "object_index", "state_name", "activity_count"]
            )
            for idx, count in enumerate(activity):
                mapping = flag_mapping[idx]
                activity_table.add_data(idx, mapping["object_type"], mapping["object_index"], mapping["state_name"], count)
            wandb.log(
                {
                    "episode_total_reward": total_reward,
                    "episode_length": steps,
                    "episode_mean_novelty": np.mean(novelty_values) if novelty_values else None,
                    f"episode_{episode + 1}_activity": activity_table,
                    "episode_index": episode,
                }
            )

        print(f"\nEpisode {episode + 1}/{num_episodes}: reward={total_reward:.2f}, steps={steps}")

    test_env.close()
    if WANDB_AVAILABLE and active_run is not None and new_run_created:
        wandb.finish()


def main():
    env_id = "MiniGrid-MultiToy-8x8-N2-v0"
    task = "MultiToy"
    room_size = 8
    max_steps = 1000
    env_kwargs: Dict[str, object] = {"room_size": room_size, "max_steps": max_steps}

    if env_id not in env_list:
        register(
            id=env_id,
            entry_point=f"mini_behavior.envs:{task}Env",
            kwargs=env_kwargs,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb_run = None
    if WANDB_AVAILABLE:
        wandb_run = wandb.init(
            project="noveld-ppo-sb3-multitoy",
            config={
                "env_id": env_id,
                "room_size": room_size,
                "max_steps": max_steps,
                "total_timesteps": TOTAL_TIMESTEPS,
                "num_envs": NUM_ENVS,
                "rollout_steps": ROLLOUT_STEPS,
                "batch_size": BATCH_SIZE,
                "int_coef": 1.0,
                "ext_coef": 0.0,
            },
        )
    try:
        model = train_agent(env_id, env_kwargs, device, wandb_run=wandb_run)
        test_agent(env_id, model, device, env_kwargs, wandb_run=wandb_run)
    finally:
        if WANDB_AVAILABLE and wandb_run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
