#!/usr/bin/env python3
"""
Run a single SB3 vanilla PPO experiment with extrinsic rewards only.
This serves as a baseline comparison for NovelD experiments.
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
import torch
from gym import Env

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Proceeding without wandb logging.")

from stable_baselines3 import PPO
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


def make_vec_env(env_id: str, env_kwargs: Optional[Dict[str, object]], seed: int, num_envs: int) -> DummyVecEnv:
    """Create a monitored DummyVecEnv with CustomObservationWrapper for PPO training."""

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
    print(f"Running SB3 Vanilla PPO Experiment: {exp_name}")
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
            "algorithm": "PPO",
            **hyperparams,
            **env_config,
        }

        wandb_run = wandb.init(
            project=wandb_config.get('project', 'sb3-ppo-extrinsic'),
            name=exp_name,
            group=wandb_config.get('group', 'ppo_baseline'),
            tags=wandb_config.get('tags', ['sb3', 'ppo', 'extrinsic']),
            config=full_wandb_config,
        )
        print(f"Wandb run initialized: {wandb_run.name}")

    # Create vectorized environments
    num_envs = hyperparams['num_envs']
    vec_env = make_vec_env(env_id, env_kwargs, seed, num_envs)

    # Create vanilla PPO model
    print("\n=== Creating SB3 PPO Model ===")
    model = PPO(
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
    checkpoint_interval = 500_000
    callbacks = build_checkpoint_callbacks(checkpoint_dir, checkpoint_interval)

    # Add action distribution callback
    if hasattr(vec_env.action_space, "nvec"):
        action_dims = vec_env.action_space.nvec.tolist()
        log_interval = num_envs * hyperparams['n_steps']
        callbacks.append(ActionDistributionCallback(action_dims, log_interval=log_interval))

    callback_arg: Optional[BaseCallback] = None
    if callbacks:
        callback_arg = callbacks[0] if len(callbacks) == 1 else CallbackList(callbacks)

    # Train the model
    print("\n=== Starting Training ===")
    print(f"Total timesteps: {hyperparams['total_timesteps']:,}")
    print(f"Num envs: {num_envs}")
    print(f"Steps per env: {hyperparams['n_steps']}")
    print(f"Batch size: {hyperparams['batch_size']}")
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
        description='Run a single SB3 vanilla PPO experiment with extrinsic rewards'
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
