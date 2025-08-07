#!/usr/bin/env python3
"""
Run a single APT experiment from a config file.
Usage: python run_single_experiment.py configs/exp_000_default.json
"""

import os
import sys
import json
import argparse
import torch
import gym
import subprocess
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from algorithms.APT_PPO import APT_PPO
from mini_behavior.register import register
from env_wrapper import CustomObservationWrapper


def make_env(env_id: str, seed: int, idx: int, env_kwargs: dict):
    """Create a callable that returns an environment instance."""
    def thunk():
        env = gym.make(env_id, **env_kwargs)
        env = CustomObservationWrapper(env)
        env.seed(seed + idx)
        return env
    return thunk


def init_env(env_name: str, num_envs: int, seed: int, env_kwargs: dict):
    """Initialize a vectorized (synchronous) environment."""
    return gym.vector.SyncVectorEnv(
        [make_env(env_name, seed, i, env_kwargs) for i in range(num_envs)]
    )


def run_experiment(config_path):
    """Run a single experiment from config file."""
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print("=" * 70)
    print(f"Running Experiment: {config['experiment_name']}")
    print(f"Description: {config['description']}")
    print("=" * 70)
    
    # Extract hyperparameters
    hp = config['hyperparameters']
    
    # Set up environment
    ENV_NAME = 'MiniGrid-MultiToy-8x8-N2-v0'
    ENV_KWARGS = {"room_size": 8, "max_steps": 1000}
    
    # Register environment
    register(
        id=ENV_NAME,
        entry_point='mini_behavior.envs.multitoy:MultiToyEnv'
    )
    
    # Device setup
    device = torch.device(config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(f"APT_Experiments_v2/results/{config['experiment_name']}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create checkpoints directory for APT_PPO
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    
    # Path for intrinsic rewards CSV
    intrinsic_rewards_csv_path = output_dir / "intrinsic_rewards_log.csv"
    
    # Save experiment config to results directory
    config_save_path = output_dir / "experiment_config.json"
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Config saved to: {config_save_path}")
    
    # Initialize environment
    seed = config.get('seed', 1)
    env = init_env(ENV_NAME, hp['num_envs'], seed, ENV_KWARGS)
    
    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)
    
    # Initialize APT_PPO agent with all hyperparameters
    agent = APT_PPO(
        env=env,
        env_id=ENV_NAME,
        env_kwargs=ENV_KWARGS,
        save_dir=str(checkpoints_dir),  # Use checkpoints subdirectory
        device=str(device),
        save_freq=500000,  # Save every 500k steps
        test_steps=200,    # 200 steps per test episode
        total_timesteps=hp['total_timesteps'],
        learning_rate=hp['learning_rate'],
        num_envs=hp['num_envs'],
        num_eps=10,        # 10 test episodes per checkpoint
        num_steps=hp['num_steps'],
        anneal_lr=True,
        gamma=hp['gamma'],
        gae_lambda=hp['gae_lambda'],
        num_minibatches=hp['num_minibatches'],
        update_epochs=hp['update_epochs'],
        norm_adv=True,
        clip_coef=hp['clip_coef'],
        clip_vloss=True,
        ent_coef=hp['ent_coef'],
        vf_coef=hp['vf_coef'],
        max_grad_norm=hp['max_grad_norm'],
        target_kl=None,
        int_coef=hp['int_coef'],
        ext_coef=hp['ext_coef'],
        int_gamma=hp['int_gamma'],
        k=hp['k'],
        c=hp['c'],
        replay_buffer_size=hp['replay_buffer_size'],
        batch_size=hp['apt_batch_size'],
        wandb_project=config.get('wandb_project', 'APT_PPO_V2'),
        wandb_entity=config.get('wandb_entity', None),
        wandb_run_name=config['experiment_name'],
        use_wandb=config.get('use_wandb', True),
        aggregation_method=hp.get('aggregation_method', 'mean'),
        intrinsic_rewards_csv_path=str(intrinsic_rewards_csv_path)
    )
    
    # Print hyperparameters
    print("\nHyperparameters:")
    print("-" * 40)
    for key, value in hp.items():
        print(f"  {key}: {value}")
    print("-" * 40)
    print()
    
    # Run training
    try:
        agent.train()
        
        # Run visualization on checkpoint activity logs
        activity_logs_dir = checkpoints_dir / "activity_logs"
        if activity_logs_dir.exists():
            print(f"\n{'=' * 70}")
            print("Running checkpoint activity visualization...")
            print(f"{'=' * 70}")
            
            # Run visualize_checkpoints.py
            viz_script = Path(__file__).parent.parent / "post_processing" / "visualize_checkpoints.py"
            viz_output_dir = output_dir / "visualizations"
            
            cmd = [
                sys.executable,
                str(viz_script),
                str(activity_logs_dir),
                "--output-dir", str(viz_output_dir),
                "--top-n", "15"
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                print(result.stdout)
                print(f"Visualizations saved to: {viz_output_dir}")
            except subprocess.CalledProcessError as e:
                print(f"Warning: Visualization script failed: {e}")
                print(f"Error output: {e.stderr}")
        else:
            print(f"Warning: No activity logs found at {activity_logs_dir}")
        
        # Mark experiment as successful
        success_file = output_dir / "SUCCESS"
        success_file.touch()
        print(f"\n{'=' * 70}")
        print(f"Experiment {config['experiment_name']} completed successfully!")
        print(f"Results saved to: {output_dir}")
        print(f"{'=' * 70}")
        
    except Exception as e:
        print(f"\n{'=' * 70}")
        print(f"Experiment {config['experiment_name']} failed with error:")
        print(f"{e}")
        print(f"{'=' * 70}")
        
        # Save error log
        error_file = output_dir / "ERROR.txt"
        with open(error_file, 'w') as f:
            f.write(f"Experiment failed with error:\n{e}\n")
        raise
    
    finally:
        # Clean up
        env.close()


def main():
    parser = argparse.ArgumentParser(
        description='Run a single APT experiment from a config file'
    )
    parser.add_argument(
        'config_path',
        type=str,
        help='Path to experiment configuration JSON file'
    )
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not os.path.exists(args.config_path):
        print(f"Error: Config file not found: {args.config_path}")
        sys.exit(1)
    
    # Run the experiment
    run_experiment(args.config_path)


if __name__ == "__main__":
    main()