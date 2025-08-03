#!/usr/bin/env python3
"""
Run a single APT experiment with specified hyperparameters.
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path

# Add parent directory to path to import algorithms
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import gym
from algorithms.APT_PPO import APT_PPO
from env_wrapper import CustomObservationWrapper
from mini_behavior.envs.multitoy import MultiToyEnv

def make_env(idx):
    """Create a single environment instance."""
    def thunk():
        env = MultiToyEnv(room_size=8)
        env = CustomObservationWrapper(env)
        env.seed(1 + idx)  # Seed each environment differently
        return env
    return thunk

def create_env():
    """Create the MultiToy environment with proper wrapper."""
    # Use SyncVectorEnv like test_APT.py does
    vec_env = gym.vector.SyncVectorEnv([make_env(i) for i in range(8)])  # 8 parallel environments
    return vec_env

def run_experiment(config, output_dir):
    """Run a single experiment with the given configuration."""
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config to output directory
    with open(output_dir / "experiment_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create checkpoint directory for this experiment
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Create a symlink from the default checkpoints folder to our experiment folder
    default_checkpoint_path = Path("checkpoints")
    
    # Backup existing checkpoints if they exist
    if default_checkpoint_path.exists() and not default_checkpoint_path.is_symlink():
        backup_path = Path("checkpoints_backup")
        if backup_path.exists():
            shutil.rmtree(backup_path)
        shutil.move(default_checkpoint_path, backup_path)
        print(f"Backed up existing checkpoints to {backup_path}")
    
    # Remove existing symlink if it exists
    if default_checkpoint_path.exists():
        if default_checkpoint_path.is_symlink():
            default_checkpoint_path.unlink()
        else:
            print(f"Warning: {default_checkpoint_path} exists and is not a symlink")
    
    # Create symlink to experiment checkpoint directory
    default_checkpoint_path.symlink_to(checkpoint_dir.absolute())
    print(f"Created checkpoint symlink: {default_checkpoint_path} -> {checkpoint_dir.absolute()}")
    
    try:
        # Create environment
        env = create_env()
        
        # Register the environment if needed
        from mini_behavior.register import register
        env_id = 'MiniGrid-MultiToy-8x8-N2-v0'
        try:
            gym.make(env_id)
        except:
            register(
                id=env_id,
                entry_point='mini_behavior.envs.multitoy:MultiToyEnv',
            )
        
        # Create APT_PPO agent with experiment hyperparameters
        agent = APT_PPO(
            env=env,
            env_id=env_id,
            env_kwargs={"room_size": 8},
            save_dir=str(output_dir),
            device="cuda" if torch.cuda.is_available() else "cpu",
            total_timesteps=config.get('total_timesteps', 2500000),
            learning_rate=config['learning_rate'],
            num_envs=8,
            num_steps=config.get('n_steps', 125),
            gamma=config['gamma'],
            gae_lambda=config['gae_lambda'],
            num_minibatches=4,
            update_epochs=config.get('n_epochs', 4),
            clip_coef=config.get('clip_range', 0.2),
            ent_coef=config['ent_coef'],
            vf_coef=config['vf_coef'],
            max_grad_norm=config['max_grad_norm'],
            k=config['k'],
            int_gamma=config['int_gamma'],
            int_coef=config.get('int_coef', 1.0),
            ext_coef=config.get('ext_coef', 0.0),
            aggregation_method=config.get('aggregation_method', 'mean'),
            batch_size=config.get('apt_batch_size', 1024),  # APT batch size for k-NN
            wandb_run_name=config['experiment_id'],  # Use experiment ID as wandb run name
            use_wandb=True  # Enable wandb for experiment runs
        )
        
        # Print experiment info
        print(f"\\nStarting experiment: {config['experiment_id']}")
        print(f"Ablated parameter: {config['ablated_param']} = {config['ablated_value']}")
        print(f"Total timesteps: {config.get('total_timesteps', 2500000)}")
        
        # Train the agent
        agent.train()
        
        # Note: APT_PPO saves checkpoints automatically during training
        
        # Create success marker
        with open(output_dir / "SUCCESS", 'w') as f:
            f.write(f"Experiment {config['experiment_id']} completed successfully\\n")
        
        print(f"\\nExperiment {config['experiment_id']} completed successfully!")
        
    finally:
        # Clean up symlink
        if default_checkpoint_path.exists() and default_checkpoint_path.is_symlink():
            default_checkpoint_path.unlink()
            print(f"Removed checkpoint symlink")
        
        # Restore backed up checkpoints if they exist
        backup_path = Path("checkpoints_backup")
        if backup_path.exists() and not default_checkpoint_path.exists():
            shutil.move(backup_path, default_checkpoint_path)
            print(f"Restored backed up checkpoints")
        
        env.close()

def main():
    parser = argparse.ArgumentParser(description='Run a single APT experiment')
    parser.add_argument('config_file', type=str, help='Path to experiment config JSON file')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: results/experiment_id)')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config_file, 'r') as f:
        config = json.load(f)
    
    # Set output directory
    if args.output_dir is None:
        output_dir = Path("results") / config['experiment_id']
    else:
        output_dir = Path(args.output_dir)
    
    # Run experiment
    run_experiment(config, output_dir)

if __name__ == "__main__":
    main()