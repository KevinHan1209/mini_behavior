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

def create_env():
    """Create the MiniGrid environment."""
    env = gym.make('MiniGrid-MultiToy-8x8-N2-v0', room_size=8, max_steps=1000)
    return env

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
        
        # Create APT_PPO agent with experiment hyperparameters
        agent = APT_PPO(
            env,
            k=config['k'],
            int_gamma=config['int_gamma'],
            learning_rate=config['learning_rate'],
            n_steps=config['n_steps'],
            batch_size=config['batch_size'],
            n_epochs=config['n_epochs'],
            gamma=config['gamma'],
            gae_lambda=config['gae_lambda'],
            clip_range=config['clip_range'],
            ent_coef=config['ent_coef'],
            vf_coef=config['vf_coef'],
            max_grad_norm=config['max_grad_norm'],
            target_kl=None,
            verbose=1
        )
        
        # Print experiment info
        print(f"\\nStarting experiment: {config['experiment_id']}")
        print(f"Ablated parameter: {config['ablated_param']} = {config['ablated_value']}")
        print(f"Total timesteps: {config['total_timesteps']}")
        
        # Train the agent
        agent.learn(total_timesteps=config['total_timesteps'])
        
        # Save final model
        agent.save(str(output_dir / "final_model"))
        
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