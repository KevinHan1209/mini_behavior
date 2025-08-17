#!/usr/bin/env python3
"""
Run a single NovelD experiment with specified hyperparameters
"""
import sys
import os
import json
import argparse
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.NovelD_PPO import NovelD_PPO
import torch
from mini_behavior.register import register

def run_experiment(config_path, output_dir='results'):
    """Run a single experiment based on config file"""
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    exp_name = config['name']
    hyperparams = config['hyperparameters']
    
    print(f"\n{'='*60}")
    print(f"Running Experiment: {exp_name}")
    print(f"Ablated Parameter: {config['ablated_param']} = {config['ablated_value']}")
    print(f"{'='*60}\n")
    
    # Create experiment-specific output directory
    exp_output_dir = os.path.join(output_dir, exp_name)
    os.makedirs(exp_output_dir, exist_ok=True)
    
    # Save config to output directory
    config_copy_path = os.path.join(exp_output_dir, 'experiment_config.json')
    shutil.copy2(config_path, config_copy_path)
    
    # Register environment
    env_id = 'MiniGrid-MultiToy-8x8-N2-v0'
    TASK = 'MultiToy'
    ROOM_SIZE = 8
    MAX_STEPS = 1000
    env_kwargs = {"room_size": ROOM_SIZE, "max_steps": MAX_STEPS}
    register(
        id=env_id,
        entry_point=f'mini_behavior.envs:{TASK}Env',
        kwargs=env_kwargs
    )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create temporary checkpoint directory for this experiment
    original_checkpoint_dir = "checkpoints"
    exp_checkpoint_dir = os.path.join(exp_output_dir, "checkpoints")
    
    # Temporarily rename the original checkpoints directory if it exists
    checkpoint_backup = None
    if os.path.exists(original_checkpoint_dir):
        checkpoint_backup = f"{original_checkpoint_dir}_backup"
        # Remove any existing backup first
        if os.path.exists(checkpoint_backup):
            if os.path.islink(checkpoint_backup):
                os.unlink(checkpoint_backup)
            else:
                shutil.rmtree(checkpoint_backup)
        os.rename(original_checkpoint_dir, checkpoint_backup)
    
    try:
        # Create symlink to experiment checkpoint directory
        os.makedirs(exp_checkpoint_dir, exist_ok=True)
        os.symlink(os.path.abspath(exp_checkpoint_dir), original_checkpoint_dir)
        
        # Create agent with experiment hyperparameters
        agent = NovelD_PPO(
            env_id,
            device,
            total_timesteps=hyperparams['total_timesteps'],
            learning_rate=hyperparams['learning_rate'],
            num_envs=hyperparams['num_envs'],
            num_steps=hyperparams['num_steps'],
            gamma=hyperparams['gamma'],
            gae_lambda=hyperparams['gae_lambda'],
            num_minibatches=hyperparams['num_minibatches'],
            update_epochs=hyperparams['update_epochs'],
            clip_coef=hyperparams['clip_coef'],
            ent_coef=hyperparams['ent_coef'],
            vf_coef=hyperparams['vf_coef'],
            max_grad_norm=hyperparams['max_grad_norm'],
            int_coef=hyperparams['int_coef'],
            ext_coef=hyperparams['ext_coef'],
            int_gamma=hyperparams['int_gamma'],
            alpha=hyperparams['alpha'],
            update_proportion=hyperparams['update_proportion'],
            wandb_project="NovelD_Hyperparameter_Ablation",  # Descriptive project name
            wandb_run_name=exp_name  # Use experiment name as wandb run name
        )
        
        # Train agent
        print(f"\nStarting training with hyperparameters:")
        print(f"  alpha: {hyperparams['alpha']}")
        print(f"  update_proportion: {hyperparams['update_proportion']}")
        print(f"  int_gamma: {hyperparams['int_gamma']}")
        print(f"  ent_coef: {hyperparams['ent_coef']}")
        print(f"  total_timesteps: {hyperparams['total_timesteps']:,}")
        
        agent.train()
        
        print(f"\nTraining completed for {exp_name}")
        
        # Create success marker
        with open(os.path.join(exp_output_dir, 'SUCCESS'), 'w') as f:
            f.write(f"Experiment {exp_name} completed successfully\n")
        
    finally:
        # Clean up symlink
        if os.path.islink(original_checkpoint_dir):
            os.unlink(original_checkpoint_dir)
        
        # Restore original checkpoints directory
        if checkpoint_backup and os.path.exists(checkpoint_backup):
            os.rename(checkpoint_backup, original_checkpoint_dir)
    
    return exp_output_dir

def main():
    parser = argparse.ArgumentParser(description='Run a single NovelD experiment')
    parser.add_argument('config', help='Path to experiment configuration JSON file')
    parser.add_argument('--output-dir', default='results',
                       help='Directory to save results (default: results)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Error: Config file '{args.config}' not found")
        sys.exit(1)
    
    try:
        output_dir = run_experiment(args.config, args.output_dir)
        print(f"\nResults saved to: {output_dir}")
    except Exception as e:
        print(f"\nError running experiment: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()