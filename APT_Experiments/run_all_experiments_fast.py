#!/usr/bin/env python3
"""
Faster version of run_all_experiments that runs experiments in the same process
to avoid repeated import overhead.
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime

# Add parent directory to path ONCE
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import heavy modules ONCE at the beginning
import torch
import gym
from algorithms.APT_PPO import APT_PPO

def create_env():
    """Create the MiniGrid environment."""
    env = gym.make('MiniGrid-MultiToy-8x8-N2-v0', room_size=8, max_steps=1000)
    return env

def run_experiment_inline(config, output_dir):
    """Run a single experiment inline (not as subprocess)."""
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config to output directory
    with open(output_dir / "experiment_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create checkpoint directory for this experiment
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Handle checkpoint symlink
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
        
        # Clean up to free memory
        del agent
        env.close()
        
        # Force garbage collection
        import gc
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return True
        
    except Exception as e:
        print(f"Error in experiment {config['experiment_id']}: {e}")
        import traceback
        traceback.print_exc()
        return False
        
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

def generate_configs():
    """Generate experiment configurations."""
    print("\\n" + "="*60)
    print("Generating experiment configurations...")
    print("="*60)
    
    # Import and run generate_configs inline
    from generate_configs import generate_experiment_configs, save_configs
    configs = generate_experiment_configs()
    save_configs(configs)
    return True

def run_experiments(subset=None, quick_test=False):
    """Run all experiments in the same process."""
    print("\\n" + "="*60)
    print("Running experiments (fast mode - same process)...")
    print("="*60)
    
    # Load all experiment configs
    with open("configs/all_experiments.json", 'r') as f:
        all_configs = json.load(f)
    
    # Filter experiments if subset specified
    if subset:
        filtered_configs = {
            exp_id: config for exp_id, config in all_configs.items()
            if config['ablated_param'] == subset
        }
        print(f"\\nRunning subset: {subset} ({len(filtered_configs)} experiments)")
        all_configs = filtered_configs
    
    # Quick test mode - reduce timesteps
    if quick_test:
        print("\\nQUICK TEST MODE: Reducing timesteps to 100,000")
        for config in all_configs.values():
            config['total_timesteps'] = 100000
    
    total = len(all_configs)
    completed = 0
    skipped = 0
    failed = 0
    
    for i, (exp_id, config) in enumerate(all_configs.items(), 1):
        print(f"\\n{'='*60}")
        print(f"Experiment {i}/{total}: {exp_id}")
        print(f"Ablating: {config['ablated_param']} = {config['ablated_value']}")
        print(f"{'='*60}")
        
        # Check if already completed
        output_dir = Path("results") / exp_id
        if (output_dir / "SUCCESS").exists():
            print(f"Skipping {exp_id} - already completed")
            skipped += 1
            continue
        
        # Run experiment inline
        start_time = datetime.now()
        success = run_experiment_inline(config, output_dir)
        
        if success:
            completed += 1
            duration = datetime.now() - start_time
            print(f"\\nCompleted in {duration}")
        else:
            failed += 1
            print(f"\\nFAILED: {exp_id}")
    
    # Print summary
    print(f"\\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Total experiments: {total}")
    print(f"Completed: {completed}")
    print(f"Skipped (already done): {skipped}")
    print(f"Failed: {failed}")
    print(f"{'='*60}")
    
    return failed == 0

def analyze_results():
    """Analyze experiment results inline."""
    print("\\n" + "="*60)
    print("Analyzing results...")
    print("="*60)
    
    # Import and run analyze_results inline
    from analyze_results import main as analyze_main
    
    # Temporarily modify sys.argv for argparse
    old_argv = sys.argv
    sys.argv = ['analyze_results.py']
    try:
        analyze_main()
        return True
    except Exception as e:
        print(f"Error analyzing results: {e}")
        return False
    finally:
        sys.argv = old_argv

def main():
    parser = argparse.ArgumentParser(description='Run all APT experiments (fast version)')
    parser.add_argument('--skip-generation', action='store_true',
                        help='Skip config generation (use existing configs)')
    parser.add_argument('--skip-experiments', action='store_true',
                        help='Skip running experiments (only analyze existing results)')
    parser.add_argument('--subset', type=str, default=None,
                        choices=['k', 'int_gamma', 'ent_coef'],
                        help='Run only experiments for a specific parameter')
    parser.add_argument('--quick-test', action='store_true',
                        help='Quick test mode - reduce timesteps to 100K')
    
    args = parser.parse_args()
    
    print("APT Hyperparameter Ablation Experiments (Fast Mode)")
    print(f"Started at: {datetime.now()}")
    
    # Step 1: Generate configurations
    if not args.skip_generation:
        if not generate_configs():
            print("Failed to generate configurations")
            return 1
    
    # Step 2: Run experiments
    if not args.skip_experiments:
        if not run_experiments(subset=args.subset, quick_test=args.quick_test):
            print("Some experiments failed")
            # Continue to analysis anyway
    
    # Step 3: Analyze results
    if not analyze_results():
        print("Failed to analyze results")
        return 1
    
    print(f"\\nCompleted at: {datetime.now()}")
    return 0

if __name__ == "__main__":
    sys.exit(main())