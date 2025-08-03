#!/usr/bin/env python3
"""
Master script to run all APT hyperparameter ablation experiments.
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

def run_command(cmd):
    """Run a command and capture output."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running command: {result.stderr}")
        return False
    print(result.stdout)
    return True

def generate_configs():
    """Generate experiment configurations."""
    print("\\n" + "="*60)
    print("Generating experiment configurations...")
    print("="*60)
    return run_command([sys.executable, "generate_configs.py"])

def run_experiments(subset=None, quick_test=False):
    """Run all experiments sequentially."""
    print("\\n" + "="*60)
    print("Running experiments...")
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
        
        # Save potentially modified config (for quick test)
        if quick_test:
            config_path = Path("configs") / f"{exp_id}_quick.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        else:
            config_path = Path("configs") / f"{exp_id}.json"
        
        # Run experiment
        start_time = datetime.now()
        success = run_command([
            sys.executable, "run_single_experiment.py", 
            str(config_path)
        ])
        
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
    """Analyze experiment results and calculate JS divergences."""
    print("\\n" + "="*60)
    print("Analyzing results...")
    print("="*60)
    
    return run_command([sys.executable, "analyze_results.py"])

def main():
    parser = argparse.ArgumentParser(description='Run all APT hyperparameter ablation experiments')
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
    
    print("APT Hyperparameter Ablation Experiments")
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