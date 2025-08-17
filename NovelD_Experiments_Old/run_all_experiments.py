#!/usr/bin/env python3
"""
Master script to run all NovelD hyperparameter ablation experiments
"""
import os
import sys
import subprocess
import json
import time
import argparse
from datetime import datetime

def run_command(cmd, description):
    """Run a command and print its output"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                             universal_newlines=True)
    
    for line in iter(process.stdout.readline, ''):
        print(line.rstrip())
    
    process.wait()
    
    if process.returncode != 0:
        print(f"\nError: Command failed with return code {process.returncode}")
        return False
    
    return True

def run_experiment_batch(config_files, max_parallel=1):
    """Run a batch of experiments"""
    completed = 0
    failed = 0
    
    for i, config_file in enumerate(config_files):
        print(f"\n{'#'*70}")
        print(f"EXPERIMENT {i+1}/{len(config_files)}")
        print(f"{'#'*70}")
        
        # Load config to get experiment name
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        exp_name = config['name']
        print(f"Starting: {exp_name}")
        print(f"Ablated: {config['ablated_param']} = {config['ablated_value']}")
        
        # Run the experiment
        cmd = [sys.executable, 'run_single_experiment.py', config_file]
        success = run_command(cmd, f"Running experiment: {exp_name}")
        
        if success:
            completed += 1
            print(f"\n✓ Completed: {exp_name}")
        else:
            failed += 1
            print(f"\n✗ Failed: {exp_name}")
        
        # Small delay between experiments
        if i < len(config_files) - 1:
            print(f"\nWaiting 5 seconds before next experiment...")
            time.sleep(5)
    
    return completed, failed

def main():
    parser = argparse.ArgumentParser(description='Run all NovelD experiments')
    parser.add_argument('--skip-generation', action='store_true',
                       help='Skip config generation if already exists')
    parser.add_argument('--skip-experiments', action='store_true',
                       help='Skip running experiments (only analyze)')
    parser.add_argument('--subset', type=str,
                       help='Only run experiments for specific parameter (alpha, update_proportion, int_gamma, ent_coef)')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run with reduced timesteps for testing')
    
    args = parser.parse_args()
    
    start_time = datetime.now()
    print(f"NovelD Hyperparameter Ablation Pipeline")
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Generate configurations
    if not args.skip_generation:
        success = run_command(
            [sys.executable, 'generate_configs.py'],
            "Step 1: Generating experiment configurations"
        )
        if not success:
            print("Failed to generate configurations")
            sys.exit(1)
    
    # Load configurations
    configs_path = 'configs/all_experiments.json'
    if not os.path.exists(configs_path):
        print(f"Error: Configurations not found at {configs_path}")
        print("Run without --skip-generation flag")
        sys.exit(1)
    
    with open(configs_path, 'r') as f:
        all_configs = json.load(f)
    
    # Filter configurations if subset specified
    if args.subset:
        all_configs = [c for c in all_configs if c['ablated_param'] == args.subset]
        if not all_configs:
            print(f"Error: No configurations found for parameter '{args.subset}'")
            sys.exit(1)
        print(f"\nFiltered to {len(all_configs)} experiments for parameter '{args.subset}'")
    
    # Modify configs for quick test
    if args.quick_test:
        print("\nQuick test mode: Reducing timesteps to 100,000")
        for config in all_configs:
            config['hyperparameters']['total_timesteps'] = 100000
            # Save modified config
            config_path = f"configs/{config['name']}.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
    
    # Step 2: Run experiments
    if not args.skip_experiments:
        # Filter out completed experiments
        completed_experiments = []
        for c in all_configs:
            result_dir = f"results/{c['name']}"
            success_file = os.path.join(result_dir, "SUCCESS")
            if os.path.exists(success_file):
                completed_experiments.append(c['name'])
                print(f"Skipping completed experiment: {c['name']}")
        
        # Filter configs to only include non-completed experiments
        remaining_configs = [c for c in all_configs if c['name'] not in completed_experiments]
        config_files = [f"configs/{c['name']}.json" for c in remaining_configs]
        
        if not config_files:
            print("\nAll experiments have been completed!")
        else:
            print(f"\n{'='*60}")
            print(f"Step 2: Running {len(config_files)} experiments ({len(completed_experiments)} already completed)")
            print(f"{'='*60}")
            
            completed, failed = run_experiment_batch(config_files)
            
            print(f"\n{'='*60}")
            print(f"Experiments Summary:")
            print(f"  Completed: {completed}")
            print(f"  Failed: {failed}")
            print(f"{'='*60}")
            
            if failed > 0:
                print("\nWarning: Some experiments failed. Check logs for details.")
    
    # Step 3: Analyze results
    print("\nWaiting 10 seconds for files to sync...")
    time.sleep(10)
    
    success = run_command(
        [sys.executable, 'analyze_results.py'],
        "Step 3: Analyzing results and calculating JS divergences"
    )
    
    if not success:
        print("Failed to analyze results")
    
    # Print summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n{'='*70}")
    print(f"Pipeline completed!")
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Ended: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration}")
    print(f"{'='*70}")
    
    print("\nResults saved to:")
    print("  - Individual experiments: results/exp_*/")
    print("  - Summary report: experiment_results.md")
    print("\nTo view results:")
    print("  cat experiment_results.md")

if __name__ == "__main__":
    main()