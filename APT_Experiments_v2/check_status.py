#!/usr/bin/env python3
"""
Check the status of all experiments in APT_Experiments_v2.
"""

import os
import json
from pathlib import Path
from datetime import datetime

def check_experiment_status():
    """Check and display the status of all experiments."""
    
    configs_dir = Path("configs")
    results_dir = Path("results")
    
    # Get all config files
    config_files = sorted(configs_dir.glob("*.json"))
    
    print("=" * 80)
    print("APT_EXPERIMENTS_V2 STATUS REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()
    
    # Summary statistics
    total_experiments = len(config_files)
    completed = 0
    failed = 0
    not_started = 0
    
    # Detailed status
    print("EXPERIMENT STATUS:")
    print("-" * 80)
    print(f"{'Experiment':<30} {'Status':<15} {'Checkpoints':<15} {'Visualizations'}")
    print("-" * 80)
    
    for config_file in config_files:
        # Load config to get experiment name
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        exp_name = config['experiment_name']
        exp_dir = results_dir / exp_name
        
        # Check status
        if not exp_dir.exists():
            status = "NOT STARTED"
            not_started += 1
            checkpoints = "-"
            has_viz = "-"
        elif (exp_dir / "SUCCESS").exists():
            status = "✓ COMPLETED"
            completed += 1
            
            # Count checkpoints
            checkpoint_dir = exp_dir / "checkpoints"
            if checkpoint_dir.exists():
                checkpoint_files = list(checkpoint_dir.glob("checkpoint_*.pt"))
                checkpoints = str(len(checkpoint_files))
            else:
                checkpoints = "0"
            
            # Check for visualizations
            viz_dir = exp_dir / "visualizations"
            has_viz = "✓" if viz_dir.exists() else "✗"
            
        elif (exp_dir / "ERROR.txt").exists():
            status = "✗ FAILED"
            failed += 1
            checkpoints = "-"
            has_viz = "-"
        else:
            status = "IN PROGRESS"
            
            # Count checkpoints
            checkpoint_dir = exp_dir / "checkpoints"
            if checkpoint_dir.exists():
                checkpoint_files = list(checkpoint_dir.glob("checkpoint_*.pt"))
                checkpoints = str(len(checkpoint_files))
            else:
                checkpoints = "0"
            has_viz = "-"
        
        # Print row
        print(f"{exp_name:<30} {status:<15} {checkpoints:<15} {has_viz}")
    
    # Print summary
    print()
    print("=" * 80)
    print("SUMMARY:")
    print("-" * 80)
    print(f"Total Experiments: {total_experiments}")
    print(f"  ✓ Completed:     {completed}")
    print(f"  ✗ Failed:        {failed}")
    print(f"  ⚬ Not Started:   {not_started}")
    print(f"  ⟳ In Progress:   {total_experiments - completed - failed - not_started}")
    print()
    
    # Check for failed experiments
    if failed > 0:
        print("FAILED EXPERIMENTS:")
        print("-" * 80)
        for config_file in config_files:
            with open(config_file, 'r') as f:
                config = json.load(f)
            exp_name = config['experiment_name']
            error_file = results_dir / exp_name / "ERROR.txt"
            
            if error_file.exists():
                print(f"\n{exp_name}:")
                with open(error_file, 'r') as f:
                    error_text = f.read()
                    # Print first 3 lines of error
                    lines = error_text.split('\n')[:3]
                    for line in lines:
                        print(f"  {line}")
        print()
    
    print("=" * 80)

if __name__ == "__main__":
    check_experiment_status()