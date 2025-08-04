#!/usr/bin/env python3
"""
Run all NovelD experiments with extrinsic rewards sequentially
"""
import os
import sys
import json
import time
import subprocess
from datetime import datetime

def run_all_experiments(configs_dir='configs', output_dir='results'):
    """Run all experiments from config directory"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all experiment config files
    config_files = sorted([f for f in os.listdir(configs_dir) 
                          if f.startswith('exp_') and f.endswith('.json')])
    
    if not config_files:
        print("No experiment config files found!")
        return
    
    print(f"Found {len(config_files)} experiments to run")
    
    # Track results
    results = {
        'start_time': datetime.now().isoformat(),
        'experiments': []
    }
    
    for i, config_file in enumerate(config_files):
        config_path = os.path.join(configs_dir, config_file)
        
        # Load config to get experiment name
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        exp_name = config['name']
        print(f"\n{'='*60}")
        print(f"Running experiment {i+1}/{len(config_files)}: {exp_name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Run experiment
        cmd = [sys.executable, 'run_single_experiment.py', config_path, 
               '--output-dir', output_dir]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                status = 'SUCCESS'
                print(f"✓ {exp_name} completed successfully")
            else:
                status = 'FAILED'
                print(f"✗ {exp_name} failed")
                print(f"Error: {result.stderr}")
            
        except Exception as e:
            status = 'ERROR'
            print(f"✗ {exp_name} encountered an error: {e}")
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Record result
        results['experiments'].append({
            'name': exp_name,
            'config_file': config_file,
            'status': status,
            'duration_seconds': duration,
            'description': config.get('description', ''),
            'extrinsic_rewards': config.get('extrinsic_rewards', {})
        })
        
        # Save intermediate results
        results_path = os.path.join(output_dir, 'experiment_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Time taken: {duration/60:.2f} minutes")
    
    # Final summary
    results['end_time'] = datetime.now().isoformat()
    results['total_duration_hours'] = sum(exp['duration_seconds'] 
                                         for exp in results['experiments']) / 3600
    
    # Save final results
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    
    successful = sum(1 for exp in results['experiments'] if exp['status'] == 'SUCCESS')
    failed = sum(1 for exp in results['experiments'] if exp['status'] != 'SUCCESS')
    
    print(f"Total experiments: {len(results['experiments'])}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {results['total_duration_hours']:.2f} hours")
    
    if failed > 0:
        print("\nFailed experiments:")
        for exp in results['experiments']:
            if exp['status'] != 'SUCCESS':
                print(f"  - {exp['name']} ({exp['status']})")

if __name__ == "__main__":
    run_all_experiments()