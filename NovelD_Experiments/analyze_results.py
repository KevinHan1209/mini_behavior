#!/usr/bin/env python3
"""
Analyze experiment results and calculate JS divergences
"""
import sys
import os
import json
import glob
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from post_processing.batch_js_divergence import process_activity_logs

def analyze_experiment(exp_dir, pkl_path):
    """Analyze a single experiment's results"""
    # Load experiment config
    config_path = os.path.join(exp_dir, 'experiment_config.json')
    if not os.path.exists(config_path):
        return None
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Check if experiment completed successfully
    success_marker = os.path.join(exp_dir, 'SUCCESS')
    if not os.path.exists(success_marker):
        print(f"  ⚠️  Experiment {config['name']} did not complete successfully")
        return None
    
    # Find activity logs
    activity_logs_dir = os.path.join(exp_dir, 'checkpoints', 'activity_logs')
    if not os.path.exists(activity_logs_dir):
        print(f"  ⚠️  No activity logs found for {config['name']}")
        return None
    
    print(f"\n  Analyzing {config['name']}...")
    
    # Calculate JS divergences
    js_results = process_activity_logs(activity_logs_dir, pkl_path, verbose=False)
    
    # Prepare results
    result = {
        'experiment_name': config['name'],
        'ablated_param': config['ablated_param'],
        'ablated_value': config['ablated_value'],
        'hyperparameters': config['hyperparameters'],
        'js_results': js_results
    }
    
    # Get final JS divergence (from last checkpoint)
    if js_results:
        valid_results = [r for r in js_results if 'error' not in r and r['js_divergence'] is not None]
        if valid_results:
            # Sort by checkpoint and get last one
            sorted_results = sorted(valid_results, key=lambda x: x.get('checkpoint', 0))
            result['final_js_divergence'] = sorted_results[-1]['js_divergence']
            result['final_checkpoint'] = sorted_results[-1].get('checkpoint', 'unknown')
        else:
            result['final_js_divergence'] = None
            result['final_checkpoint'] = None
    
    return result

def generate_summary_report(all_results, output_path):
    """Generate a markdown summary report"""
    with open(output_path, 'w') as f:
        f.write("# NovelD Hyperparameter Ablation Results\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Group results by ablated parameter
        grouped_results = {}
        for result in all_results:
            if result:
                param = result['ablated_param']
                if param not in grouped_results:
                    grouped_results[param] = []
                grouped_results[param].append(result)
        
        # Summary statistics
        f.write("## Summary\n\n")
        total_experiments = sum(len(experiments) for experiments in grouped_results.values())
        successful_experiments = len([r for r in all_results if r and r.get('final_js_divergence') is not None])
        f.write(f"- Total experiments: {total_experiments}\n")
        f.write(f"- Successful experiments: {successful_experiments}\n")
        f.write(f"- Failed experiments: {total_experiments - successful_experiments}\n\n")
        
        # Default hyperparameters
        if all_results and all_results[0]:
            defaults = {
                'alpha': 0.5,
                'update_proportion': 0.25,
                'int_gamma': 0.99,
                'ent_coef': 0.01
            }
            f.write("## Default Hyperparameters\n\n")
            for param, value in defaults.items():
                f.write(f"- {param}: {value}\n")
            f.write("\n")
        
        # Results by parameter
        f.write("## Results by Parameter\n\n")
        
        for param in ['alpha', 'update_proportion', 'int_gamma', 'ent_coef']:
            if param in grouped_results:
                f.write(f"### {param}\n\n")
                
                # Sort by value
                param_results = sorted(grouped_results[param], 
                                     key=lambda x: x['ablated_value'])
                
                # Create table
                f.write("| Value | Final JS Divergence | Checkpoint | Experiment Name |\n")
                f.write("|-------|--------------------:|------------|----------------|\n")
                
                for result in param_results:
                    value = result['ablated_value']
                    js = result.get('final_js_divergence')
                    checkpoint = result.get('final_checkpoint')
                    name = result['experiment_name']
                    
                    if js is not None:
                        f.write(f"| {value} | {js:.4f} | {checkpoint:,} | {name} |\n")
                    else:
                        f.write(f"| {value} | FAILED | - | {name} |\n")
                
                # Find best value
                valid_param_results = [r for r in param_results if r.get('final_js_divergence') is not None]
                if valid_param_results:
                    best_result = min(valid_param_results, key=lambda x: x['final_js_divergence'])
                    f.write(f"\n**Best value**: {best_result['ablated_value']} ")
                    f.write(f"(JS = {best_result['final_js_divergence']:.4f})\n\n")
                
                f.write("\n")
        
        # Overall best configuration
        f.write("## Best Overall Configuration\n\n")
        valid_results = [r for r in all_results if r and r.get('final_js_divergence') is not None]
        if valid_results:
            best_overall = min(valid_results, key=lambda x: x['final_js_divergence'])
            f.write(f"**Experiment**: {best_overall['experiment_name']}\n")
            f.write(f"**Final JS Divergence**: {best_overall['final_js_divergence']:.4f}\n")
            f.write(f"**Ablated Parameter**: {best_overall['ablated_param']} = {best_overall['ablated_value']}\n\n")
            
            # Show full hyperparameters for best config
            f.write("**Full Hyperparameters**:\n```json\n")
            f.write(json.dumps(best_overall['hyperparameters'], indent=2))
            f.write("\n```\n\n")
        
        # Detailed results for each experiment
        f.write("## Detailed Results\n\n")
        
        for param in ['alpha', 'update_proportion', 'int_gamma', 'ent_coef']:
            if param in grouped_results:
                f.write(f"### {param} Experiments\n\n")
                
                for result in sorted(grouped_results[param], key=lambda x: x['ablated_value']):
                    f.write(f"#### {result['experiment_name']}\n")
                    f.write(f"- **Value**: {result['ablated_value']}\n")
                    
                    if result.get('js_results'):
                        f.write("- **JS Divergence by Checkpoint**:\n")
                        valid_js = [r for r in result['js_results'] if 'error' not in r]
                        for js_result in sorted(valid_js, key=lambda x: x.get('checkpoint', 0)):
                            checkpoint = js_result.get('checkpoint', 'unknown')
                            js_div = js_result['js_divergence']
                            f.write(f"  - Checkpoint {checkpoint:,}: {js_div:.4f}\n")
                    else:
                        f.write("- **Status**: FAILED\n")
                    
                    f.write("\n")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze NovelD experiment results')
    parser.add_argument('--results-dir', default='results',
                       help='Directory containing experiment results')
    parser.add_argument('--pkl-path', default='../post_processing/averaged_state_distributions.pkl',
                       help='Path to agent distribution pickle file')
    parser.add_argument('--output', default='experiment_results.md',
                       help='Output markdown file path')
    
    args = parser.parse_args()
    
    # Check if results directory exists
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory '{args.results_dir}' not found")
        sys.exit(1)
    
    # Find all experiment directories
    exp_dirs = [d for d in glob.glob(os.path.join(args.results_dir, 'exp_*')) 
                if os.path.isdir(d)]
    
    if not exp_dirs:
        print(f"No experiment directories found in {args.results_dir}")
        sys.exit(1)
    
    print(f"Found {len(exp_dirs)} experiment directories")
    
    # Analyze each experiment
    all_results = []
    for exp_dir in sorted(exp_dirs):
        result = analyze_experiment(exp_dir, args.pkl_path)
        all_results.append(result)
    
    # Generate summary report
    print(f"\nGenerating summary report...")
    generate_summary_report(all_results, args.output)
    print(f"Report saved to: {args.output}")
    
    # Print quick summary
    valid_results = [r for r in all_results if r and r.get('final_js_divergence') is not None]
    if valid_results:
        best = min(valid_results, key=lambda x: x['final_js_divergence'])
        print(f"\nBest configuration: {best['ablated_param']} = {best['ablated_value']}")
        print(f"Best JS divergence: {best['final_js_divergence']:.4f}")

if __name__ == "__main__":
    main()