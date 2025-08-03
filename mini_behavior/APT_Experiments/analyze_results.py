#!/usr/bin/env python3
"""
Analyze APT experiment results and calculate JS divergences.
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from post_processing import batch_js_divergence

def analyze_experiment(exp_id, config, results_dir):
    """Analyze a single experiment's results."""
    exp_path = results_dir / exp_id
    
    if not exp_path.exists():
        return None
    
    # Check if experiment completed
    if not (exp_path / "SUCCESS").exists():
        return None
    
    # Get checkpoint activity logs
    activity_logs_dir = exp_path / "checkpoints" / "activity_logs"
    if not activity_logs_dir.exists():
        print(f"Warning: No activity logs found for {exp_id}")
        return None
    
    # Process activity logs to get JS divergences
    try:
        divergences = batch_js_divergence.process_activity_logs(
            str(activity_logs_dir),
            "../behavioral_diversity_benchmarks.pkl"
        )
    except Exception as e:
        print(f"Error processing {exp_id}: {e}")
        return None
    
    # Extract results
    result = {
        'experiment_id': exp_id,
        'ablated_param': config['ablated_param'],
        'ablated_value': config['ablated_value'],
        'k': config['k'],
        'int_gamma': config['int_gamma'],
        'ent_coef': config['ent_coef'],
    }
    
    # Add JS divergences for each checkpoint
    for checkpoint, js_div in divergences.items():
        result[f'js_div_{checkpoint}'] = js_div
    
    # Calculate final and average JS divergence
    if divergences:
        js_values = list(divergences.values())
        result['final_js_div'] = js_values[-1]  # Last checkpoint
        result['avg_js_div'] = np.mean(js_values)
        result['min_js_div'] = np.min(js_values)
    
    return result

def generate_summary_report(results_df, output_file="experiment_summary.md"):
    """Generate a markdown summary report of all experiments."""
    
    with open(output_file, 'w') as f:
        f.write("# APT Hyperparameter Ablation Results\\n\\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
        
        # Overall summary
        f.write("## Overall Summary\\n\\n")
        f.write(f"- Total experiments completed: {len(results_df)}\\n")
        f.write(f"- Parameters tested: k, int_gamma, ent_coef\\n\\n")
        
        # Best overall configuration
        if 'final_js_div' in results_df.columns:
            best_idx = results_df['final_js_div'].idxmin()
            best_exp = results_df.loc[best_idx]
            
            f.write("## Best Configuration (Lowest Final JS Divergence)\\n\\n")
            f.write(f"- Experiment: {best_exp['experiment_id']}\\n")
            f.write(f"- k: {best_exp['k']}\\n")
            f.write(f"- int_gamma: {best_exp['int_gamma']}\\n")
            f.write(f"- ent_coef: {best_exp['ent_coef']}\\n")
            f.write(f"- Final JS Divergence: {best_exp['final_js_div']:.4f}\\n\\n")
        
        # Results by parameter
        params = ['k', 'int_gamma', 'ent_coef']
        
        for param in params:
            f.write(f"## Results for {param}\\n\\n")
            
            # Group by this parameter
            param_groups = results_df.groupby('ablated_param').get_group(param) if param in results_df['ablated_param'].values else pd.DataFrame()
            
            if not param_groups.empty:
                # Sort by ablated value
                param_groups = param_groups.sort_values('ablated_value')
                
                # Create table
                f.write("| Value | Final JS Div | Avg JS Div | Min JS Div | Experiment ID |\\n")
                f.write("|-------|--------------|------------|------------|---------------|\\n")
                
                for _, row in param_groups.iterrows():
                    f.write(f"| {row['ablated_value']} | ")
                    if 'final_js_div' in row:
                        f.write(f"{row['final_js_div']:.4f} | ")
                        f.write(f"{row['avg_js_div']:.4f} | ")
                        f.write(f"{row['min_js_div']:.4f} | ")
                    else:
                        f.write("N/A | N/A | N/A | ")
                    f.write(f"{row['experiment_id']} |\\n")
                
                # Find best value for this parameter
                if 'final_js_div' in param_groups.columns:
                    best_idx = param_groups['final_js_div'].idxmin()
                    best_value = param_groups.loc[best_idx, 'ablated_value']
                    best_js = param_groups.loc[best_idx, 'final_js_div']
                    f.write(f"\\n**Best {param}: {best_value} (JS Div: {best_js:.4f})**\\n\\n")
            else:
                f.write("No results found for this parameter.\\n\\n")
        
        # Detailed results table
        f.write("## Detailed Results\\n\\n")
        f.write("| Experiment | k | int_gamma | ent_coef | Final JS | Avg JS | Min JS |\\n")
        f.write("|------------|---|-----------|----------|----------|--------|--------|\\n")
        
        # Sort by final JS divergence
        if 'final_js_div' in results_df.columns:
            sorted_df = results_df.sort_values('final_js_div')
        else:
            sorted_df = results_df
            
        for _, row in sorted_df.iterrows():
            f.write(f"| {row['experiment_id']} | ")
            f.write(f"{row['k']} | ")
            f.write(f"{row['int_gamma']} | ")
            f.write(f"{row['ent_coef']} | ")
            
            if 'final_js_div' in row:
                f.write(f"{row['final_js_div']:.4f} | ")
                f.write(f"{row['avg_js_div']:.4f} | ")
                f.write(f"{row['min_js_div']:.4f} |\\n")
            else:
                f.write("N/A | N/A | N/A |\\n")
    
    print(f"\\nSummary report saved to: {output_file}")

def main():
    """Analyze all experiment results."""
    
    # Load experiment configurations
    config_file = Path("configs/all_experiments.json")
    if not config_file.exists():
        print("Error: No experiment configurations found. Run generate_configs.py first.")
        return 1
    
    with open(config_file, 'r') as f:
        all_configs = json.load(f)
    
    # Analyze each experiment
    results = []
    results_dir = Path("results")
    
    print("Analyzing experiments...")
    for exp_id, config in all_configs.items():
        print(f"  Analyzing {exp_id}...", end='', flush=True)
        result = analyze_experiment(exp_id, config, results_dir)
        if result:
            results.append(result)
            print(" ✓")
        else:
            print(" ✗ (incomplete or missing)")
    
    if not results:
        print("\\nNo completed experiments found to analyze.")
        return 1
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Save detailed results
    results_df.to_csv("experiment_results.csv", index=False)
    print(f"\\nDetailed results saved to: experiment_results.csv")
    
    # Generate summary report
    generate_summary_report(results_df)
    
    # Print quick summary
    print("\\n" + "="*60)
    print("QUICK SUMMARY")
    print("="*60)
    print(f"Experiments analyzed: {len(results_df)}")
    
    if 'final_js_div' in results_df.columns:
        best_idx = results_df['final_js_div'].idxmin()
        best_exp = results_df.loc[best_idx]
        
        print(f"\\nBest configuration:")
        print(f"  Experiment: {best_exp['experiment_id']}")
        print(f"  k={best_exp['k']}, int_gamma={best_exp['int_gamma']}, ent_coef={best_exp['ent_coef']}")
        print(f"  Final JS Divergence: {best_exp['final_js_div']:.4f}")
        
        # Best value for each parameter
        print(f"\\nBest values by parameter:")
        for param in ['k', 'int_gamma', 'ent_coef']:
            param_df = results_df[results_df['ablated_param'] == param]
            if not param_df.empty and 'final_js_div' in param_df.columns:
                best_param_idx = param_df['final_js_div'].idxmin()
                best_value = param_df.loc[best_param_idx, 'ablated_value']
                best_js = param_df.loc[best_param_idx, 'final_js_div']
                print(f"  {param}: {best_value} (JS Div: {best_js:.4f})")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())