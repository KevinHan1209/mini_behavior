import os
import sys
from batch_js_divergence import process_activity_logs
import json

def generate_js_divergence_summary(experiments_dir="APT_Experiments/results", pkl_path="post_processing/averaged_state_distributions.pkl"):
    """
    Generate a comprehensive summary of JS divergence statistics for all experiments.
    
    Args:
        experiments_dir (str): Path to the experiments results directory
        pkl_path (str): Path to the agent distribution pickle file
    """
    experiment_results = []
    
    # Get all experiment directories
    exp_dirs = sorted([d for d in os.listdir(experiments_dir) if d.startswith("exp_") and os.path.isdir(os.path.join(experiments_dir, d))])
    
    if not exp_dirs:
        print(f"No experiment directories found in {experiments_dir}")
        return None
    
    print(f"Processing {len(exp_dirs)} experiments...\n")
    
    # Process each experiment
    for exp_dir in exp_dirs:
        activity_logs_path = os.path.join(experiments_dir, exp_dir, "checkpoints", "activity_logs")
        
        if not os.path.exists(activity_logs_path):
            continue
        
        try:
            # Process the experiment's activity logs
            results = process_activity_logs(activity_logs_path, pkl_path, verbose=False)
            
            # Calculate statistics for this experiment
            valid_results = [r for r in results if 'error' not in r and r['js_divergence'] is not None]
            
            if valid_results:
                js_values = [r['js_divergence'] for r in valid_results]
                
                # Read experiment config to get parameter info
                config_path = os.path.join(experiments_dir, exp_dir, "experiment_config.json")
                param_info = ""
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                        # Extract the ablated parameter
                        if 'ablated_param' in config and 'ablated_value' in config:
                            param_info = f"{config['ablated_param']}={config['ablated_value']}"
                
                experiment_results.append({
                    'experiment': exp_dir,
                    'parameter': param_info,
                    'avg_js': sum(js_values) / len(js_values),
                    'min_js': min(js_values),
                    'max_js': max(js_values),
                    'std_js': (sum((x - sum(js_values)/len(js_values))**2 for x in js_values) / len(js_values))**0.5,
                    'range': max(js_values) - min(js_values),
                    'num_checkpoints': len(valid_results),
                    'checkpoints': valid_results
                })
                
        except Exception as e:
            print(f"Error processing {exp_dir}: {str(e)}")
    
    # Sort by average JS divergence
    experiment_results.sort(key=lambda x: x['avg_js'])
    
    # Print comprehensive summary
    print("\n" + "="*100)
    print("JS DIVERGENCE SUMMARY FOR ALL EXPERIMENTS")
    print("="*100)
    print(f"{'Experiment':<25} {'Parameter':<25} {'Avg JS':<10} {'Min JS':<10} {'Max JS':<10} {'Std JS':<10} {'Range':<10}")
    print("-"*100)
    
    for result in experiment_results:
        print(f"{result['experiment']:<25} {result['parameter']:<25} {result['avg_js']:<10.4f} {result['min_js']:<10.4f} {result['max_js']:<10.4f} {result['std_js']:<10.4f} {result['range']:<10.4f}")
    
    # Print ranking
    print("\n" + "="*100)
    print("EXPERIMENTS RANKED BY AVERAGE JS DIVERGENCE (Best to Worst)")
    print("="*100)
    
    for i, result in enumerate(experiment_results, 1):
        print(f"{i:2d}. {result['experiment']:<25} (avg JS: {result['avg_js']:.4f}) - {result['parameter']}")
    
    # Group by parameter type
    print("\n" + "="*100)
    print("SUMMARY BY PARAMETER TYPE")
    print("="*100)
    
    # Group experiments by parameter type
    param_groups = {
        'alpha': [],
        'update_proportion': [],
        'int_gamma': [],
        'ent_coef': []
    }
    
    for result in experiment_results:
        for param_type in param_groups:
            if param_type in result['parameter']:
                param_groups[param_type].append(result)
                break
    
    # Print summary for each parameter type
    for param_type, experiments in param_groups.items():
        if experiments:
            print(f"\n{param_type.upper()} experiments:")
            experiments.sort(key=lambda x: x['avg_js'])
            for exp in experiments:
                print(f"  {exp['parameter']:<30} → avg JS: {exp['avg_js']:.4f} (range: {exp['range']:.4f})")
    
    # Statistical insights
    print("\n" + "="*100)
    print("STATISTICAL INSIGHTS")
    print("="*100)
    
    all_avg_js = [r['avg_js'] for r in experiment_results]
    overall_avg = sum(all_avg_js) / len(all_avg_js)
    overall_min = min(all_avg_js)
    overall_max = max(all_avg_js)
    
    print(f"Overall average JS divergence: {overall_avg:.4f}")
    print(f"Best performing experiment: {experiment_results[0]['experiment']} (JS: {overall_min:.4f})")
    print(f"Worst performing experiment: {experiment_results[-1]['experiment']} (JS: {overall_max:.4f})")
    print(f"Performance spread: {overall_max - overall_min:.4f}")
    
    # Find most stable experiments (lowest std)
    experiment_results.sort(key=lambda x: x['std_js'])
    print(f"\nMost stable experiment (lowest std): {experiment_results[0]['experiment']} (std: {experiment_results[0]['std_js']:.4f})")
    
    # Find most variable experiments (highest range)
    experiment_results.sort(key=lambda x: x['range'], reverse=True)
    print(f"Most variable experiment (highest range): {experiment_results[0]['experiment']} (range: {experiment_results[0]['range']:.4f})")
    
    return experiment_results

if __name__ == "__main__":
    results = generate_js_divergence_summary()