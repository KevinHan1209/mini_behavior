import os
import sys
from batch_js_divergence import process_activity_logs

def find_experiment_with_min_js_divergence(experiments_dir="APT_Experiments/results", pkl_path="post_processing/averaged_state_distributions.pkl"):
    """
    Find the experiment with minimum average JS divergence across all checkpoints.
    
    Args:
        experiments_dir (str): Path to the experiments results directory
        pkl_path (str): Path to the agent distribution pickle file
    
    Returns:
        dict: Information about the experiment with minimum JS divergence
    """
    experiment_results = []
    
    # Get all experiment directories
    exp_dirs = sorted([d for d in os.listdir(experiments_dir) if d.startswith("exp_") and os.path.isdir(os.path.join(experiments_dir, d))])
    
    if not exp_dirs:
        print(f"No experiment directories found in {experiments_dir}")
        return None
    
    print(f"Found {len(exp_dirs)} experiments to analyze\n")
    print("="*80)
    
    # Process each experiment
    for exp_dir in exp_dirs:
        print(f"\nProcessing {exp_dir}...")
        print("-"*40)
        
        activity_logs_path = os.path.join(experiments_dir, exp_dir, "checkpoints", "activity_logs")
        
        if not os.path.exists(activity_logs_path):
            print(f"  ✗ No activity_logs directory found at {activity_logs_path}")
            continue
        
        try:
            # Process the experiment's activity logs
            results = process_activity_logs(activity_logs_path, pkl_path, verbose=False)
            
            # Calculate average JS divergence for this experiment
            valid_results = [r for r in results if 'error' not in r and r['js_divergence'] is not None]
            
            if valid_results:
                avg_js = sum(r['js_divergence'] for r in valid_results) / len(valid_results)
                
                experiment_results.append({
                    'experiment': exp_dir,
                    'average_js_divergence': avg_js,
                    'num_checkpoints': len(valid_results),
                    'checkpoint_results': valid_results
                })
                
                print(f"\n  ✓ Average JS Divergence for {exp_dir}: {avg_js:.4f}")
            else:
                print(f"  ✗ No valid results for {exp_dir}")
                
        except Exception as e:
            print(f"  ✗ Error processing {exp_dir}: {str(e)}")
    
    # Find experiment with minimum JS divergence
    if experiment_results:
        print("\n" + "="*80)
        print("EXPERIMENT COMPARISON SUMMARY")
        print("="*80)
        print(f"{'Experiment':<30} {'Avg JS Divergence':<20} {'Checkpoints':<15}")
        print("-"*80)
        
        # Sort by average JS divergence
        experiment_results.sort(key=lambda x: x['average_js_divergence'])
        
        for result in experiment_results:
            print(f"{result['experiment']:<30} {result['average_js_divergence']:<20.4f} {result['num_checkpoints']:<15}")
        
        # Get the experiment with minimum JS divergence
        min_exp = experiment_results[0]
        
        print("\n" + "="*80)
        print("EXPERIMENT WITH MINIMUM JS DIVERGENCE")
        print("="*80)
        print(f"Experiment: {min_exp['experiment']}")
        print(f"Average JS Divergence: {min_exp['average_js_divergence']:.4f}")
        print(f"Number of checkpoints: {min_exp['num_checkpoints']}")
        
        # Show checkpoint details for the best experiment
        print("\nCheckpoint breakdown for best experiment:")
        print(f"{'Checkpoint':<20} {'JS Divergence':<15}")
        print("-"*35)
        for ckpt in min_exp['checkpoint_results']:
            print(f"{ckpt['checkpoint']:,} {ckpt['js_divergence']:.4f}")
        
        return min_exp
    else:
        print("\nNo valid experiment results found.")
        return None

if __name__ == "__main__":
    # Run the analysis
    result = find_experiment_with_min_js_divergence()