import os
import pandas as pd
import numpy as np
from collections import Counter
from scipy.stats import entropy
import json

def calculate_exploration_metrics(activity_logs_path):
    """
    Calculate exploration metrics from activity logs based on object state diversity.
    
    Returns:
        dict: Dictionary containing exploration metrics
    """
    metrics = {
        'object_states': [],  # List of object state vectors
        'total_checkpoints': 0,
        'state_changes': Counter(),  # Count state changes per object
        'active_objects': set(),  # Objects that changed state
        'state_diversity': {}  # Diversity of states per object
    }
    
    # Process all CSV files
    csv_files = sorted([f for f in os.listdir(activity_logs_path) if f.endswith('.csv')])
    
    all_data = []
    for csv_file in csv_files:
        csv_path = os.path.join(activity_logs_path, csv_file)
        try:
            df = pd.read_csv(csv_path)
            all_data.append(df)
            metrics['total_checkpoints'] += len(df)
        except Exception as e:
            print(f"    Error reading {csv_file}: {str(e)}")
    
    if not all_data:
        return metrics
    
    # Combine all checkpoint data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Analyze each object column
    object_columns = [col for col in combined_df.columns if col != 'checkpoint_id']
    
    for col in object_columns:
        if col in combined_df.columns:
            # Track unique states for this object
            unique_states = combined_df[col].nunique()
            metrics['state_diversity'][col] = unique_states
            
            # Track if object was active (changed states)
            if unique_states > 1:
                metrics['active_objects'].add(col)
            
            # Count state transitions
            if len(combined_df) > 1:
                state_changes = (combined_df[col].diff() != 0).sum()
                metrics['state_changes'][col] = state_changes
    
    # Store full state vectors for analysis
    metrics['object_states'] = combined_df[object_columns].values.tolist()
    
    return metrics

def compute_exploration_scores(metrics):
    """
    Compute exploration scores from raw metrics.
    
    Returns:
        dict: Dictionary containing computed exploration scores
    """
    scores = {}
    
    # 1. Number of active objects (objects that changed state)
    scores['active_objects_count'] = len(metrics['active_objects'])
    
    # 2. Average state diversity across objects
    if metrics['state_diversity']:
        scores['avg_state_diversity'] = np.mean(list(metrics['state_diversity'].values()))
        scores['max_state_diversity'] = max(metrics['state_diversity'].values())
    else:
        scores['avg_state_diversity'] = 0
        scores['max_state_diversity'] = 0
    
    # 3. State change frequency
    if metrics['state_changes']:
        total_changes = sum(metrics['state_changes'].values())
        scores['total_state_changes'] = total_changes
        scores['avg_changes_per_checkpoint'] = total_changes / max(metrics['total_checkpoints'], 1)
    else:
        scores['total_state_changes'] = 0
        scores['avg_changes_per_checkpoint'] = 0
    
    # 4. Object interaction entropy (how evenly distributed are state changes)
    if metrics['state_changes']:
        change_counts = list(metrics['state_changes'].values())
        scores['state_change_entropy'] = entropy(change_counts)
    else:
        scores['state_change_entropy'] = 0
    
    # 5. Unique state configurations
    if metrics['object_states']:
        # Convert state vectors to tuples for hashing
        unique_configs = len(set(tuple(state) for state in metrics['object_states']))
        scores['unique_state_configs'] = unique_configs
        scores['config_diversity_ratio'] = unique_configs / len(metrics['object_states'])
    else:
        scores['unique_state_configs'] = 0
        scores['config_diversity_ratio'] = 0
    
    # 6. Combined exploration score
    scores['combined_score'] = (
        scores['active_objects_count'] * 1.0 +
        scores['avg_state_diversity'] * 0.5 +
        scores['state_change_entropy'] * 0.3 +
        scores['config_diversity_ratio'] * 100 +  # Scale up
        scores['avg_changes_per_checkpoint'] * 0.1
    )
    
    return scores

def analyze_apt_exploration(experiments_dir="APT_Experiments/results"):
    """
    Analyze exploration levels across all APT experiments.
    """
    experiment_results = []
    
    # Get all experiment directories
    exp_dirs = sorted([d for d in os.listdir(experiments_dir) if d.startswith("exp_") and os.path.isdir(os.path.join(experiments_dir, d))])
    
    if not exp_dirs:
        print(f"No experiment directories found in {experiments_dir}")
        return None
    
    print(f"Analyzing exploration metrics for {len(exp_dirs)} experiments...\n")
    print("="*100)
    
    for exp_dir in exp_dirs:
        print(f"\nProcessing {exp_dir}...")
        activity_logs_path = os.path.join(experiments_dir, exp_dir, "checkpoints", "activity_logs")
        
        if not os.path.exists(activity_logs_path):
            print(f"  ✗ No activity_logs directory found")
            continue
        
        try:
            # Calculate exploration metrics
            metrics = calculate_exploration_metrics(activity_logs_path)
            scores = compute_exploration_scores(metrics)
            
            # Read experiment config
            config_path = os.path.join(experiments_dir, exp_dir, "experiment_config.json")
            param_info = ""
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    if 'ablated_param' in config and 'ablated_value' in config:
                        param_info = f"{config['ablated_param']}={config['ablated_value']}"
            
            experiment_results.append({
                'experiment': exp_dir,
                'parameter': param_info,
                **scores,
                'total_checkpoints': metrics['total_checkpoints']
            })
            
            print(f"  ✓ Analyzed: {scores['active_objects_count']} active objects, "
                  f"{scores['unique_state_configs']} unique configurations")
                  
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    if not experiment_results:
        print("\nNo valid experiment results found.")
        return None
    
    # Sort by combined score
    experiment_results.sort(key=lambda x: x['combined_score'], reverse=True)
    
    # Print summary table
    print("\n" + "="*120)
    print("EXPLORATION METRICS SUMMARY")
    print("="*120)
    print(f"{'Experiment':<25} {'Parameter':<20} {'Active':<8} {'Unique':<8} {'Changes':<10} {'Diversity':<10} {'Combined':<10}")
    print(f"{'':25} {'':20} {'Objects':<8} {'States':<8} {'Per Step':<10} {'Ratio':<10} {'Score':<10}")
    print("-"*120)
    
    for result in experiment_results[:10]:  # Top 10
        print(f"{result['experiment']:<25} {result['parameter']:<20} "
              f"{result['active_objects_count']:<8} {result['unique_state_configs']:<8} "
              f"{result['avg_changes_per_checkpoint']:<10.2f} {result['config_diversity_ratio']:<10.4f} "
              f"{result['combined_score']:<10.2f}")
    
    # Category winners
    print("\n" + "="*100)
    print("CATEGORY WINNERS")
    print("="*100)
    
    categories = [
        ('active_objects_count', 'Most Active Objects'),
        ('unique_state_configs', 'Most Unique State Configurations'),
        ('total_state_changes', 'Most Total State Changes'),
        ('state_change_entropy', 'Most Balanced Object Interactions'),
        ('config_diversity_ratio', 'Highest Configuration Diversity'),
        ('combined_score', 'Best Overall Exploration')
    ]
    
    for metric, description in categories:
        best = max(experiment_results, key=lambda x: x[metric])
        print(f"\n{description}:")
        print(f"  Winner: {best['experiment']} ({best['parameter']})")
        print(f"  Score: {best[metric]:.4f}")
    
    # Detailed breakdown of top explorer
    print("\n" + "="*100)
    print("HIGHEST EXPLORATION EXPERIMENT (DETAILED)")
    print("="*100)
    
    top_explorer = experiment_results[0]
    print(f"Experiment: {top_explorer['experiment']}")
    print(f"Parameter: {top_explorer['parameter']}")
    print(f"\nDetailed Metrics:")
    print(f"  - Active Objects: {top_explorer['active_objects_count']}")
    print(f"  - Unique State Configurations: {top_explorer['unique_state_configs']}")
    print(f"  - Total State Changes: {top_explorer['total_state_changes']}")
    print(f"  - Average Changes per Checkpoint: {top_explorer['avg_changes_per_checkpoint']:.2f}")
    print(f"  - Configuration Diversity Ratio: {top_explorer['config_diversity_ratio']:.4f}")
    print(f"  - State Change Entropy: {top_explorer['state_change_entropy']:.4f}")
    print(f"  - Average State Diversity: {top_explorer['avg_state_diversity']:.2f}")
    print(f"  - Combined Exploration Score: {top_explorer['combined_score']:.2f}")
    
    return experiment_results

if __name__ == "__main__":
    results = analyze_apt_exploration()