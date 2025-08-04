#!/usr/bin/env python3
"""
Analyze results from NovelD extrinsic reward experiments
"""
import os
import json
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

def load_experiment_results(results_dir='results'):
    """Load all experiment results and activity logs"""
    experiments = []
    
    for exp_dir in os.listdir(results_dir):
        exp_path = os.path.join(results_dir, exp_dir)
        if not os.path.isdir(exp_path) or exp_dir == 'experiment_results.json':
            continue
            
        # Load experiment config
        config_path = os.path.join(exp_path, 'experiment_config.json')
        if not os.path.exists(config_path):
            continue
            
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Load activity logs for final checkpoint
        activity_logs = {}
        activity_dir = os.path.join(exp_path, 'checkpoints', 'activity_logs')
        if os.path.exists(activity_dir):
            for log_file in sorted(os.listdir(activity_dir)):
                if log_file.endswith('_activity.csv'):
                    checkpoint_step = int(log_file.split('_')[1])
                    log_path = os.path.join(activity_dir, log_file)
                    activity_logs[checkpoint_step] = pd.read_csv(log_path)
        
        experiments.append({
            'name': config['name'],
            'description': config['description'],
            'extrinsic_rewards': config['extrinsic_rewards'],
            'hyperparameters': config['hyperparameters'],
            'activity_logs': activity_logs,
            'success': os.path.exists(os.path.join(exp_path, 'SUCCESS'))
        })
    
    return experiments

def analyze_reward_categories(experiments):
    """Analyze performance by reward category"""
    category_results = defaultdict(list)
    
    for exp in experiments:
        if not exp['success'] or not exp['activity_logs']:
            continue
        
        # Get final checkpoint activity
        final_checkpoint = max(exp['activity_logs'].keys())
        final_activity = exp['activity_logs'][final_checkpoint]
        
        # Calculate metrics
        total_interactions = final_activity['count'].sum()
        unique_interactions = len(final_activity)
        mean_count = final_activity['count'].mean()
        
        # Categorize experiment
        rewards = exp['extrinsic_rewards']
        if not rewards:
            category = 'baseline'
        elif len(rewards) == 1:
            category = list(rewards.keys())[0] + '_only'
        elif len(rewards) == 2:
            category = 'two_rewards'
        else:
            category = 'all_rewards'
        
        category_results[category].append({
            'name': exp['name'],
            'total_interactions': total_interactions,
            'unique_interactions': unique_interactions,
            'mean_count': mean_count,
            'rewards': rewards
        })
    
    return category_results

def plot_category_comparison(category_results, output_dir='analysis'):
    """Create comparison plots for different reward categories"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for plotting
    categories = []
    total_interactions = []
    unique_interactions = []
    
    for category, results in category_results.items():
        for result in results:
            categories.append(category)
            total_interactions.append(result['total_interactions'])
            unique_interactions.append(result['unique_interactions'])
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Total interactions by category
    df1 = pd.DataFrame({
        'Category': categories,
        'Total Interactions': total_interactions
    })
    sns.boxplot(data=df1, x='Category', y='Total Interactions', ax=ax1)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    ax1.set_title('Total Interactions by Reward Category')
    
    # Unique interactions by category
    df2 = pd.DataFrame({
        'Category': categories,
        'Unique Interactions': unique_interactions
    })
    sns.boxplot(data=df2, x='Category', y='Unique Interactions', ax=ax2)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    ax2.set_title('Unique Interactions by Reward Category')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'category_comparison.png'), dpi=150)
    plt.close()

def analyze_reward_combinations(experiments, output_dir='analysis'):
    """Analyze the effect of different reward combinations"""
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for exp in experiments:
        if not exp['success'] or not exp['activity_logs']:
            continue
        
        # Get final checkpoint activity
        final_checkpoint = max(exp['activity_logs'].keys())
        final_activity = exp['activity_logs'][final_checkpoint]
        
        # Create reward flags
        rewards = exp['extrinsic_rewards']
        has_noise = 'noise' in rewards
        has_interaction = 'interaction' in rewards
        has_location = 'location_change' in rewards
        
        results.append({
            'name': exp['name'],
            'noise': has_noise,
            'interaction': has_interaction,
            'location': has_location,
            'total_interactions': final_activity['count'].sum(),
            'unique_interactions': len(final_activity),
            'description': exp['description']
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Create heatmap of performance by reward combination
    pivot_total = df.pivot_table(
        values='total_interactions',
        index=['noise', 'interaction'],
        columns='location',
        aggfunc='mean'
    )
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_total, annot=True, fmt='.0f', cmap='YlOrRd')
    plt.title('Average Total Interactions by Reward Combination')
    plt.savefig(os.path.join(output_dir, 'reward_combination_heatmap.png'), dpi=150)
    plt.close()
    
    return df

def generate_report(experiments, output_dir='analysis'):
    """Generate a comprehensive analysis report"""
    os.makedirs(output_dir, exist_ok=True)
    
    report_lines = [
        "# NovelD Extrinsic Reward Experiment Analysis",
        f"\nTotal experiments: {len(experiments)}",
        f"Successful experiments: {sum(1 for exp in experiments if exp['success'])}",
        "\n## Experiment Summary\n"
    ]
    
    # Sort experiments by name
    experiments.sort(key=lambda x: x['name'])
    
    for exp in experiments:
        report_lines.append(f"### {exp['name']}")
        report_lines.append(f"- Description: {exp['description']}")
        report_lines.append(f"- Success: {'Yes' if exp['success'] else 'No'}")
        
        if exp['extrinsic_rewards']:
            report_lines.append("- Extrinsic Rewards:")
            for reward_type, value in exp['extrinsic_rewards'].items():
                report_lines.append(f"  - {reward_type}: {value}")
        else:
            report_lines.append("- Extrinsic Rewards: None (baseline)")
        
        if exp['success'] and exp['activity_logs']:
            final_checkpoint = max(exp['activity_logs'].keys())
            final_activity = exp['activity_logs'][final_checkpoint]
            report_lines.append(f"- Final Performance (checkpoint {final_checkpoint}):")
            report_lines.append(f"  - Total interactions: {final_activity['count'].sum()}")
            report_lines.append(f"  - Unique interactions: {len(final_activity)}")
            report_lines.append(f"  - Mean count per interaction: {final_activity['count'].mean():.2f}")
        
        report_lines.append("")
    
    # Analyze by category
    category_results = analyze_reward_categories(experiments)
    
    report_lines.append("\n## Performance by Reward Category\n")
    for category, results in sorted(category_results.items()):
        report_lines.append(f"### {category}")
        if results:
            avg_total = np.mean([r['total_interactions'] for r in results])
            avg_unique = np.mean([r['unique_interactions'] for r in results])
            report_lines.append(f"- Average total interactions: {avg_total:.0f}")
            report_lines.append(f"- Average unique interactions: {avg_unique:.1f}")
            report_lines.append(f"- Number of experiments: {len(results)}")
        report_lines.append("")
    
    # Save report
    report_path = os.path.join(output_dir, 'analysis_report.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Report saved to: {report_path}")

def main():
    """Run full analysis"""
    print("Loading experiment results...")
    experiments = load_experiment_results()
    
    if not experiments:
        print("No experiment results found!")
        return
    
    print(f"Loaded {len(experiments)} experiments")
    
    # Generate visualizations
    print("Analyzing reward categories...")
    category_results = analyze_reward_categories(experiments)
    plot_category_comparison(category_results)
    
    print("Analyzing reward combinations...")
    analyze_reward_combinations(experiments)
    
    print("Generating report...")
    generate_report(experiments)
    
    print("\nAnalysis complete! Check the 'analysis' directory for results.")

if __name__ == "__main__":
    main()