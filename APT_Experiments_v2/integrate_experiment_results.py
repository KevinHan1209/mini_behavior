"""
Integrated visualization of all APT_Experiments_v2 results.
Creates comprehensive plots showing exploration metrics across all experiments.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Set style for better-looking plots
try:
    plt.style.use('seaborn-darkgrid')
except:
    plt.style.use('ggplot')  # Fallback style
sns.set_palette("husl")

def load_experiment_config(exp_dir):
    """Load experiment configuration from JSON file."""
    config_path = exp_dir / "experiment_config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    return None

def load_activity_data(exp_dir):
    """Load activity data from checkpoint CSV files."""
    activity_logs_dir = exp_dir / "checkpoints" / "activity_logs"
    if not activity_logs_dir.exists():
        return None
    
    checkpoints = [500000, 1000000, 1500000, 2000000, 2500000]
    activity_data = []
    
    for checkpoint in checkpoints:
        csv_path = activity_logs_dir / f"checkpoint_{checkpoint}_activity.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if not df.empty:
                # Calculate total activity count (sum of all non-zero activities)
                total_activity = df.iloc[0, 1:].sum()  # Skip checkpoint_id column
                unique_states = (df.iloc[0, 1:] > 0).sum()  # Count non-zero states
                
                activity_data.append({
                    'checkpoint': checkpoint,
                    'total_activity': total_activity,
                    'unique_states': unique_states,
                    'state_coverage': unique_states / (len(df.columns) - 1)  # Percentage of states visited
                })
    
    return pd.DataFrame(activity_data) if activity_data else None

def load_intrinsic_rewards(exp_dir):
    """Load intrinsic rewards data from CSV file."""
    rewards_path = exp_dir / "intrinsic_rewards_log.csv"
    if rewards_path.exists():
        return pd.read_csv(rewards_path)
    return None

def get_experiment_label(exp_name, config):
    """Generate a descriptive label for the experiment."""
    if config:
        # Extract key parameters
        k = config.get('k', 12)  # Default k value
        ent_coef = config.get('ent_coef', 0.01)  # Default entropy coefficient
        
        if 'k_' in exp_name:
            return f"k={k}"
        elif 'ent_coef_' in exp_name:
            return f"ent_coef={ent_coef}"
        elif 'default' in exp_name:
            return "Default (k=12, ent=0.01)"
    
    # Fallback to parsing the experiment name
    if 'k_' in exp_name:
        k_value = exp_name.split('k_')[1].split('_')[0]
        return f"k={k_value}"
    elif 'ent_coef_' in exp_name:
        ent_value = exp_name.split('ent_coef_')[1]
        return f"ent_coef={ent_value}"
    elif 'default' in exp_name:
        return "Default"
    
    return exp_name

def create_integrated_visualization(results_dir):
    """Create integrated visualization combining all experiments."""
    results_path = Path(results_dir)
    
    # Collect data from all experiments
    all_experiments = []
    
    for exp_dir in sorted(results_path.iterdir()):
        if exp_dir.is_dir() and exp_dir.name.startswith('exp_'):
            print(f"Processing {exp_dir.name}...")
            
            config = load_experiment_config(exp_dir)
            label = get_experiment_label(exp_dir.name, config)
            
            # Load activity data
            activity_df = load_activity_data(exp_dir)
            if activity_df is not None:
                activity_df['experiment'] = label
                activity_df['exp_name'] = exp_dir.name
                
                # Load intrinsic rewards
                rewards_df = load_intrinsic_rewards(exp_dir)
                if rewards_df is not None:
                    # Merge rewards data at checkpoint intervals
                    checkpoint_rewards = []
                    for checkpoint in [500000, 1000000, 1500000, 2000000, 2500000]:
                        # Find the closest update to this checkpoint
                        closest_idx = np.abs(rewards_df['global_step'] - checkpoint).argmin()
                        checkpoint_rewards.append({
                            'checkpoint': checkpoint,
                            'avg_intrinsic_reward': rewards_df.iloc[closest_idx]['avg_intrinsic_reward'],
                            'std_intrinsic_reward': rewards_df.iloc[closest_idx]['std_intrinsic_reward']
                        })
                    
                    rewards_checkpoint_df = pd.DataFrame(checkpoint_rewards)
                    activity_df = pd.merge(activity_df, rewards_checkpoint_df, on='checkpoint', how='left')
                
                all_experiments.append(activity_df)
    
    if not all_experiments:
        print("No experiment data found!")
        return
    
    # Combine all experiment data
    combined_df = pd.concat(all_experiments, ignore_index=True)
    
    # Create the integrated visualization
    fig = plt.figure(figsize=(20, 16))
    
    # Define color palette for experiments
    experiments = combined_df['experiment'].unique()
    colors = sns.color_palette("husl", len(experiments))
    color_map = dict(zip(experiments, colors))
    
    # 1. Total Activity Count
    ax1 = plt.subplot(3, 3, 1)
    for exp in experiments:
        exp_data = combined_df[combined_df['experiment'] == exp]
        ax1.plot(exp_data['checkpoint'] / 1e6, exp_data['total_activity'], 
                marker='o', label=exp, linewidth=2, markersize=8, color=color_map[exp])
    ax1.set_xlabel('Training Steps (Millions)', fontsize=12)
    ax1.set_ylabel('Total Activity Count', fontsize=12)
    ax1.set_title('Total Activity Count Evolution', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Unique States Visited
    ax2 = plt.subplot(3, 3, 2)
    for exp in experiments:
        exp_data = combined_df[combined_df['experiment'] == exp]
        ax2.plot(exp_data['checkpoint'] / 1e6, exp_data['unique_states'], 
                marker='s', label=exp, linewidth=2, markersize=8, color=color_map[exp])
    ax2.set_xlabel('Training Steps (Millions)', fontsize=12)
    ax2.set_ylabel('Unique States Visited', fontsize=12)
    ax2.set_title('Unique States Exploration', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. State Coverage (%)
    ax3 = plt.subplot(3, 3, 3)
    for exp in experiments:
        exp_data = combined_df[combined_df['experiment'] == exp]
        ax3.plot(exp_data['checkpoint'] / 1e6, exp_data['state_coverage'] * 100, 
                marker='^', label=exp, linewidth=2, markersize=8, color=color_map[exp])
    ax3.set_xlabel('Training Steps (Millions)', fontsize=12)
    ax3.set_ylabel('State Coverage (%)', fontsize=12)
    ax3.set_title('State Space Coverage', fontsize=14, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. Average Intrinsic Reward
    ax4 = plt.subplot(3, 3, 4)
    for exp in experiments:
        exp_data = combined_df[combined_df['experiment'] == exp]
        if 'avg_intrinsic_reward' in exp_data.columns:
            ax4.plot(exp_data['checkpoint'] / 1e6, exp_data['avg_intrinsic_reward'], 
                    marker='d', label=exp, linewidth=2, markersize=8, color=color_map[exp])
    ax4.set_xlabel('Training Steps (Millions)', fontsize=12)
    ax4.set_ylabel('Average Intrinsic Reward', fontsize=12)
    ax4.set_title('Intrinsic Reward Evolution', fontsize=14, fontweight='bold')
    ax4.legend(loc='best', fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    # 5. Activity Growth Rate (derivative)
    ax5 = plt.subplot(3, 3, 5)
    for exp in experiments:
        exp_data = combined_df[combined_df['experiment'] == exp].sort_values('checkpoint')
        if len(exp_data) > 1:
            growth_rate = np.diff(exp_data['total_activity']) / np.diff(exp_data['checkpoint'] / 1e6)
            checkpoints_mid = (exp_data['checkpoint'].values[:-1] + exp_data['checkpoint'].values[1:]) / 2 / 1e6
            ax5.plot(checkpoints_mid, growth_rate, 
                    marker='p', label=exp, linewidth=2, markersize=8, color=color_map[exp])
    ax5.set_xlabel('Training Steps (Millions)', fontsize=12)
    ax5.set_ylabel('Activity Growth Rate', fontsize=12)
    ax5.set_title('Exploration Velocity', fontsize=14, fontweight='bold')
    ax5.legend(loc='best', fontsize=10)
    ax5.grid(True, alpha=0.3)
    ax5.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    # 6. Final Performance Comparison (Bar chart)
    ax6 = plt.subplot(3, 3, 6)
    final_data = combined_df[combined_df['checkpoint'] == 2500000]
    if not final_data.empty:
        x_pos = np.arange(len(final_data))
        bars = ax6.bar(x_pos, final_data['total_activity'], 
                      color=[color_map[exp] for exp in final_data['experiment']])
        ax6.set_xticks(x_pos)
        ax6.set_xticklabels(final_data['experiment'], rotation=45, ha='right')
        ax6.set_ylabel('Total Activity Count', fontsize=12)
        ax6.set_title('Final Activity Count (2.5M steps)', fontsize=14, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, final_data['total_activity']):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(value)}', ha='center', va='bottom', fontsize=10)
    
    # 7. Exploration Efficiency (Activity per Unique State)
    ax7 = plt.subplot(3, 3, 7)
    for exp in experiments:
        exp_data = combined_df[combined_df['experiment'] == exp]
        efficiency = exp_data['total_activity'] / exp_data['unique_states'].replace(0, np.nan)
        ax7.plot(exp_data['checkpoint'] / 1e6, efficiency, 
                marker='*', label=exp, linewidth=2, markersize=10, color=color_map[exp])
    ax7.set_xlabel('Training Steps (Millions)', fontsize=12)
    ax7.set_ylabel('Activity per Unique State', fontsize=12)
    ax7.set_title('Exploration Efficiency', fontsize=14, fontweight='bold')
    ax7.legend(loc='best', fontsize=10)
    ax7.grid(True, alpha=0.3)
    
    # 8. Heatmap of Final State Coverage
    ax8 = plt.subplot(3, 3, 8)
    final_coverage = combined_df[combined_df['checkpoint'] == 2500000][['experiment', 'state_coverage']]
    if not final_coverage.empty:
        coverage_matrix = final_coverage.pivot_table(values='state_coverage', 
                                                     index='experiment', 
                                                     aggfunc='mean')
        sns.heatmap(coverage_matrix * 100, annot=True, fmt='.1f', cmap='YlOrRd', 
                   ax=ax8, cbar_kws={'label': 'Coverage (%)'})
        ax8.set_title('Final State Coverage Heatmap', fontsize=14, fontweight='bold')
        ax8.set_ylabel('')
    
    # 9. Summary Statistics Table
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('tight')
    ax9.axis('off')
    
    # Calculate summary statistics for final checkpoint
    summary_data = []
    for exp in experiments:
        exp_final = combined_df[(combined_df['experiment'] == exp) & 
                                (combined_df['checkpoint'] == 2500000)]
        if not exp_final.empty:
            summary_data.append([
                exp,
                f"{int(exp_final['total_activity'].values[0])}",
                f"{int(exp_final['unique_states'].values[0])}",
                f"{exp_final['state_coverage'].values[0]*100:.1f}%"
            ])
    
    if summary_data:
        table = ax9.table(cellText=summary_data,
                         colLabels=['Experiment', 'Total Activity', 'Unique States', 'Coverage'],
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.3, 0.25, 0.25, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Style the header
        for i in range(4):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(summary_data) + 1):
            for j in range(4):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
    
    ax9.set_title('Summary Statistics (2.5M steps)', fontsize=14, fontweight='bold', pad=20)
    
    # Overall title and layout
    fig.suptitle('APT Experiments v2: Integrated Exploration Metrics Analysis', 
                fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the figure
    output_path = results_path / "integrated_exploration_metrics.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nIntegrated visualization saved to: {output_path}")
    
    # Also save as PDF for publication quality
    pdf_path = results_path / "integrated_exploration_metrics.pdf"
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"PDF version saved to: {pdf_path}")
    
    plt.show()
    
    # Save combined data to CSV for further analysis
    csv_path = results_path / "combined_exploration_metrics.csv"
    combined_df.to_csv(csv_path, index=False)
    print(f"Combined data saved to: {csv_path}")
    
    return combined_df

def main():
    """Main function to run the integrated visualization."""
    results_dir = "/Users/kevinhan/mini_behavior/APT_Experiments_v2/results"
    
    print("=" * 60)
    print("APT Experiments v2: Integrated Results Visualization")
    print("=" * 60)
    
    combined_data = create_integrated_visualization(results_dir)
    
    if combined_data is not None:
        print("\n" + "=" * 60)
        print("Analysis Complete!")
        print("=" * 60)
        
        # Print some key insights
        print("\nKey Insights:")
        print("-" * 40)
        
        # Find best performing experiment
        final_data = combined_data[combined_data['checkpoint'] == 2500000]
        if not final_data.empty:
            best_activity = final_data.loc[final_data['total_activity'].idxmax()]
            print(f"Highest Total Activity: {best_activity['experiment']} "
                  f"({int(best_activity['total_activity'])} activities)")
            
            best_coverage = final_data.loc[final_data['state_coverage'].idxmax()]
            print(f"Best State Coverage: {best_coverage['experiment']} "
                  f"({best_coverage['state_coverage']*100:.1f}%)")
            
            best_unique = final_data.loc[final_data['unique_states'].idxmax()]
            print(f"Most Unique States: {best_unique['experiment']} "
                  f"({int(best_unique['unique_states'])} states)")

if __name__ == "__main__":
    main()