#!/usr/bin/env python3
"""
Visualize checkpoint activity data to analyze algorithm performance.
This script creates various visualizations to understand:
- Which objects and states are being explored
- How exploration patterns change over training
- Activity diversity and coverage
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import argparse

def load_checkpoint_data(activity_logs_dir):
    """Load all checkpoint CSV files and return structured data."""
    csv_pattern = os.path.join(activity_logs_dir, "checkpoint_*_activity.csv")
    csv_files = sorted(glob.glob(csv_pattern))
    
    if not csv_files:
        print(f"No checkpoint CSV files found in {activity_logs_dir}")
        return None
    
    checkpoint_data = {}
    
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        # Extract checkpoint number
        try:
            checkpoint_num = int(filename.split('_')[1])
        except:
            continue
            
        # Load CSV and transform to long format
        df_wide = pd.read_csv(csv_file)
        
        # Skip the checkpoint_id column and get activity data
        if len(df_wide) > 0:
            activity_row = df_wide.iloc[0]
            
            # Transform wide format to long format
            data_rows = []
            for col in df_wide.columns:
                if col != 'checkpoint_id':
                    # Parse column name (e.g., "shape_toy_0_thrown")
                    parts = col.rsplit('_', 1)
                    if len(parts) == 2:
                        object_name = parts[0]
                        state_name = parts[1]
                        activity_count = int(activity_row[col])
                        
                        data_rows.append({
                            'object_type': object_name,
                            'state_name': state_name,
                            'activity_count': activity_count
                        })
            
            df = pd.DataFrame(data_rows)
            checkpoint_data[checkpoint_num] = df
        
    return checkpoint_data

def create_activity_heatmap(checkpoint_data, save_dir):
    """Create heatmaps showing activity patterns for each checkpoint."""
    n_checkpoints = len(checkpoint_data)
    fig, axes = plt.subplots(1, n_checkpoints, figsize=(5*n_checkpoints, 8))
    
    if n_checkpoints == 1:
        axes = [axes]
    
    for idx, (checkpoint, df) in enumerate(sorted(checkpoint_data.items())):
        # Create pivot table for heatmap
        pivot = df.pivot_table(
            values='activity_count', 
            index='object_type', 
            columns='state_name',
            fill_value=0,
            aggfunc='sum'
        )
        
        # Create heatmap
        sns.heatmap(pivot, 
                   cmap='YlOrRd', 
                   ax=axes[idx],
                   cbar_kws={'label': 'Activity Count'},
                   fmt='d',
                   annot=True if pivot.shape[0] * pivot.shape[1] < 50 else False)
        
        axes[idx].set_title(f'Checkpoint {checkpoint:,}')
        axes[idx].set_xlabel('State')
        axes[idx].set_ylabel('Object Type')
        
    plt.suptitle('Activity Patterns Across Checkpoints', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'activity_heatmaps.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_exploration_metrics(checkpoint_data, save_dir):
    """Plot various exploration metrics over training."""
    metrics = {
        'checkpoint': [],
        'total_activity': [],
        'unique_pairs': [],
        'entropy': [],
        'max_activity': [],
        'mean_activity': [],
        'active_pairs': []  # Pairs with activity > 0
    }
    
    for checkpoint in sorted(checkpoint_data.keys()):
        df = checkpoint_data[checkpoint]
        
        metrics['checkpoint'].append(checkpoint)
        metrics['total_activity'].append(df['activity_count'].sum())
        metrics['unique_pairs'].append(len(df))
        metrics['active_pairs'].append((df['activity_count'] > 0).sum())
        metrics['max_activity'].append(df['activity_count'].max())
        metrics['mean_activity'].append(df['activity_count'].mean())
        
        # Calculate entropy of activity distribution
        counts = df['activity_count'].values
        if counts.sum() > 0:
            probs = counts / counts.sum()
            probs = probs[probs > 0]  # Remove zeros for log
            entropy = -np.sum(probs * np.log(probs))
        else:
            entropy = 0
        metrics['entropy'].append(entropy)
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Plot each metric
    plot_configs = [
        ('total_activity', 'Total Activity Count', 'Total interactions across all object-state pairs'),
        ('active_pairs', 'Active Object-State Pairs', 'Number of pairs with at least one interaction'),
        ('entropy', 'Activity Entropy', 'Diversity of exploration (higher = more uniform)'),
        ('max_activity', 'Maximum Activity Count', 'Highest activity count for any single pair'),
        ('mean_activity', 'Mean Activity Count', 'Average activity across all pairs'),
        ('unique_pairs', 'Total Object-State Pairs', 'Total number of possible pairs')
    ]
    
    for idx, (metric, title, desc) in enumerate(plot_configs):
        ax = axes[idx]
        ax.plot(metrics['checkpoint'], metrics[metric], 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Training Steps')
        ax.set_ylabel(title)
        ax.set_title(f'{title}\n({desc})', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.ticklabel_format(style='plain', axis='x')
        ax.set_xticks(metrics['checkpoint'])
        ax.set_xticklabels([f'{c/1e6:.1f}M' for c in metrics['checkpoint']], rotation=45)
    
    plt.suptitle('Exploration Metrics Over Training', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'exploration_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_object_state_breakdown(checkpoint_data, save_dir):
    """Plot breakdown of activities by object type and state."""
    # Aggregate data across all checkpoints
    all_objects = defaultdict(int)
    all_states = defaultdict(int)
    
    for df in checkpoint_data.values():
        object_counts = df.groupby('object_type')['activity_count'].sum()
        state_counts = df.groupby('state_name')['activity_count'].sum()
        
        for obj, count in object_counts.items():
            all_objects[obj] += count
        for state, count in state_counts.items():
            all_states[state] += count
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot object breakdown
    objects = sorted(all_objects.items(), key=lambda x: x[1], reverse=True)[:20]
    if objects:
        obj_names, obj_counts = zip(*objects)
        ax1.barh(obj_names, obj_counts)
        ax1.set_xlabel('Total Activity Count')
        ax1.set_title('Top 20 Most Interacted Objects')
        ax1.invert_yaxis()
    
    # Plot state breakdown
    states = sorted(all_states.items(), key=lambda x: x[1], reverse=True)[:20]
    if states:
        state_names, state_counts = zip(*states)
        ax2.barh(state_names, state_counts)
        ax2.set_xlabel('Total Activity Count')
        ax2.set_title('Top 20 Most Changed States')
        ax2.invert_yaxis()
    
    plt.suptitle('Activity Breakdown by Object Type and State', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'object_state_breakdown.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_activity_evolution(checkpoint_data, save_dir, top_n=10):
    """Plot how activity counts evolve for top object-state pairs."""
    # Find top N most active pairs across all checkpoints
    total_activity = defaultdict(int)
    
    for df in checkpoint_data.values():
        for _, row in df.iterrows():
            pair = f"{row['object_type']}_{row['state_name']}"
            total_activity[pair] += row['activity_count']
    
    top_pairs = sorted(total_activity.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_pair_names = [pair[0] for pair in top_pairs]
    
    # Track evolution of these pairs
    evolution_data = defaultdict(lambda: defaultdict(int))
    
    for checkpoint in sorted(checkpoint_data.keys()):
        df = checkpoint_data[checkpoint]
        for _, row in df.iterrows():
            pair = f"{row['object_type']}_{row['state_name']}"
            if pair in top_pair_names:
                evolution_data[pair][checkpoint] = row['activity_count']
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    for pair in top_pair_names:
        checkpoints = sorted(evolution_data[pair].keys())
        counts = [evolution_data[pair][cp] for cp in checkpoints]
        
        # Make label more readable
        obj, state = pair.split('_', 1)
        label = f"{obj} - {state}"
        
        plt.plot(checkpoints, counts, 'o-', label=label, linewidth=2, markersize=6)
    
    plt.xlabel('Training Steps')
    plt.ylabel('Activity Count')
    plt.title(f'Evolution of Top {top_n} Most Active Object-State Pairs')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Format x-axis
    plt.gca().ticklabel_format(style='plain', axis='x')
    xticks = plt.gca().get_xticks()
    plt.gca().set_xticklabels([f'{x/1e6:.1f}M' for x in xticks], rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'activity_evolution.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_summary_report(checkpoint_data, save_dir):
    """Generate a text summary report."""
    report_path = os.path.join(save_dir, 'checkpoint_analysis_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("CHECKPOINT ACTIVITY ANALYSIS REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Number of checkpoints analyzed: {len(checkpoint_data)}\n")
        f.write(f"Checkpoints: {sorted(checkpoint_data.keys())}\n\n")
        
        # Overall statistics
        total_activities = sum(df['activity_count'].sum() for df in checkpoint_data.values())
        f.write(f"Total activities across all checkpoints: {total_activities:,}\n\n")
        
        # Per-checkpoint analysis
        f.write("PER-CHECKPOINT ANALYSIS\n")
        f.write("-"*70 + "\n")
        
        for checkpoint in sorted(checkpoint_data.keys()):
            df = checkpoint_data[checkpoint]
            
            f.write(f"\nCheckpoint {checkpoint:,}:\n")
            f.write(f"  Total activity: {df['activity_count'].sum():,}\n")
            f.write(f"  Active pairs: {(df['activity_count'] > 0).sum()}/{len(df)}\n")
            f.write(f"  Max activity count: {df['activity_count'].max()}\n")
            f.write(f"  Mean activity: {df['activity_count'].mean():.2f}\n")
            f.write(f"  Std activity: {df['activity_count'].std():.2f}\n")
            
            # Top 5 most active pairs
            top_5 = df.nlargest(5, 'activity_count')
            if not top_5.empty:
                f.write("  Top 5 most active pairs:\n")
                for _, row in top_5.iterrows():
                    f.write(f"    - {row['object_type']} / {row['state_name']}: {row['activity_count']}\n")
        
        # Exploration coverage
        f.write("\n\nEXPLORATION COVERAGE\n")
        f.write("-"*70 + "\n")
        
        # Track which pairs were ever explored
        all_pairs = set()
        explored_pairs = set()
        
        for df in checkpoint_data.values():
            for _, row in df.iterrows():
                pair = (row['object_type'], row['state_name'])
                all_pairs.add(pair)
                if row['activity_count'] > 0:
                    explored_pairs.add(pair)
        
        coverage = len(explored_pairs) / len(all_pairs) * 100 if all_pairs else 0
        f.write(f"Total possible object-state pairs: {len(all_pairs)}\n")
        f.write(f"Pairs explored at least once: {len(explored_pairs)}\n")
        f.write(f"Exploration coverage: {coverage:.1f}%\n")
        
        # Never explored pairs
        never_explored = all_pairs - explored_pairs
        if never_explored:
            f.write(f"\nNever explored pairs ({len(never_explored)}):\n")
            for obj, state in sorted(never_explored)[:20]:  # Show first 20
                f.write(f"  - {obj} / {state}\n")
            if len(never_explored) > 20:
                f.write(f"  ... and {len(never_explored) - 20} more\n")
    
    print(f"Summary report saved to: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Visualize checkpoint activity data')
    parser.add_argument('activity_logs_dir', help='Path to the activity_logs directory')
    parser.add_argument('--output-dir', '-o', default='checkpoint_visualizations',
                       help='Directory to save visualizations (default: checkpoint_visualizations)')
    parser.add_argument('--top-n', type=int, default=10,
                       help='Number of top pairs to show in evolution plot (default: 10)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load checkpoint data
    print(f"Loading checkpoint data from {args.activity_logs_dir}...")
    checkpoint_data = load_checkpoint_data(args.activity_logs_dir)
    
    if not checkpoint_data:
        return
    
    print(f"Loaded {len(checkpoint_data)} checkpoints")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    print("  - Creating activity heatmaps...")
    create_activity_heatmap(checkpoint_data, args.output_dir)
    
    print("  - Plotting exploration metrics...")
    plot_exploration_metrics(checkpoint_data, args.output_dir)
    
    print("  - Creating object/state breakdown...")
    plot_object_state_breakdown(checkpoint_data, args.output_dir)
    
    print("  - Plotting activity evolution...")
    plot_activity_evolution(checkpoint_data, args.output_dir, args.top_n)
    
    print("  - Generating summary report...")
    generate_summary_report(checkpoint_data, args.output_dir)
    
    print(f"\nAll visualizations saved to: {args.output_dir}/")
    print("\nGenerated files:")
    print("  - activity_heatmaps.png: Heatmaps of object-state interactions per checkpoint")
    print("  - exploration_metrics.png: Various metrics showing exploration progress")
    print("  - object_state_breakdown.png: Most interacted objects and states")
    print("  - activity_evolution.png: How top pairs evolve over training")
    print("  - checkpoint_analysis_report.txt: Detailed text summary")

if __name__ == "__main__":
    main()

"""
Example usage:
python post_processing/visualize_checkpoints.py checkpoints/activity_logs/
python post_processing/visualize_checkpoints.py checkpoints/activity_logs/ --output-dir my_analysis --top-n 15
"""