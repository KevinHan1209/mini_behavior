#!/usr/bin/env python3
"""
Visualize intrinsic rewards from APT training to diagnose performance issues.
Can read from both CSV logs and checkpoint files.
"""

import os
import argparse
import pandas as pd
import torch
import numpy as np
from pathlib import Path

def load_intrinsic_rewards_from_csv(csv_path):
    """Load intrinsic rewards from CSV log file."""
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    return df

def load_intrinsic_rewards_from_checkpoints(checkpoint_dir):
    """Load intrinsic rewards from checkpoint files."""
    checkpoint_files = sorted(Path(checkpoint_dir).glob("checkpoint_*.pt"))
    
    if not checkpoint_files:
        print(f"No checkpoint files found in {checkpoint_dir}")
        return None
    
    data = []
    for ckpt_file in checkpoint_files:
        try:
            checkpoint = torch.load(ckpt_file, map_location='cpu')
            if 'intrinsic_rewards_history' in checkpoint:
                step = checkpoint.get('global_step', int(ckpt_file.stem.split('_')[1]))
                last_reward = checkpoint.get('last_intrinsic_reward', None)
                last_std = checkpoint.get('last_intrinsic_std', None)
                history = checkpoint.get('intrinsic_rewards_history', [])
                
                data.append({
                    'checkpoint': ckpt_file.name,
                    'global_step': step,
                    'last_intrinsic_reward': last_reward,
                    'last_intrinsic_std': last_std,
                    'history_length': len(history),
                    'history_mean': np.mean(history) if history else None,
                    'history_std': np.std(history) if history else None,
                    'history_min': min(history) if history else None,
                    'history_max': max(history) if history else None
                })
        except Exception as e:
            print(f"Error loading {ckpt_file}: {e}")
    
    return pd.DataFrame(data) if data else None

def analyze_intrinsic_rewards(df=None, checkpoint_df=None):
    """Analyze intrinsic reward trends."""
    print("\n" + "="*70)
    print("INTRINSIC REWARD ANALYSIS")
    print("="*70)
    print("\nNOTE: Rewards are normalized (mean-centered) for training stability.")
    print("Negative values indicate below-average novelty, not 'bad' performance.")
    
    if df is not None and not df.empty:
        print("\nFrom CSV Log:")
        print(f"Total updates logged: {len(df)}")
        print(f"Global steps range: {df['global_step'].min():,} - {df['global_step'].max():,}")
        
        # Overall statistics
        print(f"\nAverage intrinsic reward: {df['avg_intrinsic_reward'].mean():.4f}")
        print(f"Std of intrinsic rewards: {df['avg_intrinsic_reward'].std():.4f}")
        print(f"Min intrinsic reward: {df['avg_intrinsic_reward'].min():.4f}")
        print(f"Max intrinsic reward: {df['avg_intrinsic_reward'].max():.4f}")
        
        # Trend analysis
        first_quarter = df.head(len(df)//4)['avg_intrinsic_reward'].mean()
        last_quarter = df.tail(len(df)//4)['avg_intrinsic_reward'].mean()
        
        print(f"\nFirst quarter avg: {first_quarter:.4f}")
        print(f"Last quarter avg: {last_quarter:.4f}")
        print(f"Change: {last_quarter - first_quarter:.4f} ({(last_quarter/first_quarter - 1)*100:.1f}%)")
        
        # Check if rewards are increasing
        correlation = df['update'].corr(df['avg_intrinsic_reward'])
        print(f"\nCorrelation between update and reward: {correlation:.4f}")
        if correlation > 0.1:
            print("✓ Intrinsic rewards are generally INCREASING")
        elif correlation < -0.1:
            print("✗ Intrinsic rewards are generally DECREASING")
        else:
            print("- Intrinsic rewards are relatively STABLE")
        
        # Plot trend in ASCII
        print("\nReward Trend (ASCII plot):")
        plot_ascii_trend(df['avg_intrinsic_reward'].values)
    
    if checkpoint_df is not None and not checkpoint_df.empty:
        print("\n\nFrom Checkpoints:")
        print(f"Total checkpoints: {len(checkpoint_df)}")
        
        # Sort by global step for proper ordering
        checkpoint_df = checkpoint_df.sort_values('global_step')
        
        # Calculate additional metrics
        print("\nDetailed Checkpoint Analysis:")
        print(f"{'Checkpoint':<20} {'Step':<10} {'Last Reward':<12} {'History Mean':<12} {'History Std':<12} {'Min':<10} {'Max':<10}")
        print("-"*104)
        
        for _, row in checkpoint_df.iterrows():
            print(f"{row['checkpoint']:<20} {row['global_step']:<10,} ", end="")
            
            if pd.notna(row['last_intrinsic_reward']):
                print(f"{row['last_intrinsic_reward']:<12.4f} ", end="")
            else:
                print(f"{'N/A':<12} ", end="")
                
            if pd.notna(row['history_mean']):
                print(f"{row['history_mean']:<12.4f} ", end="")
            else:
                print(f"{'N/A':<12} ", end="")
                
            if pd.notna(row['history_std']):
                print(f"{row['history_std']:<12.4f} ", end="")
            else:
                print(f"{'N/A':<12} ", end="")
                
            if pd.notna(row['history_min']):
                print(f"{row['history_min']:<10.4f} ", end="")
            else:
                print(f"{'N/A':<10} ", end="")
                
            if pd.notna(row['history_max']):
                print(f"{row['history_max']:<10.4f}")
            else:
                print("N/A")
        
        # Trend analysis
        print("\nTrend Analysis Across Checkpoints:")
        last_rewards = checkpoint_df['last_intrinsic_reward'].dropna()
        history_means = checkpoint_df['history_mean'].dropna()
        
        if len(last_rewards) > 1:
            # Calculate trend
            first_last = last_rewards.iloc[0]
            final_last = last_rewards.iloc[-1]
            print(f"Last reward trend: {first_last:.4f} → {final_last:.4f} (Δ = {final_last - first_last:+.4f})")
            
            # Check if rewards are recovering
            mid_point = len(last_rewards) // 2
            mid_last = last_rewards.iloc[mid_point]
            
            if first_last > mid_last and final_last > mid_last:
                print("Pattern: Initial exploration → Exploitation → Recovery (U-shaped curve)")
            elif first_last > mid_last > final_last:
                print("Pattern: Declining exploration (agent may be converging)")
            elif first_last < mid_last < final_last:
                print("Pattern: Increasing exploration (agent discovering new areas)")
            else:
                print("Pattern: Irregular (non-monotonic exploration)")
        
        if len(history_means) > 1:
            first_mean = history_means.iloc[0]
            final_mean = history_means.iloc[-1]
            print(f"History mean trend: {first_mean:.4f} → {final_mean:.4f} (Δ = {final_mean - first_mean:+.4f})")
        
        # Visualize trend
        if len(last_rewards) > 0:
            print("\nIntrinsic Reward Trend (Last Rewards):")
            values = last_rewards.values
            steps = checkpoint_df.loc[last_rewards.index, 'global_step'].values
            plot_ascii_trend_with_labels(values, steps)

def plot_ascii_trend(values, width=70, height=20):
    """Create a simple ASCII plot of the trend."""
    if len(values) == 0:
        return
    
    # Downsample if too many values
    if len(values) > width:
        indices = np.linspace(0, len(values)-1, width, dtype=int)
        values = values[indices]
    
    min_val = np.min(values)
    max_val = np.max(values)
    range_val = max_val - min_val
    
    if range_val == 0:
        range_val = 1
    
    # Normalize to height
    normalized = ((values - min_val) / range_val * (height - 1)).astype(int)
    
    # Create plot
    plot = [[' ' for _ in range(len(values))] for _ in range(height)]
    
    for i, val in enumerate(normalized):
        plot[height - 1 - val][i] = '█'
    
    # Add axis
    print(f"  {max_val:.4f} |")
    for row in plot:
        print("         |" + ''.join(row))
    print(f"  {min_val:.4f} |" + "─" * len(values))
    print(f"         0" + " " * (len(values) - 10) + f"{len(values)}")
    print("           Update Number →")

def plot_ascii_trend_with_labels(values, labels, width=70, height=20):
    """Create an ASCII plot with step labels."""
    if len(values) == 0:
        return
    
    # Ensure we have the same number of values and labels
    if len(values) != len(labels):
        return plot_ascii_trend(values, width, height)
    
    min_val = np.min(values)
    max_val = np.max(values)
    range_val = max_val - min_val
    
    if range_val == 0:
        range_val = 1
    
    # Normalize to height
    normalized = ((values - min_val) / range_val * (height - 1)).astype(int)
    
    # Create plot
    plot = [[' ' for _ in range(len(values))] for _ in range(height)]
    
    for i, val in enumerate(normalized):
        plot[height - 1 - val][i] = '█'
    
    # Add axis with values
    print(f"  {max_val:>8.4f} |")
    for row in plot:
        print("            |" + '  '.join(row))
    print(f"  {min_val:>8.4f} |" + "─" * (len(values) * 2))
    
    # Add step labels
    print("            ", end="")
    for i, label in enumerate(labels):
        if i == 0:
            print(f"{label/1000:.0f}k", end="")
        elif i == len(labels) - 1:
            print(f"{' ' * (2 * (len(labels) - i - 1))}{label/1000:.0f}k", end="")
        else:
            continue
    print("\n            Steps (in thousands) →")

def main():
    parser = argparse.ArgumentParser(description='Analyze intrinsic rewards from APT training')
    parser.add_argument('--csv', type=str, default='intrinsic_rewards_log.csv',
                       help='Path to intrinsic rewards CSV log')
    parser.add_argument('--checkpoints', type=str, default='checkpoints',
                       help='Path to checkpoint directory')
    parser.add_argument('--no-csv', action='store_true',
                       help='Skip CSV analysis')
    parser.add_argument('--no-checkpoints', action='store_true',
                       help='Skip checkpoint analysis')
    
    args = parser.parse_args()
    
    df = None
    checkpoint_df = None
    
    # Load from CSV
    if not args.no_csv:
        print(f"Loading intrinsic rewards from CSV: {args.csv}")
        df = load_intrinsic_rewards_from_csv(args.csv)
    
    # Load from checkpoints
    if not args.no_checkpoints:
        print(f"Loading intrinsic rewards from checkpoints: {args.checkpoints}")
        checkpoint_df = load_intrinsic_rewards_from_checkpoints(args.checkpoints)
    
    # Analyze
    if df is None and checkpoint_df is None:
        print("No data found to analyze!")
        return
    
    analyze_intrinsic_rewards(df, checkpoint_df)

if __name__ == "__main__":
    main()

"""
Example usage:
python post_processing/visualize_intrinsic_rewards.py
python post_processing/visualize_intrinsic_rewards.py --csv path/to/intrinsic_rewards_log.csv
python post_processing/visualize_intrinsic_rewards.py --checkpoints path/to/checkpoints
python post_processing/visualize_intrinsic_rewards.py --no-checkpoints  # Only analyze CSV
"""