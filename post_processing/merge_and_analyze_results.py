# comprehensive_merge_and_analyze_results.py
import pandas as pd
import numpy as np
import pickle
from scipy.stats import entropy
import os
import re
import argparse
import glob
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns

# Object name mapping
obj_name_mapping = {
    'a': 'alligator busy box',
    'b': 'broom',
    'bs': 'broom set', 
    'bu': 'bucket',
    'ca': 'cart',
    'cu': 'cube',
    'f': 'farm toy',
    'g': 'gear',
    'm': 'music toy',
    'p': 'piggie bank',
    'pb': 'pink beach ball',
    'r': 'rattle',
    'rb': 'red spiky ball',
    's': 'stroller',
    'sh': 'shape sorter',
    't': 'tree busy box',
    'c': 'winnie cabinet',
    'y': 'yellow donut ring'
}

def transform_object_name(obj_name):
    """Transform object name by replacing underscores with spaces and removing trailing numbers"""
    # Special case for beach ball
    if obj_name.startswith('beach_ball'):
        return 'pink beach ball'
    
    # Replace underscores with spaces
    obj_name = obj_name.replace('_', ' ')
    # Remove trailing numbers and any remaining underscores
    obj_name = re.sub(r'_\d+$', '', obj_name)
    return obj_name

def load_agent_distributions(pkl_path="post_processing/averaged_state_distributions.pkl"):
    """Load the averaged distributions from the pickle file and convert to flat distribution"""
    with open(pkl_path, 'rb') as f:
        agent_dists = pickle.load(f)
    
    # Convert to flat distribution
    flat_dist = {}
    for obj, states in agent_dists.items():
        # Transform object name using mapping
        if obj in obj_name_mapping:
            obj = obj_name_mapping[obj]
        for state, percentages in states.items():
            # Only keep the false percentage
            if isinstance(percentages, dict):
                false_percentage = percentages.get('false_percentage', 100)
            else:
                false_percentage = percentages
            flat_dist[(obj, state)] = false_percentage
    
    return flat_dist

def extract_metadata_from_path(csv_path):
    """Extract metadata (seed, entropy, step, episode) from CSV file path"""
    path_parts = Path(csv_path).parts
    
    metadata = {}
    
    for part in path_parts:
        if part.startswith('rnd_'):
            parts = part.split('_')
            
            metadata['seed'] = None
            metadata['entropy'] = None
            metadata['step'] = None
            
            i = 0
            while i < len(parts):
                p = parts[i]
                
                if 'x' in p and p.replace('x', '').replace('8', '').replace('6', '').replace('4', '') == '':
                    i += 1
                    continue
                
                if p.startswith('freq'):
                    freq_value = p[4:]  # Extract number after 'freq'
                    metadata['frequency'] = int(freq_value) if freq_value.isdigit() else None
                    i += 1
                    continue
                
                if p.startswith('decay'):
                    decay_base = p[5:]  # Get part after 'decay'
                    decay_parts = [decay_base]
                    
                    j = i + 1
                    while j < len(parts):
                        next_part = parts[j]
                        if next_part.isdigit() and len(next_part) <= 4:
                            decay_parts.append(next_part)
                            j += 1
                        else:
                            break
                    
                    if len(decay_parts) == 1:
                        metadata['decay'] = float(decay_base) if decay_base.replace('.', '').isdigit() else 0.0
                    else:
                        if decay_parts[0] == '0':
                            decimal_part = ''.join(decay_parts[1:])
                            metadata['decay'] = float(f"0.{decimal_part}")
                        else:
                            metadata['decay'] = float(decay_parts[0])
                    
                    i = j  
                    continue
                
                if p.startswith('seed'):
                    seed_value = p[4:]
                    metadata['seed'] = int(seed_value) if seed_value.isdigit() else None
                    i += 1
                    continue
                
                if p.startswith('step'):
                    step_value = p[4:]
                    metadata['step'] = step_value  # Keep as string in case it's "final"
                    i += 1
                    continue
                
                if p.startswith('ent'):
                    ent_value = p[3:]
                    try:
                        if '.' in ent_value:
                            metadata['entropy'] = float(ent_value)
                        elif '_' in ent_value:
                            ent_parts = ent_value.split('_')
                            if len(ent_parts) == 2 and ent_parts[0].isdigit() and ent_parts[1].isdigit():
                                metadata['entropy'] = float(f"{ent_parts[0]}.{ent_parts[1]}")
                            else:
                                metadata['entropy'] = float(ent_value.replace('_', '.'))
                        else:
                            metadata['entropy'] = float(ent_value) if ent_value.replace('.', '').isdigit() else None
                    except ValueError:
                        metadata['entropy'] = None
                    i += 1
                    continue
                
                i += 1
            
            break 
    
    filename = Path(csv_path).name
    if filename.startswith('episode_'):
        try:
            episode_num = filename.split('_')[1]
            metadata['episode'] = int(episode_num)
        except (IndexError, ValueError):
            pass
    
    if 'frequency' not in metadata or metadata['frequency'] is None:
        metadata['frequency'] = 1  # Default frequency
    if 'decay' not in metadata or metadata['decay'] is None:
        metadata['decay'] = 0.0  # Default decay
    
    return metadata

def merge_all_activity_csvs(results_dir):
    """Merge all activity CSV files from the results directory with robust error handling"""
    pattern = os.path.join(results_dir, "**/csvs/episode_*_activity.csv")
    csv_files = glob.glob(pattern, recursive=True)
    
    if not csv_files:
        print(f"No activity CSV files found in {results_dir}")
        return None
    
    print(f"Found {len(csv_files)} activity CSV files")
    
    merged_data = []
    skipped_files = []
    processed_files = []
    
    for csv_path in csv_files:
        try:
            metadata = extract_metadata_from_path(csv_path)
            
            required_fields = ['seed', 'step', 'episode']
            missing_fields = [field for field in required_fields if metadata.get(field) is None]
            
            if missing_fields:
                if metadata.get('episode') is None:
                    print(f"  Skipping due to missing episode number")
                    skipped_files.append(csv_path)
                    continue
            
            df = pd.read_csv(csv_path)
            
            for key, value in metadata.items():
                df[key] = value
            
            merged_data.append(df)
            processed_files.append(csv_path)
            
            if len(processed_files) <= 5:
                print(f"  Processed: {Path(csv_path).name} -> seed={metadata.get('seed')}, "
                      f"frequency={metadata.get('frequency')}, decay={metadata.get('decay')}, "
                      f"step={metadata.get('step')}, episode={metadata.get('episode')}")
            
        except Exception as e:
            print(f"Error processing {csv_path}: {e}")
            skipped_files.append(csv_path)
            continue
    
    if not merged_data:
        print("No valid CSV files could be processed")
        return None
    
    merged_df = pd.concat(merged_data, ignore_index=True)
    
    print(f"\nMerged data shape: {merged_df.shape}")
    print(f"Successfully processed: {len(merged_data)} files")
    print(f"Skipped files: {len(skipped_files)}")
    
    if len(merged_df) > 0:
        print(f"Seeds found: {sorted([s for s in merged_df['seed'].unique() if s is not None])}")
        print(f"Frequencies found: {sorted([f for f in merged_df['frequency'].unique() if f is not None])}")
        print(f"Decay values found: {sorted([d for d in merged_df['decay'].unique() if d is not None])}")
        print(f"Steps found: {sorted([s for s in merged_df['step'].unique() if s is not None])}")
        print(f"Episodes found: {sorted([e for e in merged_df['episode'].unique() if e is not None])}")
    
    return merged_df

def find_action_probability_files(results_dir):
    """Find all action probability pickle files in the results directory"""
    pattern = os.path.join(results_dir, "**/action_probabilities/action_probs_step_*.pkl")
    action_files = glob.glob(pattern, recursive=True)
    
    print(f"Found {len(action_files)} action probability files")
    return action_files

def load_action_probability_data(action_files, verbose=False):
    """Load all action probability data and organize by configuration"""
    all_action_data = {}
    
    for file_path in action_files:
        try:
            metadata = extract_metadata_from_path(file_path)
            
            with open(file_path, 'rb') as f:
                action_data = pickle.load(f)
            
            config_key = (
                metadata.get('frequency', 1),
                metadata.get('decay', 0.0),
                metadata.get('seed', 'unknown')
            )
            
            if config_key not in all_action_data:
                all_action_data[config_key] = []
            
            action_data['step'] = metadata.get('step', 'unknown')
            action_data['metadata'] = metadata
            
            all_action_data[config_key].append(action_data)
            
            if verbose:
                print(f"Loaded: freq={metadata.get('frequency')}, "
                      f"decay={metadata.get('decay')}, "
                      f"seed={metadata.get('seed')}, "
                      f"step={metadata.get('step')}")
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    for config_key in all_action_data:
        all_action_data[config_key].sort(key=lambda x: x['step'] if isinstance(x['step'], int) else 999999)
    
    return all_action_data

def prepare_action_plot_data(all_action_data):
    """Convert action probability data into format suitable for plotting"""
    plot_data = []
    
    total_data_points = 0
    
    for config_key, checkpoints in all_action_data.items():
        freq, decay, seed = config_key
        
        for checkpoint_data in checkpoints:
            step = checkpoint_data['step']
            checkpoint_data_points = 0
            
            for episode_data in checkpoint_data['episodes_data']:
                episode_action_probs = episode_data['action_probs']
                
                for step_data in episode_action_probs:
                    action_probs = step_data['action_probs']
                    
                    # For each action dimension
                    for dim_idx, dim_probs in enumerate(action_probs):
                        # For each action in this dimension
                        for action_idx, prob in enumerate(dim_probs):
                            plot_data.append({
                                'frequency': freq,
                                'decay': decay,
                                'seed': seed,
                                'checkpoint_step': step,
                                'action_dimension': dim_idx,
                                'action_index': action_idx,
                                'probability': prob,
                                'config': f"freq{freq}_decay{str(decay).replace('.', '_')}_seed{seed}",
                                'action_label': f"Dim{dim_idx}_Act{action_idx}"
                            })
                            checkpoint_data_points += 1
            
            total_data_points += checkpoint_data_points
            print(f"  Config freq={freq}, decay={decay}, seed={seed}, step={step}: {checkpoint_data_points} data points")
    
    print(f"Total action probability data points: {total_data_points}")
    return pd.DataFrame(plot_data)

def get_action_names():
    """Define action names for each dimension based on the actual MiniBehaviorEnv"""
    action_names = {
        0: [  # Left arm manipulation actions (ObjectActions)
            "Toggle",
            "Pickup 0", 
            "Drop 0",
            "Throw 0",
            "Push",
            "Pull",
            "Noise Toggle",
            "Takeout",
            "Drop In",
            "Assemble",
            "Disassemble", 
            "Hit",
            "Hit With Object",
            "Mouthing",
            "Brush"
        ],
        1: [  # Right arm manipulation actions (ObjectActions) - same as left
            "Toggle",
            "Pickup 0",
            "Drop 0", 
            "Throw 0",
            "Push",
            "Pull",
            "Noise Toggle",
            "Takeout",
            "Drop In",
            "Assemble",
            "Disassemble",
            "Hit", 
            "Hit With Object",
            "Mouthing",
            "Brush"
        ],
        2: [  # Locomotion actions (LocoActions)
            "Turn Left",
            "Turn Right",
            "Move Forward", 
            "Kick",
            "Climb"
        ]
    }
    return action_names

def create_action_probability_stacked_bars(plot_df, output_dir):
    """Create stacked bar charts showing action probability evolution for each configuration"""
    
    if len(plot_df) == 0:
        print("No action probability data to plot")
        return
    
    action_names = get_action_names()
    
    action_plots_dir = os.path.join(output_dir, 'action_probability_plots')
    os.makedirs(action_plots_dir, exist_ok=True)
    
    configs = plot_df['config'].unique()
    action_dims = sorted(plot_df['action_dimension'].unique())
    
    print(f"Creating stacked bar charts for {len(configs)} configurations")
    print(f"Action dimensions: {action_dims}")
    
    for config in configs:
        config_df = plot_df[plot_df['config'] == config]
        
        # Get metadata for this config
        freq = config_df['frequency'].iloc[0]
        decay = config_df['decay'].iloc[0]
        seed = config_df['seed'].iloc[0]
        
        print(f"Processing config: {config}")
        
        fig, axes = plt.subplots(len(action_dims), 1, figsize=(15, 8 * len(action_dims)))
        if len(action_dims) == 1:
            axes = [axes]
        
        fig.suptitle(f'Frequency: {freq}, Decay: {decay}, Seed: {seed}', fontsize=12)
        
        for dim_idx, ax in zip(action_dims, axes):
            dim_df = config_df[config_df['action_dimension'] == dim_idx]
            
            if len(dim_df) == 0:
                ax.set_title(f'Action Dimension {dim_idx} - No Data')
                continue
            
            # Calculate mean probabilities for each checkpoint step and action
            prob_summary = dim_df.groupby(['checkpoint_step', 'action_index'])['probability'].mean().reset_index()
            
            # Pivot to get actions as columns
            prob_pivot = prob_summary.pivot(index='checkpoint_step', columns='action_index', values='probability')
            prob_pivot = prob_pivot.fillna(0)  # Fill any missing values with 0
            
            def sort_key(x):
                if x == 'final':
                    return 999999999
                return int(x)
            
            prob_pivot = prob_pivot.reindex(sorted(prob_pivot.index, key=sort_key))
            
            # Create stacked bar chart with action names
            colors = plt.cm.tab20(np.linspace(0, 1, len(prob_pivot.columns)))
            bottom = np.zeros(len(prob_pivot))
            
            for i, action_idx in enumerate(prob_pivot.columns):
                probs = prob_pivot[action_idx].values
                
                if dim_idx in action_names and action_idx < len(action_names[dim_idx]):
                    action_name = action_names[dim_idx][action_idx]
                else:
                    action_name = f"Action {action_idx}"
                
                ax.bar(range(len(prob_pivot)), probs, bottom=bottom, 
                      label=action_name, color=colors[i], alpha=0.8, edgecolor='white', linewidth=0.5)
                bottom += probs
            
            dimension_names = {
                0: "Left Arm Actions", 
                1: "Right Arm Actions", 
                2: "Locomotion Actions"
            }
            dim_name = dimension_names.get(dim_idx, f"Action Dimension {dim_idx}")
            ax.set_title(f'{dim_name}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Training Checkpoint', fontsize=12)
            ax.set_ylabel('Action Probability', fontsize=12)
            ax.set_ylim(0, 1.0)
            
            step_labels = []
            for step in prob_pivot.index:
                if step == 'final':
                    step_labels.append('Final')
                else:
                    step_millions = int(step) / 1000000
                    if step_millions >= 1:
                        step_labels.append(f'{step_millions:.1f}M')
                    else:
                        step_labels.append(f'{int(step)//1000}K')
            
            ax.set_xticks(range(len(step_labels)))
            ax.set_xticklabels(step_labels, rotation=45)
            
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            
            ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, linewidth=1)
            
            ax.grid(True, alpha=0.3, axis='y')
            
            for j, step_idx in enumerate(range(len(prob_pivot))):
                cumulative = 0
                for action_idx in prob_pivot.columns:
                    prob = prob_pivot.iloc[step_idx][action_idx]
                    if prob > 0.10:  # Only label if probability > 10%
                        ax.text(j, cumulative + prob/2, f'{prob:.0%}', 
                               ha='center', va='center', fontsize=8)
                    cumulative += prob
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"stacked_bars_{config}.png"
        plot_path = os.path.join(action_plots_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {plot_filename}")

def create_summary_action_plots(plot_df, output_dir):
    """Create summary plots comparing all configurations"""
    
    action_names = get_action_names()
    action_plots_dir = os.path.join(output_dir, 'action_probability_plots')
    os.makedirs(action_plots_dir, exist_ok=True)
    
    action_dims = sorted(plot_df['action_dimension'].unique())
    
    fig, axes = plt.subplots(len(action_dims), 1, figsize=(15, 6 * len(action_dims)))
    if len(action_dims) == 1:
        axes = [axes]
    
    for dim_idx, ax in zip(action_dims, axes):
        dim_df = plot_df[plot_df['action_dimension'] == dim_idx]
        
        # Calculate entropy for each config and checkpoint
        entropy_data = []
        
        for config in dim_df['config'].unique():
            config_df = dim_df[dim_df['config'] == config]
            
            for step in config_df['checkpoint_step'].unique():
                step_df = config_df[config_df['checkpoint_step'] == step]
                
                # Calculate entropy for this step
                prob_summary = step_df.groupby('action_index')['probability'].mean()
                probs = prob_summary.values
                probs = probs / probs.sum()  # Normalize to ensure sum = 1
                entropy_val = -np.sum(probs * np.log(probs + 1e-10))  # Add small epsilon
                
                entropy_data.append({
                    'config': config,
                    'checkpoint_step': step,
                    'entropy': entropy_val,
                    'frequency': config_df['frequency'].iloc[0],
                    'decay': config_df['decay'].iloc[0],
                    'seed': config_df['seed'].iloc[0]
                })
        
        entropy_df = pd.DataFrame(entropy_data)
        
        # Plot entropy evolution for each config
        colors = plt.cm.tab10(np.linspace(0, 1, len(entropy_df['config'].unique())))
        
        for i, config in enumerate(entropy_df['config'].unique()):
            config_entropy = entropy_df[entropy_df['config'] == config]
            
            # Sort by checkpoint step
            def sort_key(x):
                if x == 'final':
                    return 999999999
                return int(x)
            
            config_entropy = config_entropy.sort_values('checkpoint_step', key=lambda x: x.map(sort_key))
            
            # Create readable config label
            freq = config_entropy['frequency'].iloc[0]
            decay = config_entropy['decay'].iloc[0]
            seed = config_entropy['seed'].iloc[0]
            config_label = f"Freq:{freq}, Decay:{decay}, Seed:{seed}"
            
            ax.plot(range(len(config_entropy)), config_entropy['entropy'], 
                   marker='o', label=config_label, linewidth=2, color=colors[i])
        
        # Customize the plot
        dimension_names = {
            0: "Left Arm Actions", 
            1: "Right Arm Actions", 
            2: "Locomotion Actions"
        }
        dim_name = dimension_names.get(dim_idx, f"Action Dimension {dim_idx}")
        ax.set_title(f'{dim_name} - Policy Entropy Evolution\n(Higher = More Random, Lower = More Deterministic)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Training Checkpoint', fontsize=12)
        ax.set_ylabel('Policy Entropy', fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Set x-axis labels (use first config's steps as reference)
        if len(entropy_df) > 0:
            first_config = entropy_df['config'].iloc[0]
            first_config_data = entropy_df[entropy_df['config'] == first_config]
            first_config_data = first_config_data.sort_values('checkpoint_step', key=lambda x: x.map(sort_key))
            
            step_labels = []
            for step in first_config_data['checkpoint_step']:
                if step == 'final':
                    step_labels.append('Final')
                else:
                    step_millions = int(step) / 1000000
                    if step_millions >= 1:
                        step_labels.append(f'{step_millions:.1f}M')
                    else:
                        step_labels.append(f'{int(step)//1000}K')
            
            ax.set_xticks(range(len(step_labels)))
            ax.set_xticklabels(step_labels, rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(action_plots_dir, 'entropy_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Summary entropy plot created")

def create_action_probability_boxplots_by_config(plot_df, output_dir, max_configs_per_plot=4):
    """Replaced by create_action_probability_stacked_bars"""
    create_action_probability_stacked_bars(plot_df, output_dir)

def analyze_action_probability_trends(plot_df, output_dir):
    """Analyze trends in action probability evolution"""
    
    if len(plot_df) == 0:
        return
    
    analysis_results = []
    
    for config in plot_df['config'].unique():
        config_df = plot_df[plot_df['config'] == config]
        
        freq = config_df['frequency'].iloc[0]
        decay = config_df['decay'].iloc[0]
        seed = config_df['seed'].iloc[0]

        for dim_idx in sorted(config_df['action_dimension'].unique()):
            dim_df = config_df[config_df['action_dimension'] == dim_idx]
            
            for action_idx in sorted(dim_df['action_index'].unique()):
                action_df = dim_df[dim_df['action_index'] == action_idx]
                
                step_stats = action_df.groupby('checkpoint_step')['probability'].agg([
                    'mean', 'std', 'min', 'max', 'count'
                ])
                
                steps = list(step_stats.index)
                means = list(step_stats['mean'])
                
                if len(steps) > 1:
                    first_half_mean = np.mean(means[:len(means)//2])
                    second_half_mean = np.mean(means[len(means)//2:])
                    trend = "increasing" if second_half_mean > first_half_mean else "decreasing"
                    trend_magnitude = abs(second_half_mean - first_half_mean)
                else:
                    trend = "insufficient_data"
                    trend_magnitude = 0
                
                analysis_results.append({
                    'config': config,
                    'frequency': freq,
                    'decay': decay,
                    'seed': seed,
                    'action_dimension': dim_idx,
                    'action_index': action_idx,
                    'num_checkpoints': len(steps),
                    'initial_prob_mean': means[0] if means else np.nan,
                    'final_prob_mean': means[-1] if means else np.nan,
                    'trend': trend,
                    'trend_magnitude': trend_magnitude,
                    'overall_variance': np.var(means) if means else np.nan,
                    'steps_analyzed': steps
                })
    
    analysis_df = pd.DataFrame(analysis_results)
    analysis_path = os.path.join(output_dir, 'action_probability_trend_analysis.csv')
    analysis_df.to_csv(analysis_path, index=False)
    print(f"Action probability trend analysis saved to: {analysis_path}")
    
    print(f"\nACTION PROBABILITY TREND SUMMARY:")
    print(f"Configurations analyzed: {len(analysis_df['config'].unique())}")
    print(f"Total action-dimension combinations: {len(analysis_df)}")
    
    if len(analysis_df) > 0:
        print(f"Trends observed:")
        trend_counts = analysis_df['trend'].value_counts()
        for trend, count in trend_counts.items():
            print(f"  {trend}: {count}")
        
        # Most stable actions (low variance)
        stable_actions = analysis_df.nsmallest(5, 'overall_variance')
        print(f"\nMost stable actions (low variance):")
        for _, row in stable_actions.iterrows():
            print(f"  {row['config']} Dim{row['action_dimension']} Act{row['action_index']}: "
                  f"variance={row['overall_variance']:.6f}")
        
        # Most variable actions
        variable_actions = analysis_df.nlargest(5, 'overall_variance')
        print(f"\nMost variable actions:")
        for _, row in variable_actions.iterrows():
            print(f"  {row['config']} Dim{row['action_dimension']} Act{row['action_index']}: "
                  f"variance={row['overall_variance']:.6f}")

def calculate_merged_csv_distributions(merged_df, group_by_columns=['frequency', 'decay', 'step']):
    """Calculate flat distribution from the merged CSV file, grouped by specified columns"""
    # Filter out states containing 'robot'
    df = merged_df[~merged_df['state_name'].str.contains('robot', case=False, na=False)].copy()
    
    # Define object pairs to average
    object_pairs = {
        'coin': 'piggie bank',
        'gear_toy': 'gear',
        'shape_toy': 'shape sorter'
    }
    
    results = []
    
    # Group by the specified columns
    for group_values, group_df in df.groupby(group_by_columns):
        # Handle single column grouping
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        
        group_dict = dict(zip(group_by_columns, group_values))
        
        flat_dist = {}
        
        total_episodes = group_df['episode'].nunique()

        steps_per_episode = []
        for episode_num in group_df['episode'].unique():
            episode_df = group_df[group_df['episode'] == episode_num]

            max_activity_in_episode = episode_df['activity_count'].max()
            estimated_steps = min(1000, max_activity_in_episode * 2)  # Conservative estimate
            steps_per_episode.append(estimated_steps)

        avg_steps_per_episode = np.mean(steps_per_episode) if steps_per_episode else 1000
        total_opportunities = total_episodes * avg_steps_per_episode
        
        # Process each object type
        for obj_type in group_df['object_type'].unique():
            if obj_type in object_pairs:
                # For objects that need to be averaged
                obj_df = group_df[group_df['object_type'] == obj_type]
                final_obj_name = object_pairs[obj_type]
                
                for state in obj_df['state_name'].unique():
                    state_df = obj_df[obj_df['state_name'] == state]
                    total_count = state_df['activity_count'].sum()
                    if total_opportunities > 0:
                        false_percentage = max(0, min(100, 100 - (total_count / total_opportunities * 100)))
                    else:
                        false_percentage = 100
                    flat_dist[(final_obj_name, state)] = false_percentage
            else:
                # For individual objects, process each index separately
                for obj_idx in group_df[group_df['object_type'] == obj_type]['object_index'].unique():
                    obj_df = group_df[(group_df['object_type'] == obj_type) & (group_df['object_index'] == obj_idx)]
                    
                    for state in obj_df['state_name'].unique():
                        state_df = obj_df[obj_df['state_name'] == state]
                        total_count = state_df['activity_count'].sum()
                        if total_opportunities > 0:
                            false_percentage = max(0, min(100, 100 - (total_count / total_opportunities * 100)))
                        else:
                            false_percentage = 100
                        obj_name = transform_object_name(f"{obj_type}_{obj_idx}")
                        flat_dist[(obj_name, state)] = false_percentage
        
        result = group_dict.copy()
        result['flat_distribution'] = flat_dist
        result['total_episodes'] = total_episodes
        result['total_opportunities'] = total_opportunities
        results.append(result)
    
    return results

def calculate_js_divergence_robust(p, q, epsilon=1e-12):
    """Calculate Jensen-Shannon divergence with robust error handling"""
    try:
        p = np.array(p, dtype=np.float64)
        q = np.array(q, dtype=np.float64)
        
        if np.any(p < 0) or np.any(q < 0):
            return np.inf
        
        # Check for zero sums
        p_sum = np.sum(p)
        q_sum = np.sum(q)
        if p_sum <= 0 or q_sum <= 0:
            return np.inf
        
        # Normalize to ensure they sum to 1
        p = p / p_sum
        q = q / q_sum
        
        # Add epsilon to avoid log(0)
        p = p + epsilon
        q = q + epsilon
        
        # Renormalize after adding epsilon
        p = p / np.sum(p)
        q = q / np.sum(q)
        
        # Calculate the average distribution
        m = 0.5 * (p + q)
        
        # Calculate KL divergences
        kl_pm = entropy(p, m)
        kl_qm = entropy(q, m)
        
        # Check for invalid results
        if not (np.isfinite(kl_pm) and np.isfinite(kl_qm)):
            return np.inf
        
        # Calculate JS divergence
        js_div = 0.5 * (kl_pm + kl_qm)
        
        return js_div if np.isfinite(js_div) else np.inf
        
    except Exception:
        return np.inf

def calculate_kl_divergence_robust(p, q, epsilon=1e-12):
    """Calculate KL divergence from p to q with robust error handling"""
    try:
        # Convert to numpy arrays and ensure float64 precision
        p = np.array(p, dtype=np.float64)
        q = np.array(q, dtype=np.float64)
        
        # Check for invalid inputs
        if np.any(p < 0) or np.any(q < 0):
            return np.inf
        
        # Check for zero sums
        p_sum = np.sum(p)
        q_sum = np.sum(q)
        if p_sum <= 0 or q_sum <= 0:
            return np.inf
        
        # Normalize to ensure they sum to 1
        p = p / p_sum
        q = q / q_sum
        
        # Add epsilon to avoid log(0)
        p = p + epsilon
        q = q + epsilon
        
        # Renormalize after adding epsilon
        p = p / np.sum(p)
        q = q / np.sum(q)
        
        # Calculate KL divergence
        kl_div = entropy(p, q)
        
        return kl_div if np.isfinite(kl_div) else np.inf
        
    except Exception:
        return np.inf

def debug_distributions(agent_dist, csv_dist, group_info, verbose=False):
    """Debug distributions to identify problematic cases"""
    common_keys = set(agent_dist.keys()) & set(csv_dist.keys())
    
    problematic_pairs = []
    extreme_values = []
    
    for key in common_keys:
        agent_val = agent_dist[key]
        csv_val = csv_dist[key]
        
        # Check for extreme values
        if agent_val <= 0 or agent_val >= 100 or csv_val <= 0 or csv_val >= 100:
            extreme_values.append({
                'key': key,
                'agent_value': agent_val,
                'csv_value': csv_val,
                'issue': 'extreme_values'
            })
        
        # Check for perfect separation (one is 0, other is 100 or vice versa)
        if (agent_val == 0 and csv_val == 100) or (agent_val == 100 and csv_val == 0):
            problematic_pairs.append({
                'key': key,
                'agent_value': agent_val,
                'csv_value': csv_val,
                'issue': 'perfect_separation'
            })
    
    if verbose and (problematic_pairs or extreme_values):
        print(f"\nDEBUG - Group {group_info}:")
        print(f"Extreme values: {len(extreme_values)}")
        print(f"Perfect separations: {len(problematic_pairs)}")
        
        if extreme_values[:3]:  # Show first 3
            print("Sample extreme values:")
            for item in extreme_values[:3]:
                print(f"  {item['key']}: agent={item['agent_value']}, csv={item['csv_value']}")
    
    return len(problematic_pairs), len(extreme_values)

def calculate_divergences_for_all_groups(merged_distributions, agent_dist, verbose=False):
    """Calculate JS and KL divergences for all groups with comprehensive error handling"""
    all_results = []
    
    for group_data in merged_distributions:
        csv_dist = group_data['flat_distribution']
        group_info = {k: v for k, v in group_data.items() if k != 'flat_distribution'}
        
        if verbose:
            print(f"\nProcessing group: {group_info}")
        
        # Get only keys that exist in both distributions
        common_keys = set(agent_dist.keys()) & set(csv_dist.keys())
        
        if verbose:
            print(f"Number of common object-state pairs: {len(common_keys)}")
        
        # Debug distributions
        num_perfect_sep, num_extreme = debug_distributions(agent_dist, csv_dist, group_info, verbose)
        
        # Calculate divergences for each object-state pair
        js_divergences = {}
        kl_divergences = {}
        valid_js = []
        valid_kl = []
        
        for key in sorted(common_keys):
            # Create binary distributions for this state
            agent_value = np.array([agent_dist[key], 100 - agent_dist[key]])
            csv_value = np.array([csv_dist[key], 100 - csv_dist[key]])
            
            # Calculate JS divergence
            js_div = calculate_js_divergence_robust(agent_value, csv_value)
            js_divergences[key] = js_div
            if np.isfinite(js_div):
                valid_js.append(js_div)
            
            # Calculate KL divergence (from CSV to agent)
            kl_div = calculate_kl_divergence_robust(csv_value, agent_value)
            kl_divergences[key] = kl_div
            if np.isfinite(kl_div):
                valid_kl.append(kl_div)
        
        # Calculate statistics for JS divergence
        if valid_js:
            avg_js_div = np.mean(valid_js)
            min_js = np.min(valid_js)
            max_js = np.max(valid_js)
            std_js = np.std(valid_js)
            median_js = np.median(valid_js)
        else:
            avg_js_div = np.inf
            min_js = np.inf
            max_js = np.inf
            std_js = np.inf
            median_js = np.inf
        
        # Calculate statistics for KL divergence
        if valid_kl:
            avg_kl_div = np.mean(valid_kl)
            min_kl = np.min(valid_kl)
            max_kl = np.max(valid_kl)
            std_kl = np.std(valid_kl)
            median_kl = np.median(valid_kl)
        else:
            avg_kl_div = np.inf
            min_kl = np.inf
            max_kl = np.inf
            std_kl = np.inf
            median_kl = np.inf
        
        # Create comprehensive result record - UPDATED TO USE FREQUENCY/DECAY
        result = {
            'seed': group_data.get('seed'),
            'frequency': group_data.get('frequency'),
            'decay': group_data.get('decay'),
            'entropy': group_data.get('entropy'),
            'step': group_data.get('step'),
            'episode': group_data.get('episode'),
            
            # JS Divergence metrics
            'average_js_divergence': avg_js_div,
            'median_js_divergence': median_js,
            'min_js_divergence': min_js,
            'max_js_divergence': max_js,
            'std_js_divergence': std_js,
            
            # KL Divergence metrics
            'average_kl_divergence': avg_kl_div,
            'median_kl_divergence': median_kl,
            'min_kl_divergence': min_kl,
            'max_kl_divergence': max_kl,
            'std_kl_divergence': std_kl,
            
            # Counts and quality metrics
            'num_object_state_pairs': len(common_keys),
            'num_finite_js': len(valid_js),
            'num_finite_kl': len(valid_kl),
            'num_infinite_js': len(common_keys) - len(valid_js),
            'num_infinite_kl': len(common_keys) - len(valid_kl),
            'percent_finite_js': (len(valid_js) / len(common_keys) * 100) if common_keys else 0,
            'percent_finite_kl': (len(valid_kl) / len(common_keys) * 100) if common_keys else 0,
            'num_perfect_separations': num_perfect_sep,
            'num_extreme_values': num_extreme,
            
            # Data quality
            'total_episodes': group_data.get('total_episodes', 0),
            'total_opportunities': group_data.get('total_opportunities', 0),
            
            # Individual divergences (for detailed analysis)
            'individual_js_divergences': js_divergences,
            'individual_kl_divergences': kl_divergences
        }
        
        all_results.append(result)
        
        if verbose:
            print(f"JS Divergence: {avg_js_div:.6f} (finite: {len(valid_js)}/{len(common_keys)})")
            print(f"KL Divergence: {avg_kl_div:.6f} (finite: {len(valid_kl)}/{len(common_keys)})")
            if num_perfect_sep > 0 or num_extreme > 0:
                print(f"Issues: {num_perfect_sep} perfect separations, {num_extreme} extreme values")
    
    return all_results

def save_comprehensive_results(all_results, output_path):
    """Save comprehensive results with both JS and KL divergences to CSV files"""
    # Prepare summary data
    summary_data = []
    for result in all_results:
        summary_data.append({
            'seed': result['seed'],
            'frequency': result['frequency'],
            'entropy': result['entropy'],
            'decay': result['decay'],
            'step': result['step'],
            'episode': result.get('episode'),
            
            # JS Divergence metrics
            'average_js_divergence': result['average_js_divergence'],
            'median_js_divergence': result['median_js_divergence'],
            'min_js_divergence': result['min_js_divergence'],
            'max_js_divergence': result['max_js_divergence'],
            'std_js_divergence': result['std_js_divergence'],
            
            # KL Divergence metrics
            'average_kl_divergence': result['average_kl_divergence'],
            'median_kl_divergence': result['median_kl_divergence'],
            'min_kl_divergence': result['min_kl_divergence'],
            'max_kl_divergence': result['max_kl_divergence'],
            'std_kl_divergence': result['std_kl_divergence'],
            
            # Quality metrics
            'num_object_state_pairs': result['num_object_state_pairs'],
            'num_finite_js': result['num_finite_js'],
            'num_finite_kl': result['num_finite_kl'],
            'percent_finite_js': result['percent_finite_js'],
            'percent_finite_kl': result['percent_finite_kl'],
            'num_perfect_separations': result['num_perfect_separations'],
            'num_extreme_values': result['num_extreme_values'],
            'total_episodes': result['total_episodes'],
            'total_opportunities': result['total_opportunities']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_path, index=False)
    print(f"Summary results saved to: {output_path}")
    
    # Save detailed results
    detailed_output_path = output_path.replace('.csv', '_detailed.csv')
    detailed_data = []
    for result in all_results:
        for key in result['individual_js_divergences'].keys():
            js_div = result['individual_js_divergences'][key]
            kl_div = result['individual_kl_divergences'][key]
            obj, state = key
            detailed_data.append({
                'seed': result['seed'],
                'frequency': result['frequency'],
                'decay': result['decay'],
                'step': result['step'],
                'episode': result.get('episode'),
                'object': obj,
                'state': state,
                'js_divergence': js_div,
                'kl_divergence': kl_div,
                'js_finite': np.isfinite(js_div),
                'kl_finite': np.isfinite(kl_div)
            })
    
    detailed_df = pd.DataFrame(detailed_data)
    detailed_df.to_csv(detailed_output_path, index=False)
    print(f"Detailed results saved to: {detailed_output_path}")
    
    # Save quality report
    quality_output_path = output_path.replace('.csv', '_quality_report.csv')
    quality_data = []
    for result in all_results:
        quality_data.append({
            'seed': result['seed'],
            'frequency': result['frequency'],
            'entropy': result['entropy'],
            'decay': result['decay'],
            'step': result['step'],
            'data_quality_score': result['percent_finite_js'] + result['percent_finite_kl'],
            'has_issues': result['num_perfect_separations'] > 0 or result['num_extreme_values'] > 0,
            'total_issues': result['num_perfect_separations'] + result['num_extreme_values'],
            'episodes_analyzed': result['total_episodes']
        })
    
    quality_df = pd.DataFrame(quality_data)
    quality_df.to_csv(quality_output_path, index=False)
    print(f"Quality report saved to: {quality_output_path}")
    
    return summary_df, detailed_df, quality_df

def analyze_results_comprehensive(summary_df, detailed_df, quality_df):
    """Provide comprehensive analysis of the results"""
    print("\n" + "="*80)
    print("COMPREHENSIVE RESULTS ANALYSIS")
    print("="*80)
    
    # Basic statistics
    total_configs = len(summary_df)
    print(f"Total configurations analyzed: {total_configs}")
    
    # Data quality analysis
    finite_js_configs = summary_df[summary_df['percent_finite_js'] == 100]
    finite_kl_configs = summary_df[summary_df['percent_finite_kl'] == 100]
    both_finite = summary_df[(summary_df['percent_finite_js'] == 100) & 
                            (summary_df['percent_finite_kl'] == 100)]
    
    print(f"\nDATA QUALITY:")
    print(f"Configurations with 100% finite JS divergences: {len(finite_js_configs)}/{total_configs}")
    print(f"Configurations with 100% finite KL divergences: {len(finite_kl_configs)}/{total_configs}")
    print(f"Configurations with both metrics 100% finite: {len(both_finite)}/{total_configs}")
    
    # JS Divergence Analysis
    valid_js = summary_df[np.isfinite(summary_df['average_js_divergence'])]
    if len(valid_js) > 0:
        print(f"\nJS DIVERGENCE ANALYSIS:")
        print(f"Valid results: {len(valid_js)}/{total_configs}")
        print(f"Mean: {valid_js['average_js_divergence'].mean():.6f}")
        print(f"Median: {valid_js['median_js_divergence'].mean():.6f}")
        print(f"Best (min): {valid_js['average_js_divergence'].min():.6f}")
        print(f"Worst (max): {valid_js['average_js_divergence'].max():.6f}")
        print(f"Std: {valid_js['average_js_divergence'].std():.6f}")
        
        # Best JS configurations
        print(f"\nTOP 5 JS DIVERGENCE CONFIGURATIONS:")
        best_js = valid_js.nsmallest(5, 'average_js_divergence')
        for idx, row in best_js.iterrows():
            config_info = []
            if 'frequency' in row.index:
                config_info.append(f"Freq: {row['frequency']}")
            if 'decay' in row.index:
                config_info.append(f"Decay: {row['decay']}")
            #if 'entropy' in row.index:
            #    config_info.append(f"Entropy: {row['entropy']:.3f}")
            if 'step' in row.index:
                config_info.append(f"Step: {row['step']}")
            config_str = ", ".join(config_info)
            print(f"  {config_str}: JS = {row['average_js_divergence']:.6f}")

    # KL Divergence Analysis
    valid_kl = summary_df[np.isfinite(summary_df['average_kl_divergence'])]
    if len(valid_kl) > 0:
        print(f"\nKL DIVERGENCE ANALYSIS:")
        print(f"Valid results: {len(valid_kl)}/{total_configs}")
        print(f"Mean: {valid_kl['average_kl_divergence'].mean():.6f}")
        print(f"Median: {valid_kl['median_kl_divergence'].mean():.6f}")
        print(f"Best (min): {valid_kl['average_kl_divergence'].min():.6f}")
        print(f"Worst (max): {valid_kl['average_kl_divergence'].max():.6f}")
        print(f"Std: {valid_kl['average_kl_divergence'].std():.6f}")
        
        # Best KL configurations
        print(f"\nTOP 5 KL DIVERGENCE CONFIGURATIONS:")
        best_kl = valid_kl.nsmallest(5, 'average_kl_divergence')
        for idx, row in best_kl.iterrows():
            config_info = []
            if 'frequency' in row.index:
                config_info.append(f"Freq: {row['frequency']}")
            if 'decay' in row.index:
                config_info.append(f"Decay: {row['decay']}")
            #if 'entropy' in row.index:
            #    config_info.append(f"Entropy: {row['entropy']:.3f}")
            if 'step' in row.index:
                config_info.append(f"Step: {row['step']}")
            config_str = ", ".join(config_info)
            print(f"  {config_str}: KL = {row['average_kl_divergence']:.6f}")

    # Frequency analysis
    if 'frequency' in summary_df.columns and len(valid_js) > 0:
        print(f"\nFREQUENCY LEVEL ANALYSIS:")
        freq_js_stats = valid_js.groupby('frequency')['average_js_divergence'].agg(['mean', 'std', 'count', 'min'])
        freq_kl_stats = valid_kl.groupby('frequency')['average_kl_divergence'].agg(['mean', 'std', 'count', 'min'])
        
        print("JS Divergence by frequency:")
        for freq in sorted(freq_js_stats.index):
            stats = freq_js_stats.loc[freq]
            print(f"  Frequency {freq}: mean={stats['mean']:.6f}, "
                  f"std={stats['std']:.6f}, count={stats['count']}, min={stats['min']:.6f}")
        
        print("KL Divergence by frequency:")
        for freq in sorted(freq_kl_stats.index):
            stats = freq_kl_stats.loc[freq]
            print(f"  Frequency {freq}: mean={stats['mean']:.6f}, "
                  f"std={stats['std']:.6f}, count={stats['count']}, min={stats['min']:.6f}")

    # Decay analysis
    if 'decay' in summary_df.columns and len(valid_js) > 0:
        print(f"\nDECAY LEVEL ANALYSIS:")
        decay_js_stats = valid_js.groupby('decay')['average_js_divergence'].agg(['mean', 'std', 'count', 'min'])
        decay_kl_stats = valid_kl.groupby('decay')['average_kl_divergence'].agg(['mean', 'std', 'count', 'min'])
        
        print("JS Divergence by decay:")
        for decay in sorted(decay_js_stats.index):
            stats = decay_js_stats.loc[decay]
            print(f"  Decay {decay}: mean={stats['mean']:.6f}, "
                  f"std={stats['std']:.6f}, count={stats['count']}, min={stats['min']:.6f}")
        
        print("KL Divergence by decay:")
        for decay in sorted(decay_kl_stats.index):
            stats = decay_kl_stats.loc[decay]
            print(f"  Decay {decay}: mean={stats['mean']:.6f}, "
                  f"std={stats['std']:.6f}, count={stats['count']}, min={stats['min']:.6f}")
    
    # Entropy analysis
    if 'entropy' in summary_df.columns and len(valid_js) > 0:
        print(f"\nENTROPY LEVEL ANALYSIS:")
        entropy_js_stats = valid_js.groupby('entropy')['average_js_divergence'].agg(['mean', 'std', 'count', 'min'])
        entropy_kl_stats = valid_kl.groupby('entropy')['average_kl_divergence'].agg(['mean', 'std', 'count', 'min'])
        
        print("JS Divergence by entropy:")
        for entropy in sorted(entropy_js_stats.index):
            stats = entropy_js_stats.loc[entropy]
            print(f"  Entropy {entropy:.3f}: mean={stats['mean']:.6f}, "
                  f"std={stats['std']:.6f}, count={stats['count']}, min={stats['min']:.6f}")
        
        print("KL Divergence by entropy:")
        for entropy in sorted(entropy_kl_stats.index):
            stats = entropy_kl_stats.loc[entropy]
            print(f"  Entropy {entropy:.3f}: mean={stats['mean']:.6f}, "
                  f"std={stats['std']:.6f}, count={stats['count']}, min={stats['min']:.6f}")
    
    # Problematic configurations
    problematic = summary_df[
        (summary_df['num_perfect_separations'] > 0) | 
        (summary_df['num_extreme_values'] > 0) |
        (summary_df['percent_finite_js'] < 90) |
        (summary_df['percent_finite_kl'] < 90)
    ]
    
    if len(problematic) > 0:
        print(f"\nPROBLEMATIC CONFIGURATIONS ({len(problematic)}):")
        print("Steps with issues:", sorted(problematic['step'].unique()))
        
        if 'frequency' in problematic.columns:
            print("Frequencies with issues:", sorted(problematic['frequency'].unique()))
        if 'decay' in problematic.columns:
            print("Decay values with issues:", sorted(problematic['decay'].unique()))
        if 'entropy' in problematic.columns:
            print("Entropies with issues:", sorted(problematic['entropy'].unique()))
        
        problematic_with_score = problematic.copy()
        problematic_with_score['quality_score'] = (
            problematic_with_score['percent_finite_js'] + 
            problematic_with_score['percent_finite_kl']
        )
        worst_quality = problematic_with_score.nsmallest(3, 'quality_score')
        print("Worst quality configurations:")
        for idx, row in worst_quality.iterrows():
            config_info = []
            if 'frequency' in row.index:
                config_info.append(f"Freq: {row['frequency']}")
            if 'decay' in row.index:
                config_info.append(f"Decay: {row['decay']}")
            #if 'entropy' in row.index:
            #    config_info.append(f"Entropy: {row['entropy']:.3f}")
            if 'step' in row.index:
                config_info.append(f"Step: {row['step']}")
            config_str = ", ".join(config_info)
            
            print(f"  {config_str}: "
                  f"JS finite: {row['percent_finite_js']:.0f}%, "
                  f"KL finite: {row['percent_finite_kl']:.0f}%, "
                  f"Issues: {row['num_perfect_separations'] + row['num_extreme_values']}")
    
    # Recommendations
    print(f"\nRECOMMENDATIONS:")
    if len(both_finite) > 0:
        best_overall = both_finite.loc[both_finite['average_js_divergence'].idxmin()]
        
        config_info = []
        if 'frequency' in best_overall.index:
            config_info.append(f"Freq: {best_overall['frequency']}")
        if 'decay' in best_overall.index:
            config_info.append(f"Decay: {best_overall['decay']}")
        #if 'entropy' in best_overall.index:
        #    config_info.append(f"Entropy: {best_overall['entropy']:.3f}")
        if 'step' in best_overall.index:
            config_info.append(f"Step: {best_overall['step']}")
        config_str = ", ".join(config_info)
        
        print(f"Best overall configuration: {config_str}")
        print(f"  JS divergence: {best_overall['average_js_divergence']:.6f}")
        print(f"  KL divergence: {best_overall['average_kl_divergence']:.6f}")
    
    if len(problematic) > 0:
        print(f"Consider excluding {len(problematic)} problematic configurations from final analysis")
    
    # Parameter-specific recommendations
    available_params = [col for col in ['frequency', 'decay', 'entropy'] if col in summary_df.columns]
    if available_params and len(valid_js) > 0:
        print(f"Focus on parameter levels with consistently good performance across multiple steps")
        
        for param in available_params:
            try:
                param_analysis = valid_js.groupby(param)['average_js_divergence'].agg(['mean', 'std', 'count']).sort_values('mean')
                
                if len(param_analysis) > 0:
                    best_param_value = param_analysis.index[0]
                    best_param_stats = param_analysis.iloc[0]
                    
                    print(f"  Best {param}: {best_param_value} "
                          f"(mean JS: {best_param_stats['mean']:.6f}, "
                          f"std: {best_param_stats['std']:.6f}, "
                          f"count: {best_param_stats['count']})")
                else:
                    print(f"  No valid data for {param} analysis")
            except Exception as e:
                print(f"  Error analyzing {param}: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive merge and analysis of test results with JS and KL divergences')
    parser.add_argument('results_dir', help='Directory containing test results (e.g., results/)')
    parser.add_argument('--pkl-path', default='post_processing/averaged_state_distributions.pkl', 
                       help='Path to the agent distribution pickle file')
    parser.add_argument('--output-dir', default='comprehensive_analysis_results_6', 
                       help='Directory to save analysis results')
    parser.add_argument('--group-by', nargs='+', default=['frequency', 'decay', 'step'],
                   choices=['seed', 'frequency', 'decay', 'entropy', 'step', 'episode'],
                   help='Columns to group by for analysis')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Print detailed debugging information')
    parser.add_argument('--fix-infinity', action='store_true', default=True,
                       help='Apply robust infinity handling (default: True)')
    parser.add_argument('--min-episodes', type=int, default=3,
                       help='Minimum number of episodes required for analysis (default: 3)')
    
    args = parser.parse_args()
    
    #validate inputs
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory '{args.results_dir}' not found")
        return
    
    if not os.path.exists(args.pkl_path):
        print(f"Error: Pickle file '{args.pkl_path}' not found")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("COMPREHENSIVE MERGE AND ANALYSIS OF TEST RESULTS")
    print("="*80)
    print(f"Results directory: {args.results_dir}")
    print(f"Agent distributions: {args.pkl_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Grouping by: {args.group_by}")
    print(f"Verbose mode: {args.verbose}")
    print(f"Infinity handling: {args.fix_infinity}")
    
    print(f"\nStep 1: Merging activity CSV files...")
    merged_df = merge_all_activity_csvs(args.results_dir)
    if merged_df is None:
        print("No data to analyze. Exiting.")
        return
    
    merged_csv_path = os.path.join(args.output_dir, 'merged_activity_data.csv')
    merged_df.to_csv(merged_csv_path, index=False)
    print(f"Merged data saved to: {merged_csv_path}")
    
    print(f"\nStep 2: Loading agent distributions...")
    agent_dist = load_agent_distributions(args.pkl_path)
    print(f"Loaded {len(agent_dist)} object-state pairs from agent data")
    
    print(f"\nStep 3: Calculating distributions by {args.group_by}...")
    merged_distributions = calculate_merged_csv_distributions(merged_df, args.group_by)
    print(f"Created {len(merged_distributions)} group distributions")
    
    # Filter out groups with insufficient data
    valid_distributions = []
    for dist in merged_distributions:
        if dist.get('total_episodes', 0) >= args.min_episodes:
            valid_distributions.append(dist)
        elif args.verbose:
            print(f"Skipping group {dist} - insufficient episodes ({dist.get('total_episodes', 0)} < {args.min_episodes})")
    
    print(f"Using {len(valid_distributions)} distributions with >= {args.min_episodes} episodes")
    
    print(f"\nStep 4: Calculating JS and KL divergences...")
    all_results = calculate_divergences_for_all_groups(
        valid_distributions, agent_dist, args.verbose
    )
    
    print(f"\nStep 5: Saving results...")
    output_path = os.path.join(args.output_dir, 'comprehensive_divergence_results.csv')
    summary_df, detailed_df, quality_df = save_comprehensive_results(all_results, output_path)
    
    print(f"\nStep 6: Analyzing results...")
    analyze_results_comprehensive(summary_df, detailed_df, quality_df)
    
    analysis_summary_path = os.path.join(args.output_dir, 'analysis_summary.txt')
    with open(analysis_summary_path, 'w') as f:
        f.write("COMPREHENSIVE ANALYSIS SUMMARY\n")
        f.write("="*50 + "\n\n")
        
        # Write key findings
        valid_js = summary_df[np.isfinite(summary_df['average_js_divergence'])]
        valid_kl = summary_df[np.isfinite(summary_df['average_kl_divergence'])]
        
        f.write(f"Total configurations: {len(summary_df)}\n")
        f.write(f"Valid JS results: {len(valid_js)}\n")
        f.write(f"Valid KL results: {len(valid_kl)}\n\n")
        
        if len(valid_js) > 0:
            best_js = valid_js.loc[valid_js['average_js_divergence'].idxmin()]
            f.write(f"Best JS configuration:\n")
            f.write(f"  Entropy: {best_js['entropy']}\n")
            f.write(f"  Step: {best_js['step']}\n")
            f.write(f"  JS Divergence: {best_js['average_js_divergence']:.6f}\n\n")
        
        if len(valid_kl) > 0:
            best_kl = valid_kl.loc[valid_kl['average_kl_divergence'].idxmin()]
            f.write(f"Best KL configuration:\n")
            f.write(f"  Entropy: {best_kl['entropy']}\n")
            f.write(f"  Step: {best_kl['step']}\n")
            f.write(f"  KL Divergence: {best_kl['average_kl_divergence']:.6f}\n\n")
        
        # Entropy analysis
        if 'entropy' in summary_df.columns and len(valid_js) > 0:
            entropy_analysis = valid_js.groupby('entropy')['average_js_divergence'].mean().sort_values()
            f.write("Entropy ranking (by average JS divergence):\n")
            for entropy, avg_js in entropy_analysis.items():
                f.write(f"  {entropy}: {avg_js:.6f}\n")
    
    print(f"Analysis summary saved to: {analysis_summary_path}")
    
    # Final recommendations
    print(f"\n" + "="*80)
    print("FINAL RECOMMENDATIONS")
    print("="*80)
    
    if len(valid_js) > 0:
        best_js = valid_js.loc[valid_js['average_js_divergence'].idxmin()]
        print(f"✓ Best overall configuration: Entropy {best_js['entropy']}, Step {best_js['step']}")
        print(f"  JS divergence: {best_js['average_js_divergence']:.6f}")
        if 'average_kl_divergence' in best_js and np.isfinite(best_js['average_kl_divergence']):
            print(f"  KL divergence: {best_js['average_kl_divergence']:.6f}")
    
    # Check for consistent performers
    if 'entropy' in summary_df.columns:
        entropy_consistency = valid_js.groupby('entropy').agg({
            'average_js_divergence': ['mean', 'std', 'count']
        }).round(6)
        
        consistent_entropies = []
        for entropy in entropy_consistency.index:
            mean_js = entropy_consistency.loc[entropy, ('average_js_divergence', 'mean')]
            std_js = entropy_consistency.loc[entropy, ('average_js_divergence', 'std')]
            count = entropy_consistency.loc[entropy, ('average_js_divergence', 'count')]
            
            if count >= 2 and std_js < 0.01:  # Low variance across steps
                consistent_entropies.append((entropy, mean_js, std_js))
        
        if consistent_entropies:
            consistent_entropies.sort(key=lambda x: x[1])  # Sort by mean JS
            print(f"\nMost consistent entropy levels (low variance across steps):")
            for entropy, mean_js, std_js in consistent_entropies[:3]:
                print(f"  Entropy {entropy}: mean JS = {mean_js:.6f}, std = {std_js:.6f}")
    
    print(f"\nAll results saved to: {args.output_dir}")

    print(f"\nStep 7: Analyzing action probabilities...")
    
    # Find action probability files
    action_files = find_action_probability_files(args.results_dir)
    
    if action_files:
        print(f"Found {len(action_files)} action probability files")
        
        # Load action probability data
        all_action_data = load_action_probability_data(action_files, args.verbose)
        
        if all_action_data:
            print(f"Loaded action data for {len(all_action_data)} configurations")
            
            # Prepare data for plotting
            plot_df = prepare_action_plot_data(all_action_data)
            
            if len(plot_df) > 0:
                print(f"Prepared {len(plot_df)} action probability data points for plotting")
                
                # Create box plots by configuration
                create_action_probability_boxplots_by_config(plot_df, args.output_dir)
                
                # Create summary plots
                create_summary_action_plots(plot_df, args.output_dir)
                
                # Analyze trends
                analyze_action_probability_trends(plot_df, args.output_dir)
                
                print(f"Action probability analysis complete!")
                print(f"Plots saved to: {os.path.join(args.output_dir, 'action_probability_plots')}")
            else:
                print("No action probability data could be processed for plotting")
        else:
            print("No valid action probability data found")
    else:
        print("No action probability files found - skipping action analysis")
        print("Expected location: {results_dir}/**/action_probabilities/action_probs_step_*.pkl")

if __name__ == "__main__":
    main()