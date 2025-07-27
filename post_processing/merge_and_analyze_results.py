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
            # Pattern like: rnd_8x8_seed3_ent0.001_step12345
            parts = part.split('_')
            for i, p in enumerate(parts):
                if p.startswith('seed'):
                    metadata['seed'] = int(p[4:])
                elif p.startswith('ent'):
                    metadata['entropy'] = float(p[3:])
                elif p.startswith('step'):
                    metadata['step'] = p[4:]  # Keep as string in case it's "final"
            break
    
    # Extract episode from filename
    filename = Path(csv_path).name
    if filename.startswith('episode_'):
        episode_num = filename.split('_')[1]
        metadata['episode'] = int(episode_num)
    
    return metadata

def merge_all_activity_csvs(results_dir):
    """Merge all activity CSV files from the results directory"""
    pattern = os.path.join(results_dir, "**/csvs/episode_*_activity.csv")
    csv_files = glob.glob(pattern, recursive=True)
    
    if not csv_files:
        print(f"No activity CSV files found in {results_dir}")
        return None
    
    print(f"Found {len(csv_files)} activity CSV files")
    
    merged_data = []
    skipped_files = []
    
    for csv_path in csv_files:
        metadata = extract_metadata_from_path(csv_path)
        
        if not all(key in metadata for key in ['seed', 'entropy', 'step', 'episode']):
            print(f"Warning: Could not extract complete metadata from {csv_path}")
            skipped_files.append(csv_path)
            continue
        
        try:
            df = pd.read_csv(csv_path)
            
            # Add metadata columns
            df['seed'] = metadata['seed']
            df['entropy'] = metadata['entropy']
            df['step'] = metadata['step']
            df['episode'] = metadata['episode']
            
            merged_data.append(df)
            
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")
            skipped_files.append(csv_path)
            continue
    
    if not merged_data:
        print("No valid CSV files could be processed")
        return None
    
    merged_df = pd.concat(merged_data, ignore_index=True)
    
    print(f"Merged data shape: {merged_df.shape}")
    print(f"Successfully processed: {len(merged_data)} files")
    print(f"Skipped files: {len(skipped_files)}")
    print(f"Seeds: {sorted(merged_df['seed'].unique())}")
    print(f"Entropies: {sorted(merged_df['entropy'].unique())}")
    print(f"Steps: {sorted(merged_df['step'].unique())}")
    print(f"Episodes: {sorted(merged_df['episode'].unique())}")
    
    return merged_df

def calculate_merged_csv_distributions(merged_df, group_by_columns=['entropy', 'step']):
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
        # Use dynamic step count based on convergence testing (up to 1000)
        total_opportunities = total_episodes * 1000
        
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
        
        # Create comprehensive result record
        result = {
            'seed': group_data.get('seed'),
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
            'entropy': result['entropy'],
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
                'entropy': result['entropy'],
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
            'entropy': result['entropy'],
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
            print(f"  Entropy {row['entropy']:.3f}, Step {row['step']}: "
                  f"JS={row['average_js_divergence']:.6f} "
                  f"({row['percent_finite_js']:.0f}% finite)")
    
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
            print(f"  Entropy {row['entropy']:.3f}, Step {row['step']}: "
                  f"KL={row['average_kl_divergence']:.6f} "
                  f"({row['percent_finite_kl']:.0f}% finite)")
    
    # Entropy analysis
    if 'entropy' in summary_df.columns:
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
    
    problematic = summary_df[
        (summary_df['num_perfect_separations'] > 0) | 
        (summary_df['num_extreme_values'] > 0) |
        (summary_df['percent_finite_js'] < 90) |
        (summary_df['percent_finite_kl'] < 90)
    ]
    
    if len(problematic) > 0:
        print(f"\nPROBLEMATIC CONFIGURATIONS ({len(problematic)}):")
        print("Steps with issues:", sorted(problematic['step'].unique()))
        print("Entropies with issues:", sorted(problematic['entropy'].unique()))
        
        problematic_with_score = problematic.copy()
        problematic_with_score['quality_score'] = (
            problematic_with_score['percent_finite_js'] + 
            problematic_with_score['percent_finite_kl']
        )
        worst_quality = problematic_with_score.nsmallest(3, 'quality_score')
        print("Worst quality configurations:")
        for idx, row in worst_quality.iterrows():
            print(f"  Entropy {row['entropy']:.3f}, Step {row['step']}: "
                  f"JS finite: {row['percent_finite_js']:.0f}%, "
                  f"KL finite: {row['percent_finite_kl']:.0f}%, "
                  f"Issues: {row['num_perfect_separations'] + row['num_extreme_values']}")
    
    #recs
    print(f"\nRECOMMENDATIONS:")
    if len(both_finite) > 0:
        best_overall = both_finite.loc[both_finite['average_js_divergence'].idxmin()]
        print(f"Best overall configuration: Entropy {best_overall['entropy']:.3f}, Step {best_overall['step']}")
        print(f"  JS divergence: {best_overall['average_js_divergence']:.6f}")
        print(f"  KL divergence: {best_overall['average_kl_divergence']:.6f}")
    
    if len(problematic) > 0:
        print(f"Consider excluding {len(problematic)} problematic configurations from final analysis")
    
    print(f"Focus on entropy levels with consistently good performance across multiple steps")

def main():
    parser = argparse.ArgumentParser(description='Comprehensive merge and analysis of test results with JS and KL divergences')
    parser.add_argument('results_dir', help='Directory containing test results (e.g., results/)')
    parser.add_argument('--pkl-path', default='post_processing/averaged_state_distributions.pkl', 
                       help='Path to the agent distribution pickle file')
    parser.add_argument('--output-dir', default='comprehensive_analysis_results', 
                       help='Directory to save analysis results')
    parser.add_argument('--group-by', nargs='+', default=['entropy', 'step'],
                       choices=['seed', 'entropy', 'step', 'episode'],
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

if __name__ == "__main__":
    main()