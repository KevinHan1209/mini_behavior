import pandas as pd
import numpy as np
import pickle
from scipy.stats import entropy
import os
import re
import argparse

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
    obj_name = re.sub(r' \d+$', '', obj_name)
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

def calculate_csv_distributions_new_format(csv_path):
    """Calculate flat distribution from the new CSV format"""
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Initialize dictionary to store aggregated counts
    aggregated_counts = {}
    
    # Define object pairs to average
    object_pairs = {
        'coin': 'piggie bank',
        'gear_toy': 'gear',
        'shape_toy': 'shape sorter'
    }
    
    # Objects that should keep individual indices (like cubes)
    keep_individual_indices = {'cube'}
    
    # Process each column (except checkpoint_id)
    for column in df.columns:
        if column == 'checkpoint_id':
            continue
            
        # Parse column name to extract object_type, object_index, and state_name
        # Format: object_type_index_statename (e.g., shape_toy_0_inlefthandofrobot)
        parts = column.split('_')
        
        # Find where the state name starts (it's after the numeric index)
        state_start_idx = None
        for i, part in enumerate(parts):
            if part.isdigit():
                state_start_idx = i + 1
                break
        
        if state_start_idx is None:
            # Handle cases without numeric index (like some container objects)
            # Try to identify known state names
            known_states = ['inlefthandofrobot', 'inrighthandofrobot', 'thrown', 'gothit', 
                           'hitter', 'mouthed', 'contains', 'attached', 'noise', 'open',
                           'kicked', 'toggled', 'popup', 'pullshed', 'climbed', 'usebrush']
            for i in range(len(parts)):
                potential_state = '_'.join(parts[i:])
                if potential_state in known_states:
                    state_start_idx = i
                    break
        
        if state_start_idx is None:
            print(f"Warning: Could not parse column {column}")
            continue
        
        # Extract components
        if state_start_idx > 2:
            # Multi-word object type (e.g., tree_busy_box_0_toggled)
            object_type = '_'.join(parts[:state_start_idx-1])
            object_index = int(parts[state_start_idx-1])
        else:
            # Single word object type (e.g., gear_0_thrown)
            object_type = parts[0]
            object_index = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        
        state_name = '_'.join(parts[state_start_idx:])
        
        # Skip states containing 'robot' (except for hand states which we want to keep)
        if 'robot' in state_name and 'handofrobot' not in state_name:
            continue
        
        # Get the total count across all checkpoints
        total_count = df[column].sum()
        
        # Determine the object name for aggregation
        if object_type in object_pairs:
            # For objects that need to be paired
            obj_name = object_pairs[object_type]
        elif object_type in keep_individual_indices:
            # For objects that keep individual indices
            obj_name = transform_object_name(f"{object_type}_{object_index}")
        else:
            # For most objects, remove the index
            obj_name = transform_object_name(object_type)
            
            # Map object names to match agent distribution and aggregate related objects
            name_mappings = {
                'bucket toy': 'bucket',
                'cart toy': 'cart',
                'cube cabinet': 'winnie cabinet',
                'mini broom': 'broom',  # mini_broom aggregates with broom
                'ring toy': 'yellow donut ring',
                'winnie': 'winnie cabinet'  # winnie aggregates with winnie cabinet
            }
            
            if obj_name in name_mappings:
                obj_name = name_mappings[obj_name]
        
        # Aggregate counts
        key = (obj_name, state_name)
        if key not in aggregated_counts:
            aggregated_counts[key] = {'count': 0, 'instances': 0}
        aggregated_counts[key]['count'] += total_count
        aggregated_counts[key]['instances'] += 1
    
    # Calculate flat distribution from aggregated counts
    flat_dist = {}
    
    # Calculate episodes per checkpoint (200 steps per episode, 10 episodes)
    episodes_per_checkpoint = 10
    total_episodes = len(df) * episodes_per_checkpoint
    
    for key, data in aggregated_counts.items():
        # For averaged objects, we need to consider the number of instances
        # Maximum possible count (200 steps * total episodes * number of instances)
        max_possible_count = 200 * total_episodes * data['instances']
        
        # Calculate false percentage
        if data['count'] > 0:
            false_percentage = 100 - (data['count'] / max_possible_count * 100)
        else:
            false_percentage = 100
        
        flat_dist[key] = false_percentage
    
    return flat_dist

def calculate_js_divergence(p, q):
    """Calculate Jensen-Shannon divergence between two distributions"""
    # Convert to numpy arrays
    p = np.array(p)
    q = np.array(q)
    
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    p = p + epsilon
    q = q + epsilon
    
    # Calculate the average distribution
    m = 0.5 * (p + q)
    
    # Calculate JS divergence
    js_div = 0.5 * (entropy(p, m) + entropy(q, m))
    
    return js_div

def calculate_js_divergence_pipeline(csv_path, pkl_path="post_processing/averaged_state_distributions.pkl", verbose=False):
    """
    Main pipeline function to calculate JS divergence between CSV data and agent distribution
    
    Args:
        csv_path (str): Path to the CSV file containing algorithm run data (new format)
        pkl_path (str): Path to the agent distribution pickle file (default: post_processing/averaged_state_distributions.pkl)
        verbose (bool): Whether to print detailed debugging information
    
    Returns:
        dict: Dictionary containing JS divergence results
    """
    # Load distributions
    agent_dist = load_agent_distributions(pkl_path)
    csv_dist = calculate_csv_distributions_new_format(csv_path)
    
    if verbose:
        # Get unique object names from both distributions
        agent_objects = sorted(set(key[0] for key in agent_dist.keys()))
        csv_objects = sorted(set(key[0] for key in csv_dist.keys()))
        
        print(f"\nAgent distribution objects ({len(agent_objects)}):")
        for obj in agent_objects:
            print(f"  - {obj}")
        
        print(f"\nCSV distribution objects ({len(csv_objects)}):")
        for obj in csv_objects:
            print(f"  - {obj}")
    
    # Get different key sets
    common_keys = set(agent_dist.keys()) & set(csv_dist.keys())
    agent_only = set(agent_dist.keys()) - set(csv_dist.keys())
    csv_only = set(csv_dist.keys()) - set(agent_dist.keys())
    
    # For calculation, use all CSV keys (common + csv_only)
    # Ignore agent-only keys
    keys_to_calculate = common_keys | csv_only
    
    if verbose:
        print(f"\nNumber of common object-state pairs: {len(common_keys)}")
        print(f"Number of CSV-only pairs (will use 100% false for agent): {len(csv_only)}")
        print(f"Number of agent-only pairs (will be ignored): {len(agent_only)}")
        print(f"Total pairs for calculation: {len(keys_to_calculate)}")
        
        if agent_only:
            print(f"\nKeys only in agent distribution (IGNORED) ({len(agent_only)}):")
            for key in sorted(agent_only)[:10]:
                print(f"  - {key}")
            if len(agent_only) > 10:
                print(f"  ... and {len(agent_only) - 10} more")
        
        if csv_only:
            print(f"\nKeys only in CSV distribution (agent set to 100% false) ({len(csv_only)}):")
            for key in sorted(csv_only)[:10]:
                print(f"  - {key}")
            if len(csv_only) > 10:
                print(f"  ... and {len(csv_only) - 10} more")
    
    # Calculate JS divergence for each object-state pair
    js_divergences = {}
    for key in sorted(keys_to_calculate):
        # Get agent value (100 if not in agent dist, meaning 100% false)
        agent_false_pct = agent_dist.get(key, 100)
        
        # Create binary distributions for this state
        agent_value = np.array([agent_false_pct, 100 - agent_false_pct])
        csv_value = np.array([csv_dist[key], 100 - csv_dist[key]])
        
        # Calculate JS divergence for this state
        js_div = calculate_js_divergence(agent_value, csv_value)
        js_divergences[key] = js_div
    
    # Calculate average JS divergence
    avg_js_div = np.mean(list(js_divergences.values()))
    
    # Calculate statistics
    js_values = list(js_divergences.values())
    min_js = np.min(js_values)
    max_js = np.max(js_values)
    std_js = np.std(js_values)
    
    results = {
        'average_js_divergence': avg_js_div,
        'min_js_divergence': min_js,
        'max_js_divergence': max_js,
        'std_js_divergence': std_js,
        'num_object_state_pairs': len(js_divergences),
        'individual_divergences': js_divergences
    }
    
    if verbose:
        print("\nJS Divergence Statistics:")
        print(f"  Average: {avg_js_div:.4f}")
        print(f"  Min: {min_js:.4f}")
        print(f"  Max: {max_js:.4f}")
        print(f"  Std: {std_js:.4f}")
        print(f"  Number of pairs: {len(js_divergences)}")
        
        print("\nTop 10 highest divergences:")
        sorted_divergences = sorted(js_divergences.items(), key=lambda x: x[1], reverse=True)
        for i, (key, js_div) in enumerate(sorted_divergences[:10]):
            print(f"  {i+1}. {key}: {js_div:.4f}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Calculate JS divergence between algorithm run and agent distribution (new CSV format)')
    parser.add_argument('csv_path', help='Path to the CSV file containing algorithm run data')
    parser.add_argument('--pkl-path', default='post_processing/averaged_state_distributions.pkl', 
                       help='Path to the agent distribution pickle file')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Print detailed debugging information')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.csv_path):
        print(f"Error: CSV file '{args.csv_path}' not found")
        return
    
    if not os.path.exists(args.pkl_path):
        print(f"Error: Pickle file '{args.pkl_path}' not found")
        return
    
    # Run the pipeline
    results = calculate_js_divergence_pipeline(args.csv_path, args.pkl_path, args.verbose)
    
    # Print summary
    print(f"\nSummary for {args.csv_path}:")
    print(f"Average JS Divergence: {results['average_js_divergence']:.4f}")

if __name__ == "__main__":
    main()

"""
Example command to get JS divergence for new format:
python mini_behavior/post_processing/calculate_js_pipeline_new.py test/activity_logs/checkpoint_1000000_activity.csv --verbose
"""