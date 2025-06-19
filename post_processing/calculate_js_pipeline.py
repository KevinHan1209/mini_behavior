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

def calculate_csv_distributions(csv_path):
    """Calculate flat distribution from the CSV file"""
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Filter out states containing 'robot'
    df = df[~df['state_name'].str.contains('robot', case=False)]
    
    # Initialize dictionary to store flat distribution
    flat_dist = {}
    
    # Define object pairs to average
    object_pairs = {
        'coin': 'piggie bank',
        'gear_toy': 'gear',
        'shape_toy': 'shape sorter'
    }
    
    # Process each object type
    for obj_type in df['object_type'].unique():
        if obj_type in object_pairs:
            # For objects that need to be averaged
            obj_df = df[df['object_type'] == obj_type]
            final_obj_name = object_pairs[obj_type]
            
            for state in obj_df['state_name'].unique():
                state_df = obj_df[obj_df['state_name'] == state]
                total_count = state_df['activity_count'].sum()
                if total_count > 0:
                    false_percentage = 100 - (total_count / 5000 * 100)
                else:
                    false_percentage = 100  # If no activity, it's completely false
                flat_dist[(final_obj_name, state)] = false_percentage
        else:
            # For individual objects, process each index separately
            for obj_idx in df[df['object_type'] == obj_type]['object_index'].unique():
                obj_df = df[(df['object_type'] == obj_type) & (df['object_index'] == obj_idx)]
                
                for state in obj_df['state_name'].unique():
                    state_df = obj_df[obj_df['state_name'] == state]
                    total_count = state_df['activity_count'].sum()
                    if total_count > 0:
                        false_percentage = 100 - (total_count / 5000 * 100)
                    else:
                        false_percentage = 100  # If no activity, it's completely false
                    # Transform object name
                    obj_name = transform_object_name(f"{obj_type}_{obj_idx}")
                    flat_dist[(obj_name, state)] = false_percentage
    
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
        csv_path (str): Path to the CSV file containing algorithm run data
        pkl_path (str): Path to the agent distribution pickle file (default: post_processing/averaged_state_distributions.pkl)
        verbose (bool): Whether to print detailed debugging information
    
    Returns:
        dict: Dictionary containing JS divergence results
    """
    # Load distributions
    agent_dist = load_agent_distributions(pkl_path)
    csv_dist = calculate_csv_distributions(csv_path)
    
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
    
    # Get only keys that exist in both distributions
    common_keys = set(agent_dist.keys()) & set(csv_dist.keys())
    
    if verbose:
        print(f"\nNumber of common object-state pairs: {len(common_keys)}")
    
    # Calculate JS divergence for each object-state pair
    js_divergences = {}
    for key in sorted(common_keys):
        # Create binary distributions for this state
        agent_value = np.array([agent_dist[key], 100 - agent_dist[key]])
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
        print(f"\nJS Divergence Statistics:")
        print(f"  Average: {avg_js_div:.4f}")
        print(f"  Min: {min_js:.4f}")
        print(f"  Max: {max_js:.4f}")
        print(f"  Std: {std_js:.4f}")
        print(f"  Number of pairs: {len(js_divergences)}")
        
        print(f"\nTop 10 highest divergences:")
        sorted_divergences = sorted(js_divergences.items(), key=lambda x: x[1], reverse=True)
        for i, (key, js_div) in enumerate(sorted_divergences[:10]):
            print(f"  {i+1}. {key}: {js_div:.4f}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Calculate JS divergence between algorithm run and agent distribution')
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
Example command to get JS divergence:
python mini_behavior/post_processing/calculate_js_pipeline.py post_processing/merged_rnd_multi_seed.csv --verbose
"""