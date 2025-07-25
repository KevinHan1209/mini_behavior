import pandas as pd
import pickle
from calculate_js_pipeline_new import load_agent_distributions, calculate_csv_distributions_new_format

def analyze_missing_states(csv_path, pkl_path="post_processing/averaged_state_distributions.pkl"):
    """Analyze which states are present in only one distribution"""
    
    # Load both distributions
    agent_dist = load_agent_distributions(pkl_path)
    csv_dist = calculate_csv_distributions_new_format(csv_path)
    
    # Get all keys
    agent_keys = set(agent_dist.keys())
    csv_keys = set(csv_dist.keys())
    common_keys = agent_keys & csv_keys
    agent_only = agent_keys - csv_keys
    csv_only = csv_keys - agent_keys
    
    print("="*80)
    print("OBJECT-STATE DISTRIBUTION ANALYSIS")
    print("="*80)
    print(f"\nTotal states in agent distribution: {len(agent_keys)}")
    print(f"Total states in CSV distribution: {len(csv_keys)}")
    print(f"Common states: {len(common_keys)}")
    print(f"States only in agent: {len(agent_only)}")
    print(f"States only in CSV: {len(csv_only)}")
    
    # Group by object
    print("\n" + "="*80)
    print("STATES ONLY IN AGENT DISTRIBUTION (from experiments)")
    print("="*80)
    
    agent_only_by_obj = {}
    for obj, state in sorted(agent_only):
        if obj not in agent_only_by_obj:
            agent_only_by_obj[obj] = []
        agent_only_by_obj[obj].append(state)
    
    for obj, states in sorted(agent_only_by_obj.items()):
        print(f"\n{obj}:")
        for state in sorted(states):
            print(f"  - {state}")
    
    print("\n" + "="*80)
    print("STATES ONLY IN CSV DISTRIBUTION (from training)")
    print("="*80)
    
    csv_only_by_obj = {}
    for obj, state in sorted(csv_only):
        if obj not in csv_only_by_obj:
            csv_only_by_obj[obj] = []
        csv_only_by_obj[obj].append(state)
    
    for obj, states in sorted(csv_only_by_obj.items()):
        print(f"\n{obj}:")
        for state in sorted(states):
            print(f"  - {state}")
    
    # Analyze patterns
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    # States that appear in CSV but not agent
    csv_only_states = set(state for _, state in csv_only)
    agent_only_states = set(state for _, state in agent_only)
    
    print("\nUnique state types only in CSV:")
    for state in sorted(csv_only_states):
        count = sum(1 for _, s in csv_only if s == state)
        print(f"  - {state}: appears for {count} objects")
    
    print("\nUnique state types only in agent:")
    for state in sorted(agent_only_states):
        count = sum(1 for _, s in agent_only if s == state)
        print(f"  - {state}: appears for {count} objects")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = "test/activity_logs/checkpoint_1500000_activity.csv"
    
    analyze_missing_states(csv_path)