import pandas as pd
import pickle
import os

# Load the pickle file to get agent distribution object names
def get_pkl_object_names(pkl_path="averaged_state_distributions.pkl"):
    with open(pkl_path, 'rb') as f:
        agent_dists = pickle.load(f)
    
    # Object name mapping from the original pipeline
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
    
    pkl_objects = set()
    for obj in agent_dists.keys():
        if obj in obj_name_mapping:
            pkl_objects.add(obj_name_mapping[obj])
        else:
            pkl_objects.add(obj)
    
    return sorted(pkl_objects)

# Get object names from old CSV format
def get_old_csv_object_names(csv_path="merged_rnd_multi_seed.csv"):
    df = pd.read_csv(csv_path)
    
    # Object pairs to average
    object_pairs = {
        'coin': 'piggie bank',
        'gear_toy': 'gear',
        'shape_toy': 'shape sorter'
    }
    
    csv_objects = set()
    for obj_type in df['object_type'].unique():
        if obj_type in object_pairs:
            csv_objects.add(object_pairs[obj_type])
        else:
            # Transform object name by replacing underscores with spaces
            obj_name = obj_type.replace('_', ' ')
            # Special case for beach ball
            if obj_name.startswith('beach ball'):
                obj_name = 'pink beach ball'
            csv_objects.add(obj_name)
    
    return sorted(csv_objects)

# Get object names from new CSV format
def get_new_csv_object_names(csv_path="../test/activity_logs/checkpoint_1000000_activity.csv"):
    df = pd.read_csv(csv_path)
    
    # Extract unique object names from column headers
    csv_objects = set()
    
    for column in df.columns:
        if column == 'checkpoint_id':
            continue
        
        # Parse column name
        parts = column.split('_')
        
        # Find where the state name starts
        state_start_idx = None
        for i, part in enumerate(parts):
            if part.isdigit():
                state_start_idx = i + 1
                break
        
        if state_start_idx is None:
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
            continue
        
        # Extract object type
        if state_start_idx > 2:
            object_type = '_'.join(parts[:state_start_idx-1])
        else:
            object_type = parts[0]
        
        # Transform to readable name
        obj_name = object_type.replace('_', ' ')
        if obj_name.startswith('beach ball'):
            obj_name = 'pink beach ball'
        
        csv_objects.add(obj_name)
    
    return sorted(csv_objects)

# Main comparison
print("=== OBJECT NAME COMPARISON ===\n")

# Get object names from pickle file
pkl_objects = get_pkl_object_names()
print(f"Pickle file objects ({len(pkl_objects)}):")
for obj in pkl_objects:
    print(f"  - {obj}")

print("\n" + "="*50 + "\n")

# Get object names from old CSV format
old_csv_objects = get_old_csv_object_names()
print(f"Old CSV format objects ({len(old_csv_objects)}):")
for obj in old_csv_objects:
    print(f"  - {obj}")

print("\n" + "="*50 + "\n")

# Get object names from new CSV format
new_csv_objects = get_new_csv_object_names()
print(f"New CSV format objects ({len(new_csv_objects)}):")
for obj in new_csv_objects:
    print(f"  - {obj}")

print("\n" + "="*50 + "\n")

# Show differences
print("Objects in pickle but not in new CSV:")
for obj in sorted(set(pkl_objects) - set(new_csv_objects)):
    print(f"  - {obj}")

print("\nObjects in new CSV but not in pickle:")
for obj in sorted(set(new_csv_objects) - set(pkl_objects)):
    print(f"  - {obj}")

print("\n" + "="*50 + "\n")

# Show the mapping needed
print("Suggested mappings for new CSV to match pickle:")
new_only = sorted(set(new_csv_objects) - set(pkl_objects))
pkl_only = sorted(set(pkl_objects) - set(new_csv_objects))

mappings = {
    'bucket toy': 'bucket',
    'cart toy': 'cart',
    'cube cabinet': 'winnie cabinet',
    'mini broom': 'broom',
    'ring toy': 'yellow donut ring',
    'winnie': 'winnie cabinet'
}

for new_obj in new_only:
    if new_obj in mappings:
        print(f"  '{new_obj}' -> '{mappings[new_obj]}'")
    else:
        print(f"  '{new_obj}' -> ???")