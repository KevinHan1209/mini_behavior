#!/usr/bin/env python3
"""
Generate top 25 most active states visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import csv

# Load all CSV files
csv_files = sorted(Path('/home/kevin/mini_behavior').glob('wandb_export_2025-11-08T14*.csv'))
print(f"Found {len(csv_files)} evaluation files")

# Load data
activity_vectors = []
state_info = []

for i, csv_file in enumerate(csv_files):
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        activities = []
        if i == 0:  # Only need to get state info once
            for row in reader:
                activities.append(int(row['activity_count']))
                state_info.append({
                    'object_type': row['object_type'],
                    'object_index': row['object_index'],
                    'state_name': row['state_name']
                })
        else:
            for row in reader:
                activities.append(int(row['activity_count']))

        activity_vectors.append(np.array(activities))
    print(f"Checkpoint {i}: {csv_file.name} - {len(activities)} state flags")

activity_matrix = np.array(activity_vectors)

# Create visualization
fig = plt.figure(figsize=(12, 10))

# Top 25 most active states across all checkpoints
total_per_state = activity_matrix.sum(axis=0)
top_25_indices = np.argsort(total_per_state)[-25:][::-1]

# Get state names for top 25
state_labels = [f"{state_info[idx]['object_type']}_{state_info[idx]['object_index']}_{state_info[idx]['state_name']}"
                for idx in top_25_indices]

plt.barh(range(25), total_per_state[top_25_indices], color='coral')
plt.yticks(range(25), state_labels, fontsize=9)
plt.xlabel('Total Activity Count', fontsize=12)
plt.title('Top 25 Most Active States Across All Checkpoints', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('/home/kevin/mini_behavior/top_25_states.png', dpi=300, bbox_inches='tight')
print("\nSaved visualization to top_25_states.png")

# Print the values
print("\n" + "="*60)
print("TOP 25 MOST ACTIVE STATES")
print("="*60)
for i, idx in enumerate(top_25_indices):
    info = state_info[idx]
    count = total_per_state[idx]
    print(f"{i+1:2d}. {info['object_type']:20s}_{info['object_index']}_{info['state_name']:25s} : {int(count):4d}")
print("="*60)
