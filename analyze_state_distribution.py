#!/usr/bin/env python3
"""
Generate state activity distribution from most to least frequented.
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

# Calculate total activity per state across all checkpoints
total_per_state = activity_matrix.sum(axis=0)

# Sort from most to least active
sorted_indices = np.argsort(total_per_state)[::-1]
sorted_activity = total_per_state[sorted_indices]

# Create visualization
fig = plt.figure(figsize=(14, 6))

# Plot the sorted distribution
plt.plot(range(len(sorted_activity)), sorted_activity, linewidth=2, color='steelblue')
plt.fill_between(range(len(sorted_activity)), sorted_activity, alpha=0.3, color='steelblue')

plt.xlabel('State Rank (Most to Least Active)', fontsize=12)
plt.ylabel('Total Activity Count', fontsize=12)
plt.title('State Activity Distribution: All States Ranked by Frequency', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)

# Add some annotations for key points
max_activity = sorted_activity[0]
num_active = np.sum(sorted_activity > 0)
num_inactive = np.sum(sorted_activity == 0)

# Annotate the plot
plt.text(0.02, 0.98, f'Total states: {len(sorted_activity)}\nActive states: {num_active}\nInactive states: {num_inactive}\nMax activity: {int(max_activity)}',
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Add vertical line at the cutoff between active and inactive
if num_active < len(sorted_activity):
    plt.axvline(x=num_active, color='red', linestyle='--', alpha=0.5, label=f'Active/Inactive cutoff ({num_active} states)')
    plt.legend()

plt.tight_layout()
plt.savefig('/home/kevin/mini_behavior/state_activity_distribution.png', dpi=300, bbox_inches='tight')
print("\nSaved visualization to state_activity_distribution.png")

# Print statistics
print("\n" + "="*60)
print("STATE ACTIVITY DISTRIBUTION STATISTICS")
print("="*60)
print(f"Total number of state flags: {len(sorted_activity)}")
print(f"Active states (count > 0): {num_active}")
print(f"Inactive states (count = 0): {num_inactive}")
print(f"Percentage active: {100 * num_active / len(sorted_activity):.1f}%")
print(f"\nMaximum activity: {int(max_activity)}")
print(f"Mean activity (all states): {np.mean(sorted_activity):.2f}")
print(f"Mean activity (active states only): {np.mean(sorted_activity[sorted_activity > 0]):.2f}")
print(f"Median activity (active states only): {np.median(sorted_activity[sorted_activity > 0]):.2f}")
print(f"Std dev (active states only): {np.std(sorted_activity[sorted_activity > 0]):.2f}")

# Quantiles
if num_active > 0:
    active_sorted = sorted_activity[sorted_activity > 0]
    print(f"\nActivity quantiles (active states only):")
    print(f"  25th percentile: {np.percentile(active_sorted, 25):.2f}")
    print(f"  50th percentile (median): {np.percentile(active_sorted, 50):.2f}")
    print(f"  75th percentile: {np.percentile(active_sorted, 75):.2f}")
    print(f"  90th percentile: {np.percentile(active_sorted, 90):.2f}")
    print(f"  95th percentile: {np.percentile(active_sorted, 95):.2f}")

print("="*60)
