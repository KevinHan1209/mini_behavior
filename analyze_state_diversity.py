#!/usr/bin/env python3
"""
Analyze state diversity across evaluation episodes from NovelD training.
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
# Checkpoints are at 500k intervals starting from 500k
checkpoint_timesteps = [(i + 1) * 500_000 for i in range(len(csv_files))]
checkpoints = [f"{ts//1000}k" for ts in checkpoint_timesteps]
checkpoint_labels_full = [f"CP {i}\n({ts//1000}k)" for i, ts in enumerate(checkpoint_timesteps)]

# ========== Visualizations ==========

fig = plt.figure(figsize=(16, 8))

# 1. Total activity per checkpoint
ax1 = plt.subplot(2, 2, 1)
total_activity = activity_matrix.sum(axis=1)
ax1.bar(range(len(checkpoints)), total_activity, color='steelblue')
ax1.set_xlabel('Evaluation Checkpoint')
ax1.set_ylabel('Total State Changes')
ax1.set_title('Total State Activity Per Checkpoint')
ax1.set_xticks(range(len(checkpoints)))
ax1.set_xticklabels(checkpoints)
ax1.grid(axis='y', alpha=0.3)

# 2. Number of unique active states per checkpoint
ax2 = plt.subplot(2, 2, 2)
num_active_states = (activity_matrix > 0).sum(axis=1)
ax2.bar(range(len(checkpoints)), num_active_states, color='darkgreen')
ax2.set_xlabel('Evaluation Checkpoint')
ax2.set_ylabel('Number of Active States')
ax2.set_title('Number of Unique States Changed Per Checkpoint')
ax2.set_xticks(range(len(checkpoints)))
ax2.set_xticklabels(checkpoints)
ax2.grid(axis='y', alpha=0.3)

# 3. Cosine similarity between consecutive checkpoints
ax3 = plt.subplot(2, 2, 3)
similarities = []
for i in range(len(activity_vectors) - 1):
    v1 = activity_vectors[i].astype(float)
    v2 = activity_vectors[i + 1].astype(float)
    # Cosine similarity
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 > 0 and norm2 > 0:
        sim = np.dot(v1, v2) / (norm1 * norm2)
    else:
        sim = 0
    similarities.append(sim)

ax3.plot(range(1, len(similarities) + 1), similarities, marker='o', linewidth=2, markersize=8, color='purple')
ax3.set_xlabel('Checkpoint Transition')
ax3.set_ylabel('Cosine Similarity')
ax3.set_title('Behavioral Similarity Between Consecutive Checkpoints')
ax3.set_ylim([0, 1.05])
ax3.grid(alpha=0.3)

# 4. Top 10 most active states across all checkpoints
ax4 = plt.subplot(2, 2, 4)
total_per_state = activity_matrix.sum(axis=0)
top_10_indices = np.argsort(total_per_state)[-10:][::-1]

# Get state names for top 10
state_labels = [f"{state_info[idx]['object_type']}_{state_info[idx]['object_index']}_{state_info[idx]['state_name']}"
                for idx in top_10_indices]

ax4.barh(range(10), total_per_state[top_10_indices], color='coral')
ax4.set_yticks(range(10))
ax4.set_yticklabels(state_labels, fontsize=8)
ax4.set_xlabel('Total Activity Count')
ax4.set_title('Top 10 Most Active States Across All Checkpoints')
ax4.invert_yaxis()
ax4.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('/home/kevin/mini_behavior/state_diversity_analysis.png', dpi=300, bbox_inches='tight')
print("\nSaved visualization to state_diversity_analysis.png")

# ========== Quantitative Analysis ==========

print("\n" + "="*60)
print("QUANTITATIVE ANALYSIS")
print("="*60)

print(f"\nTotal number of state flags tracked: {activity_matrix.shape[1]}")

for i, checkpoint in enumerate(checkpoints):
    total = total_activity[i]
    unique = num_active_states[i]
    avg = total / unique if unique > 0 else 0
    print(f"\n{checkpoint_labels_full[i]}:")
    print(f"  Total state changes: {total}")
    print(f"  Unique states changed: {unique}")
    print(f"  Average changes per active state: {avg:.2f}")

print("\n" + "-"*60)
print("DIVERSITY METRICS")
print("-"*60)

# Jaccard similarity (overlap of active states)
print("\nJaccard Similarity (active state overlap):")
for i in range(len(activity_vectors) - 1):
    binary_v1 = (activity_vectors[i] > 0).astype(int)
    binary_v2 = (activity_vectors[i + 1] > 0).astype(int)
    intersection = np.sum(binary_v1 & binary_v2)
    union = np.sum(binary_v1 | binary_v2)
    jaccard = intersection / union if union > 0 else 0
    print(f"  {checkpoints[i]} -> {checkpoints[i+1]}: {jaccard:.3f}")

# Cosine similarity
print("\nCosine Similarity (activity pattern similarity):")
for i, sim in enumerate(similarities):
    print(f"  {checkpoints[i]} -> {checkpoints[i+1]}: {sim:.3f}")

# Check if behavior is converging (high similarity between later checkpoints)
print("\n" + "-"*60)
print("CONVERGENCE ANALYSIS")
print("-"*60)

if len(similarities) >= 2:
    early_avg = np.mean(similarities[:2]) if len(similarities) >= 2 else similarities[0]
    late_avg = np.mean(similarities[-2:]) if len(similarities) >= 2 else similarities[-1]

    print(f"\nAverage similarity (early checkpoints): {early_avg:.3f}")
    print(f"Average similarity (late checkpoints): {late_avg:.3f}")

    if late_avg > 0.9:
        print("\n✓ HIGH CONVERGENCE: Behavior is very consistent in later checkpoints")
        print("  → Entropy should be decreasing (deterministic policy)")
    elif late_avg > 0.7:
        print("\n~ MODERATE CONVERGENCE: Some consistency but still variation")
        print("  → Check if entropy is stabilizing")
    else:
        print("\n✗ LOW CONVERGENCE: High variation between checkpoints")
        print("  → Entropy may still be high (stochastic policy)")

# Trend analysis
print("\n" + "-"*60)
print("TREND ANALYSIS")
print("-"*60)

if len(similarities) >= 3:
    # Check if similarity is increasing over time (converging behavior)
    early_half = similarities[:len(similarities)//2]
    late_half = similarities[len(similarities)//2:]

    trend = "increasing" if np.mean(late_half) > np.mean(early_half) else "decreasing"
    print(f"\nSimilarity trend: {trend}")

    if trend == "increasing":
        print("  → Behavior is becoming more deterministic over training")
        print("  → This correlates with DECREASING entropy")
    else:
        print("  → Behavior is becoming more varied over training")
        print("  → This is unusual - check training dynamics")

print("\n" + "="*60)
