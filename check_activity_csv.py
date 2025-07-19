#!/usr/bin/env python3
"""Quick script to check the activity CSV files"""

import os
import glob
import pandas as pd

checkpoint_dir = "checkpoints"
csv_dir = os.path.join(checkpoint_dir, "activity_logs")
csv_pattern = os.path.join(csv_dir, "checkpoint_*_activity.csv")
csv_files = sorted(glob.glob(csv_pattern))

if not csv_files:
    print(f"No CSV files found matching pattern: {csv_pattern}")
    print("Run training first to generate checkpoint CSVs.")
else:
    print(f"Found {len(csv_files)} checkpoint CSV files:")
    for csv_path in csv_files:
        print(f"\n{'='*60}")
        print(f"File: {os.path.basename(csv_path)}")
        
        df = pd.read_csv(csv_path)
        print(f"Shape: {df.shape}")
        print(f"Columns: {len(df.columns)} total")
        
        # Show checkpoint ID
        checkpoint_id = df['checkpoint_id'].iloc[0]
        print(f"Checkpoint ID: {checkpoint_id}")
        
        # Show non-zero counts
        row = df.iloc[0]
        non_zero = [(col, int(row[col])) for col in df.columns[1:] if row[col] > 0]
        if non_zero:
            print(f"\nTop 10 active states:")
            for col, count in sorted(non_zero, key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {col}: {count}")
            print(f"\nTotal active states: {len(non_zero)}")
            total_activity = sum(count for _, count in non_zero)
            print(f"Total activity count: {total_activity}")