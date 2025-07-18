#!/usr/bin/env python3
"""Quick script to check the activity CSV format"""

import os
import pandas as pd

csv_path = "checkpoints/all_checkpoints_activity.csv"

if not os.path.exists(csv_path):
    print(f"CSV file not found: {csv_path}")
    print("Run training first to generate the CSV.")
else:
    df = pd.read_csv(csv_path)
    print(f"\nCSV Shape: {df.shape}")
    print(f"\nColumns ({len(df.columns)}):")
    print(df.columns.tolist()[:10], "..." if len(df.columns) > 10 else "")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Show non-zero counts
    print("\nNon-zero activity counts:")
    for idx, row in df.iterrows():
        checkpoint = row['checkpoint_id']
        non_zero = [(col, int(row[col])) for col in df.columns[1:] if row[col] > 0]
        if non_zero:
            print(f"\nCheckpoint {checkpoint}:")
            for col, count in sorted(non_zero, key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {col}: {count}")