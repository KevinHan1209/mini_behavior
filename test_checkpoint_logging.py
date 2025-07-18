import pandas as pd
import os

def analyze_checkpoint_csv(csv_path):
    """Analyze the consolidated checkpoint CSV file"""
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    print(f"\nCSV Analysis for: {csv_path}")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)[:5]}... (showing first 5)")
    
    # Check unique checkpoints
    if 'checkpoint_id' in df.columns:
        checkpoints = df['checkpoint_id'].unique()
        print(f"\nUnique checkpoints: {checkpoints}")
        print(f"Number of checkpoints: {len(checkpoints)}")
        
        # For each checkpoint, check episodes and steps
        for checkpoint in checkpoints:
            checkpoint_data = df[df['checkpoint_id'] == checkpoint]
            episodes = checkpoint_data['episode'].unique()
            print(f"\nCheckpoint {checkpoint}:")
            print(f"  Episodes: {len(episodes)} (should be 10)")
            print(f"  Total steps: {len(checkpoint_data)} (should be ~2000)")
            
            # Check steps per episode
            for ep in episodes[:3]:  # Show first 3 episodes
                ep_steps = len(checkpoint_data[checkpoint_data['episode'] == ep])
                print(f"    Episode {ep}: {ep_steps} steps")
    
    # Check binary flag columns
    binary_columns = [col for col in df.columns if col not in ['checkpoint_id', 'episode', 'step']]
    print(f"\nBinary flag columns: {len(binary_columns)}")
    
    # Calculate total activity across all flags
    if binary_columns:
        total_activity = df[binary_columns].sum().sum()
        print(f"Total activity across all flags: {total_activity}")
        
        # Show most active flags
        flag_activity = df[binary_columns].sum().sort_values(ascending=False)
        print(f"\nTop 5 most active flags:")
        for flag, count in flag_activity.head().items():
            print(f"  {flag}: {count} changes")

if __name__ == "__main__":
    # Expected CSV location after training
    csv_path = "checkpoints/all_checkpoints_activity.csv"
    analyze_checkpoint_csv(csv_path)