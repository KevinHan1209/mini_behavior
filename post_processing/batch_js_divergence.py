import os
import glob
from calculate_js_pipeline_new import calculate_js_divergence_pipeline
import argparse

def process_activity_logs(activity_logs_dir, pkl_path="post_processing/averaged_state_distributions.pkl", verbose=False):
    """
    Process all CSV files in the activity_logs directory and calculate JS divergence for each.
    
    Args:
        activity_logs_dir (str): Path to the activity_logs directory
        pkl_path (str): Path to the agent distribution pickle file
        verbose (bool): Whether to print detailed information for each file
    """
    # Find all CSV files in the directory
    csv_pattern = os.path.join(activity_logs_dir, "*.csv")
    csv_files = sorted(glob.glob(csv_pattern))
    
    if not csv_files:
        print(f"No CSV files found in {activity_logs_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files in {activity_logs_dir}\n")
    
    # Process each CSV file
    results = []
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        print(f"Processing {filename}...")
        
        try:
            # Calculate JS divergence
            result = calculate_js_divergence_pipeline(csv_file, pkl_path, verbose=False)
            
            # Extract checkpoint number from filename (e.g., checkpoint_1000000_activity.csv)
            checkpoint_num = None
            if filename.startswith("checkpoint_") and "_activity.csv" in filename:
                try:
                    checkpoint_num = int(filename.split("_")[1])
                except:
                    pass
            
            results.append({
                'filename': filename,
                'checkpoint': checkpoint_num,
                'js_divergence': result['average_js_divergence'],
                'min_js': result['min_js_divergence'],
                'max_js': result['max_js_divergence'],
                'std_js': result['std_js_divergence'],
                'num_pairs': result['num_object_state_pairs']
            })
            
            print(f"  ✓ JS Divergence: {result['average_js_divergence']:.4f}")
            
        except Exception as e:
            print(f"  ✗ Error processing {filename}: {str(e)}")
            results.append({
                'filename': filename,
                'checkpoint': checkpoint_num,
                'js_divergence': None,
                'error': str(e)
            })
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY OF JS DIVERGENCE SCORES")
    print("="*70)
    
    # Sort results by checkpoint number if available
    if all(r.get('checkpoint') is not None for r in results if 'error' not in r):
        results.sort(key=lambda x: x.get('checkpoint', 0))
    
    print(f"{'Checkpoint':<15} {'JS Divergence':<15} {'Min':<10} {'Max':<10} {'Std':<10} {'Pairs':<10}")
    print("-"*70)
    
    for result in results:
        if 'error' in result:
            print(f"{result['filename']:<15} {'ERROR':<15} {result.get('error', '')}")
        else:
            checkpoint_str = f"{result['checkpoint']:,}" if result['checkpoint'] else result['filename'][:15]
            print(f"{checkpoint_str:<15} {result['js_divergence']:<15.4f} {result['min_js']:<10.4f} {result['max_js']:<10.4f} {result['std_js']:<10.4f} {result['num_pairs']:<10}")
    
    # Calculate overall statistics
    valid_results = [r for r in results if 'error' not in r and r['js_divergence'] is not None]
    if valid_results:
        print("\n" + "="*70)
        print("OVERALL STATISTICS")
        print("="*70)
        
        avg_js = sum(r['js_divergence'] for r in valid_results) / len(valid_results)
        min_js = min(r['js_divergence'] for r in valid_results)
        max_js = max(r['js_divergence'] for r in valid_results)
        
        print(f"Average JS Divergence across all checkpoints: {avg_js:.4f}")
        print(f"Minimum JS Divergence: {min_js:.4f}")
        print(f"Maximum JS Divergence: {max_js:.4f}")
        print(f"Number of checkpoints processed: {len(valid_results)}")
        
        # Show trend if checkpoints are sequential
        if all(r.get('checkpoint') is not None for r in valid_results):
            print("\nJS Divergence Trend:")
            for result in valid_results:
                bar_length = int(result['js_divergence'] * 100)  # Scale for visualization
                bar = "█" * bar_length
                print(f"  Checkpoint {result['checkpoint']:>10,}: {bar} {result['js_divergence']:.4f}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Calculate JS divergence for all CSV files in activity_logs directory')
    parser.add_argument('activity_logs_dir', help='Path to the activity_logs directory')
    parser.add_argument('--pkl-path', default='post_processing/averaged_state_distributions.pkl',
                       help='Path to the agent distribution pickle file')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Print detailed information for each file')
    
    args = parser.parse_args()
    
    # Check if directory exists
    if not os.path.isdir(args.activity_logs_dir):
        print(f"Error: Directory '{args.activity_logs_dir}' not found")
        return
    
    # Check if pickle file exists
    if not os.path.exists(args.pkl_path):
        print(f"Error: Pickle file '{args.pkl_path}' not found")
        return
    
    # Process all CSV files
    process_activity_logs(args.activity_logs_dir, args.pkl_path, args.verbose)

if __name__ == "__main__":
    main()

"""
Example usage:
python post_processing/batch_js_divergence.py test/activity_logs/
"""