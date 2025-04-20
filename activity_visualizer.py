#!/usr/bin/env python3
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Specify the folder containing the CSV files.
    folder_path = "activity/checkpoint_2000000"  # Change this to your folder path.
    
    # Get all CSV files in the folder.
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in the folder: {folder_path}")
        return

    # Read each CSV file and collect them into a list.
    data_frames = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            data_frames.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    # Combine all the DataFrames into one.
    data = pd.concat(data_frames, ignore_index=True)

    # Define the states to exclude.
    excluded_states = ['infovofrobot', 'inreachofrobot']

    # Filter out rows with the excluded states.
    filtered_data = data[~data['state_name'].isin(excluded_states)]

    # Group by both object type and state_name, and sum the activity counts.
    summary = filtered_data.groupby(['object_type', 'state_name'], as_index=False)['activity_count'].sum()

    # Create a pivot table with object_type as rows and state_name as columns.
    pivot_table = summary.pivot(index='object_type', columns='state_name', values='activity_count').fillna(0)

    # Calculate the overall total number of interactions.
    overall_total = filtered_data['activity_count'].sum()

    # Plot the stacked bar chart.
    ax = pivot_table.plot(kind='bar', stacked=True, figsize=(10, 6))

    # Annotate each bar with the total count for that object type.
    totals = pivot_table.sum(axis=1)
    for i, total in enumerate(totals):
        ax.text(i, total + max(totals)*0.01, int(total), ha='center', va='bottom')

    # Add the overall total interactions as a text box in the plot.
    plt.text(0.02, 0.95, f"Overall Total Interactions: {int(overall_total)}",
             transform=plt.gca().transAxes,
             fontsize=12,
             verticalalignment='top',
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.xlabel("Object Type")
    plt.ylabel("Total Activity Count")
    plt.title("Stacked Activity Count per Object Type by Interaction")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
