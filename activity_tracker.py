#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Load the data from CSV. Make sure the CSV file is in the same directory as this script.
    # The CSV file should have columns: object_type, state, activity_count
    data_file = "2mil_ep5.csv"
    try:
        data = pd.read_csv(data_file)
    except FileNotFoundError:
        print(f"Error: The file '{data_file}' was not found.")
        return

    # Define the states to exclude
    excluded_states = ['infovofrobot', 'inreachofrobot']

    # Filter out rows with the excluded states
    filtered_data = data[~data['state_name'].isin(excluded_states)]

    # Group by object type and sum the activity counts
    summary = filtered_data.groupby('object_type', as_index=False)['activity_count'].sum()

    # Display the results in the terminal
    print("Activity count per object type:")
    print(summary)

    # Visualize the results using a bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(summary['object_type'], summary['activity_count'], color='skyblue')
    plt.xlabel("Object Type")
    plt.ylabel("Total Activity Count")
    plt.title("Activity Count per Object Type")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
