import os
import csv

def remove_experiment_from_csv(file_path, experiment_id):
    """Remove all rows matching the experiment_id from a CSV file"""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return False
    
    # Read all rows except those matching the experiment_id
    rows_to_keep = []
    with open(file_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader, None)  # preserve header
        if header:
            rows_to_keep.append(header)
        
        for row in reader:
            if row and row[0] != experiment_id:
                rows_to_keep.append(row)
    
    # Write the filtered rows back to the file
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows_to_keep)
    
    return True

def main():
    experiment_id = input("Enter the experiment ID you want to delete: ").strip()
    if not experiment_id:
        print("No experiment ID provided. Exiting.")
        return
    
    confirmation = input(f'WARNING: This will permanently delete all data for experiment {experiment_id}. Enter "CONFIRM" to continue: ').strip()
    if confirmation != "CONFIRM":
        print("Deletion cancelled.")
        return
    
    # Define the files to process
    files = [
        "Logs/OLogs/GenerationStatsLog.csv",
        "Logs/OLogs/ExperimentStatsLog.csv",
        "Logs/OLogs/IndividualLog.csv"
    ]
    
    # Process each file
    for file in files:
        print(f"Processing {file}...")
        success = remove_experiment_from_csv(file, experiment_id)
        if success:
            print(f" - Removed entries for experiment {experiment_id}")
        else:
            print(f" - No changes made to {file}")
    
    print("Operation complete.")

if __name__ == "__main__":
    main()