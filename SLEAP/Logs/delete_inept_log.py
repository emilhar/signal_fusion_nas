import pandas as pd

# File paths
experiment_stats_path = "Logs/OLogs/ExperimentStatsLog.csv"
generation_stats_path = "Logs/OLogs/GenerationStatsLog.csv"
individual_log_path = "Logs/OLogs/IndividualLog.csv"

# Load ExperimentStatsLog to find the max Experiment_ID
experiment_stats_df = pd.read_csv(experiment_stats_path)
max_experiment_id = experiment_stats_df["experiment_id"].max()

# Function to filter and overwrite CSV
def filter_by_experiment_id(path, max_id):
    df = pd.read_csv(path)
    filtered_df = df[df["experiment_id"] <= max_id]
    filtered_df.to_csv(path, index=False)

# Apply the filtering
filter_by_experiment_id(generation_stats_path, max_experiment_id)
filter_by_experiment_id(individual_log_path, max_experiment_id)

print(f"✅ Filtered logs to keep only Experiment_ID ≤ {max_experiment_id}")
