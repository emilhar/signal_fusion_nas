import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("./Logs/TLogs/GenerationStatsLog.csv")

# Get the two experiment IDs you want to compare
# For example, let's assume you want to compare 'exp1' and 'exp2'
exp1_id = 2  # Replace with your first experiment ID
exp2_id = 3  # Replace with your second experiment ID

# Filter data for the two experiments
exp1_data = df[df['experiment_id'] == exp1_id]
exp2_data = df[df['experiment_id'] == exp2_id]

# Create the plot
plt.figure(figsize=(10, 6))

# Plot lines for each experiment
plt.plot(exp1_data['generation'], exp1_data['fitness_mean'], 
         label=f'Experiment {exp1_id}', marker='o')
plt.plot(exp2_data['generation'], exp2_data['fitness_mean'], 
         label=f'Experiment {exp2_id}', marker='s')

# Add labels and title
plt.xlabel('Generation')
plt.ylabel('Average Fitness')
plt.title('Average Fitness Across Generations')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()