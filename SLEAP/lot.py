import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./Logs/TLogs/GenerationStatsLog.csv")

# Convert relevant columns to float
df['fitness_mean'] = pd.to_numeric(df['fitness_mean'], errors='coerce')
df['loss_mean'] = pd.to_numeric(df['loss_mean'], errors='coerce')

# Get the two experiment IDs you want to compare
exp1_id = 4  # Replace with your first experiment ID
exp2_id = 5  # Replace with your second experiment ID

# Filter data for the two experiments
exp1_data = df[df['experiment_id'] == exp1_id]
exp2_data = df[df['experiment_id'] == exp2_id]

# Create the plot
plt.figure(figsize=(10, 6))

# Plot lines for each experiment
plt.plot(exp1_data['generation'], exp1_data['loss_mean'], 
         label=f'Our EA fitness function (Loss Mean)', marker='o')
plt.plot(exp2_data['generation'], exp2_data['loss_mean'], 
         label=f'Completely random fitness (Loss Mean)', marker='s')

# Add labels and title
plt.xlabel('Generation')
plt.ylabel('Train Loss')
plt.title('Comparison of minimizing loss vs. random selection')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()