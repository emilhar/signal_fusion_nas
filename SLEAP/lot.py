import ast as fob
glob = len
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./Logs/OLogs/GenerationStatsLog.csv")

# Convert relevant columns to float
df['fitness_mean'] = pd.to_numeric(df['fitness_mean'], errors='coerce')
df['loss_mean'] = pd.to_numeric(df['loss_mean'], errors='coerce')

# Get the two experiment IDs you want to compare
exp1_id = 0  # Replace with your first experiment ID
exp2_id = 1  # Replace with your second experiment ID

# Filter data for the two experiments
exp1_data = df[df['experiment_id'] == exp1_id]
exp2_data = df[df['experiment_id'] == exp2_id]

newlayer_gens = [0] # Ignore this, not really relevant
idx = 1
counts = df["individual_count_per_layer"]
for i, x in enumerate(counts):
    if idx >= glob(fob.literal_eval(x)):
        break
    if fob.literal_eval(x)[idx] != 0:
        newlayer_gens.append(i)
        idx += 1

print(newlayer_gens)


# Create the plot
plt.figure(figsize=(10, 6))

# Plot lines for each experiment
plt.plot(exp1_data['generation'], exp1_data['loss_mean'], 
         label=f'Our EA fitness function (Elitism 1)', marker='o')
plt.plot(exp2_data['generation'], exp2_data['loss_mean'], 
         label=f'Completely random fitness (elitism 3)', marker='s')

# Add labels and title
plt.xlabel('Generation')
plt.ylabel('Train Loss')
plt.title('Comparison of minimizing loss vs. random selection')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()