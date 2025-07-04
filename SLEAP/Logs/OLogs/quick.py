import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('Logs/OLogs/GenerationStatsLog.csv')

# Get unique experiment IDs
experiment_ids = df['Experiment_ID'].unique()
print(f"Available experiments: {experiment_ids}")

# Ask user to select an experiment
while True:
    try:
        selected_exp = int(input("Enter the Experiment_ID you want to visualize: "))
        if selected_exp not in experiment_ids:
            raise ValueError
        break
    except ValueError:
        print(f"Please enter a valid Experiment_ID from: {experiment_ids}")

# Filter data for selected experiment
exp_data = df[df['Experiment_ID'] == selected_exp]

# Get all unique generations (as integers)
generations = exp_data['Generation'].unique()
generations = sorted(generations.astype(int))  # Convert to integers and sort

# Plotting
plt.figure(figsize=(12, 6))

# Plot mean fitness
plt.plot(exp_data['Generation'], exp_data['Fitness_Mean'], 
        label='Average Fitness', color='blue', linewidth=2, marker='o')

# Plot min and max as points
plt.scatter(exp_data['Generation'], exp_data['Fitness_Min'], 
           label='Min Fitness', color='red', marker='v', alpha=0.7, s=100)
plt.scatter(exp_data['Generation'], exp_data['Fitness_Max'], 
           label='Max Fitness', color='green', marker='^', alpha=0.7, s=100)

# Customize x-axis to show every generation as integer
plt.xticks(generations, labels=[str(g) for g in generations])
plt.xlim(min(generations)-0.5, max(generations)+0.5)  # Add some padding

# Add labels and title
plt.xlabel('Generation (Integer Values)')
plt.ylabel('Fitness')
plt.title(f'Fitness Evolution for Experiment {selected_exp}')
plt.legend()
plt.grid(True, alpha=0.3)

# Show the plot
plt.tight_layout()
plt.savefig("YoMama")
plt.show()