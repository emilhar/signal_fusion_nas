import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Load the data from CSV
df = pd.read_csv('Logs/OLogs/IndividualLog.csv')

def plot_individual_trainloss(individual_id, attr):
    # Filter data for the specific individual
    individual_data = df[df['experiment_id'] == 1]
    individual_data = individual_data[individual_data['individual_id'] == individual_id]
    
    if individual_data.empty:
        print(f"No data found for individual ID {individual_id}")
        return
    
    # Create x-axis values (training run numbers)
    x = np.arange(1, len(individual_data)+1)
    y = individual_data[attr].values
    
    # Calculate linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    line = slope * x + intercept
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Scatter plot of actual points
    plt.scatter(x, y, color='royalblue', s=100, label=attr, zorder=3)
    
    # Trend line
    plt.plot(x, line, color='crimson', linestyle='--', 
             label=f'Trend (slope: {slope:.3f}, R²: {r_value**2:.3f}), Standard Error: {std_err:.3f}')
    
    # Customize the plot
    plt.title(f'{attr} for Individual ID {individual_id}\nWith Linear Trend Line', pad=20)
    plt.xlabel('Training Run Number')
    plt.ylabel(f'{attr} Value')
    plt.xticks(x)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Annotate each point with its value
    for i, txt in enumerate(y):
        plt.annotate(f"{txt:.3f}", (x[i], y[i]), 
                    textcoords="offset points", 
                    xytext=(0,10), ha='center')
    
    plt.tight_layout()
    plt.show()

# Interactive prompt
print("✨ TrainLoss Scatter Plot with Trend Line ✨")
thing = input("attr: ")
while True:
    print("\nEnter an individual ID to view (or 'q' to quit)")
    user_input = input("Individual ID: ").strip()
    
    if user_input.lower() == 'q':
        print("Goodbye! 👋")
        break    

    individual_id = int(user_input)
    plot_individual_trainloss(individual_id, thing)