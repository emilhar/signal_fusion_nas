# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # Load CSV and select latest experiment
# df = pd.read_csv("./Logs/OLogs/IndividualLog.csv")
# df = df[df["Experiment_ID"] == df["Experiment_ID"].max()]

# # Extract data
# generations = df["Generation"].to_numpy()
# fitness = df["Train_Loss"].to_numpy()

# # Plot as scatter (no lines)
# plt.figure(figsize=(10, 6))
# plt.scatter(generations, fitness, color='blue')

# # Labeling and formatting
# plt.xlabel("Generation")
# plt.ylabel("Train_Loss")
# plt.title("Fitness over Generations for Latest Experiment")
# plt.grid(True)

# # Make sure x-axis ticks are all integers
# plt.xticks(sorted(df["Generation"].unique()))

# plt.tight_layout()
# plt.show()
a = [1,2,3,4]
for i in range(4):
    print(a[i:])