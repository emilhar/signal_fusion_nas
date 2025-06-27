import pandas as pd
import matplotlib.pyplot as plt
import ast

# Load the CSV
stages = ("wake", "REM")
for i, stage in enumerate(stages):
    print(f"{i}: {stage}")

st = int(input("Choose stage: "))

df = pd.read_csv(f"{stages[st]}_full_train_log.csv")

# Parse the kernel_sizes column
df["kernel_sizes"] = df["kernel_sizes"].apply(ast.literal_eval)
df["kernel_id"] = df["kernel_sizes"].apply(lambda x: x[0][0])  # Extract first number

# Plot each unique kernel_id
plt.figure(figsize=(10, 6))
for kernel_id in df["kernel_id"].unique():
    sub_df = df[df["kernel_id"] == kernel_id]
    plt.plot(sub_df["epoch"], sub_df["f1"], label=f"kernel: {kernel_id}")

plt.title("F1 Score per Epoch by Kernel Size")
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
