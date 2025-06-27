import pandas as pd
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
for kernel_id in df["kernel_id"].unique():
    sub_df = df[df["kernel_id"] == kernel_id]
    print(f"{kernel_id}, f1:{sub_df["f1"].max()}, tl:{sub_df["train_loss"].min()}")
