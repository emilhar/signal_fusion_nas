import pandas as pd
import matplotlib.pyplot as plt

# Load logs
exp_stats = pd.read_csv("Logs/ExperimentStatsLog.csv")
gen_stats = pd.read_csv("Logs/GenerationStatsLog.csv")
individuals = pd.read_csv("Logs/IndividualLog.csv")

# Ensure proper types
gen_stats["tournament_of_champions"] = gen_stats["tournament_of_champions"].astype(bool)
individuals["champion"] = individuals["champion"].astype(bool)

# Display experiment options
print("\nAvailable Experiments:")
for idx, row in exp_stats.iterrows():
    print(f"  [{row['experiment_id']}] sleepstage: {row['sleepstage']}, signal_type: {row['signal_type']}")

# Prompt user for selection
selection = input("\nEnter an experiment ID to view or type 'all' to view all: ").strip()

# Set up the plot
plt.figure(figsize=(10, 6))

# Color palette
colors = plt.cm.tab10.colors

# Function to plot one experiment
def plot_experiment(exp_id, color_idx=0):

    gen_df = gen_stats[gen_stats["experiment_id"] == exp_id]
    ind_df = individuals[individuals["experiment_id"] == exp_id]

    color_base = colors[color_idx % len(colors)]
    color_dark = tuple(c * 0.8 for c in color_base)

    plt.plot(gen_df["generation"], gen_df["fitness_mean"] * 100,
             label=f"Avg (Exp {exp_id})", color=color_base, linewidth=2)
    
    plt.plot(gen_df["generation"], gen_df["fitness_max"] * 100,
             label=f"Best (Exp {exp_id})", color=color_dark, linewidth=2)
    
    plt.scatter(ind_df["generation"], ind_df["Accuracy"] * 100,
                color=color_base, edgecolor="black", s=80, zorder=5, label=f"Full Trained (Exp {exp_id})")

    for gen in gen_df[gen_df["tournament_of_champions"]]["generation"]:
        plt.axvline(x=gen, color=color_dark, linestyle="--", linewidth=1, alpha=0.6)

# Handle single or all
if selection.lower() == "all":
    for i, exp_id in enumerate(exp_stats["experiment_id"].unique()):
        plot_experiment(exp_id, i)
else:
    try:
        exp_id = int(selection)
        if exp_id not in exp_stats["experiment_id"].values:
            raise ValueError()
        plot_experiment(exp_id)
    except ValueError:
        print("Invalid experiment ID.")
        exit()

# Finalize plot
plt.xlabel("Generation")
plt.ylabel("Validation Accuracy (%)")
plt.title("Validation Accuracy over Generations")
plt.legend()
plt.grid(True)
plt.tight_layout()
#plt.savefig("experiment_accuracy_plot.png", dpi=300)

# Gather all y-values for dynamic scaling
all_accuracies = []

for exp_id in exp_stats["experiment_id"].unique() if selection.lower() == "all" else [int(selection)]:
    gen_df = gen_stats[gen_stats["experiment_id"] == exp_id]
    ind_df = individuals[(individuals["experiment_id"] == exp_id) & (individuals["champion"])]

    all_accuracies.extend(gen_df["fitness_mean"] * 100)
    all_accuracies.extend(gen_df["fitness_max"] * 100)
    all_accuracies.extend(ind_df["Accuracy"] * 100)

# Determine max with margin
if all_accuracies:
    ymax = max(all_accuracies)
    plt.ylim(top=ymax * 1.05)  # Add 5% headroom


plt.show()
