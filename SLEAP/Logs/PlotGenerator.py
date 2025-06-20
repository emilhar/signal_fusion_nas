import pandas as pd
import matplotlib.pyplot as plt
import os

# Load logs
exp_stats = pd.read_csv("Logs/ExperimentStatsLog.csv")
gen_stats = pd.read_csv("Logs/GenerationStatsLog.csv")
individuals = pd.read_csv("Logs/IndividualLog.csv")

colors = plt.cm.tab10.colors

def main():
    # Display experiment options
    print("\nAvailable Experiments:")
    for idx, row in exp_stats.iterrows():
        print(f"  [{row['experiment_id']}] sleepstage: {row['sleepstage']}, signal_type: {row['signal_type']}")

    # Prompt user for selection
    selection = input("\nEnter an experiment ID to view or type 'all' to view all: ").strip()

    # Set up the plot
    plt.figure(figsize=(12, 8))

    # Handle single or all
    if selection.lower() == "all":
        title = "Validation Accuracy over Generations (All Experiments)"
        for i, exp_id in enumerate(exp_stats["experiment_id"].unique()):
            plot_experiment(exp_id, i, plot_all=True)
        exp_id="All"
    else:
        while True:
            exp_id = int(selection)
            if exp_id not in exp_stats["experiment_id"].values:
                print("Invalid")
                selection = input("\nEnter an experiment ID to view or type 'all' to view all: ").strip()
            else:
                break 
        plot_experiment(exp_id)
        title = get_experiment_title(exp_id)

    # Finalize plot
    plt.xlabel("Generation")
    plt.ylabel("Validation Accuracy (%)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"Logs/Plots/Exp{exp_id}.png")

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
    os.system('cls' if os.name == 'nt' else 'clear')

def get_experiment_title(exp_id):
    exp_row = exp_stats[exp_stats["experiment_id"] == exp_id].iloc[0]
    sleepstage = exp_row["sleepstage"]
    signal_type = exp_row["signal_type"]
    title = f"Exp {exp_id}: {sleepstage}, {signal_type}\nValidation Accuracy over Generations"

    return title

def plot_experiment(exp_id, color_idx=0,plot_all=False): ## ALL [signal type]
    gen_df = gen_stats[gen_stats["experiment_id"] == exp_id]
    ind_df = individuals[(individuals["experiment_id"] == exp_id) & (individuals["champion"])]

    # Fetch sleepstage and signal_type
    exp_row = exp_stats[exp_stats["experiment_id"] == exp_id].iloc[0]
    sleepstage = exp_row["sleepstage"]
    signal_type = exp_row["signal_type"]

    color_base = colors[color_idx % len(colors)]
    color_dark = tuple(c * 0.6 for c in color_base)

    label_suffix = f"(Exp {exp_id}: {sleepstage}, {signal_type})"

    # Plot average and max fitness over generations
    plt.plot(gen_df["generation"], gen_df["fitness_mean"] * 100,
             label=f"Avg {label_suffix}", color=color_base, linewidth=2)
    plt.plot(gen_df["generation"], gen_df["fitness_max"] * 100,
             label=f"Best {label_suffix}", color=color_dark, linewidth=2)

    # Mark ToC generations and champion dots
    toc_generations = gen_df[gen_df["tournament_of_champions"]]["generation"]

    if not plot_all:
        for gen in toc_generations:
            plt.axvline(x=gen, color='#d62728', linestyle="--", linewidth=2, alpha=0.6)

            # Get champion(s) from this generation
            champions_in_gen = ind_df[ind_df["generation"] == gen]
            if not champions_in_gen.empty:
                # Get the champion with the best accuracy
                best_row = champions_in_gen.loc[champions_in_gen["fitness"].idxmax()]
                acc = best_row["fitness"]*100
                plt.scatter(gen, acc, color='#d62728', edgecolor='black', zorder=5, s=80,
                            label="Best Champion (ToC)" if gen == toc_generations.iloc[0] else "")  # Avoid duplicate label


    for gen in gen_df[gen_df["tournament_of_champions"]]["generation"]:
        plt.axvline(x=gen, color='#d62728', linestyle="--", linewidth=2, alpha=0.6)
        
while True:
    main()