import pandas as pd
import matplotlib.pyplot as plt
import ast

from Globals import LoggingManager

def main():
    # Choose logging ID
    while True:
        print("\n", LoggingManager.LOG_IDS)
        potential_log_id = input("Enter logging ID: ").upper().strip()
        if potential_log_id in LoggingManager.LOG_IDS:
            LoggingManager.LOGGER_ID = potential_log_id
            break
        else:
            print("❌ Please enter valid ID\n")

    # Load dataset
    try:
        df = pd.read_csv(f"Logs/{LoggingManager.LOGGER_ID}Logs/IndividualLog.csv")
    except FileNotFoundError:
        df = pd.read_csv(f"SLEAP/Logs/{LoggingManager.LOGGER_ID}Logs/IndividualLog.csv")

    experiment_id = input("Enter experiment ID to analyze: ")
    if experiment_id == "":
        df = df[df["experiment_id"] == df["experiment_id"].max()]
    else:
        df = df[df["experiment_id"] == int(experiment_id)]

    df["individual"] = df["individual"].apply(ast.literal_eval).apply(lambda x: x[0])
    df[["kernel 1", "kernel 2", "kernel 3"]] = pd.DataFrame(df["individual"].tolist(), index=df.index)

    generations = sorted(df["generation"].unique())

    # Determine step mode or fixed generation
    step_mode = False
    generation_input = input("gen: ")
    if generation_input == "":
        step_mode = True
        start_index = 0
    else:
        start_index = generations.index(int(generation_input))

    # Setup the plot once
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter([], [], [], c=[], s=[])
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Fitness Score', fontsize=12, fontweight='bold')

    gen_index = start_index

    while 0 <= gen_index < len(generations):
        gen = generations[gen_index]
        gen_df = df[df["generation"] == gen].copy()

        norm_fitness = (gen_df["fitness"] - gen_df["fitness"].min()) / (gen_df["fitness"].max() - gen_df["fitness"].min())
        gen_df["size"] = 50 + 150 * norm_fitness

        ax.clear()

        scatter = ax.scatter(
            gen_df["kernel 1"], gen_df["kernel 2"], gen_df["kernel 3"],
            c=gen_df["fitness"],
            s=gen_df["size"],
            cmap="cividis",
            alpha=0.8,
            depthshade=True,
            edgecolor="w",
            linewidth=0.5
        )

        if "champion" in gen_df.columns and gen_df["champion"].any():
            champ = gen_df[gen_df["champion"]]
            ax.scatter(
                champ["kernel 1"], champ["kernel 2"], champ["kernel 3"],
                s=300,
                c='red',
                marker='*',
                edgecolor='gold',
                linewidth=1.5,
                label='Champion'
            )

        ax.set_xlabel('Kernel 1', fontsize=12, fontweight='bold')
        ax.set_ylabel('Kernel 2', fontsize=12, fontweight='bold')
        ax.set_zlabel('Kernel 3', fontsize=12, fontweight='bold')
        ax.set_title(f'Kernel-space with Fitness\nExperiment {experiment_id}, Generation {gen}', fontsize=16, pad=15)

        ax.set_xlim(0, 1500)
        ax.set_ylim(0, 1500)
        ax.set_zlim(0, 1500)
        ax.view_init(elev=25, azim=45)

        ax.grid(True, linestyle='--', alpha=0.7)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')

        cbar.update_normal(scatter)

        ax.text2D(0.05, 0.95,
                  f"Marker size represents fitness\n(min: {gen_df['fitness'].min():.3f}, max: {gen_df['fitness'].max():.3f})",
                  transform=ax.transAxes, fontsize=10,
                  bbox=dict(facecolor='white', alpha=0.8))

        plt.draw()
        plt.pause(0.001)

        # 🚀 Navigation controls
        user_input = input("➡️  Press [Enter] for next | [p] for previous | [q] to quit: ").strip().lower()

        if user_input == "q":
            print("👋 Exiting.")
            break
        elif user_input == "p":
            gen_index = max(0, gen_index - 1)
        else:
            gen_index += 1

    else:
        print("✅ Reached beginning or end of generations.")
        input("Press Enter to exit...")


if __name__ == "__main__":
    while True:
        main()
