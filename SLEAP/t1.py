import pandas as pd
import matplotlib.pyplot as plt
import ast


def main():
    experiment_id = int(input("Enter experiment ID to analyze: "))

    df = pd.read_csv("SLEAP/Logs/IndividualLog.csv")
    df = df[df["experiment_id"] == experiment_id]
    #df = df[df["generation"] == 1]

    df["individual"] = df["individual"].apply(ast.literal_eval)
    df["individual"] = df["individual"].apply(lambda x: x[0])

    df[["kernel 1", "kernel 2", "kernel 3"]] = pd.DataFrame(df["individual"].tolist(), index=df.index)

    # Normalize fitness for visualization
    norm_fitness = (df["fitness"] - df["fitness"].min()) / (df["fitness"].max() - df["fitness"].min())
    df["size"] = 50 + 150 * norm_fitness  # Marker size (50-200 range)

    #df["size"] = 50 + 300 * norm_fitness


    # Create 3D plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Scatter plot with color mapping
    cmap = "cividis"
    scatter = ax.scatter(
        df["kernel 1"], df["kernel 2"], df["kernel 3"],
        c=df["fitness"], 
        s=df["size"],
        cmap=cmap,
        alpha=0.8,
        depthshade=True,
        edgecolor="w",
        linewidth=0.5
    )

    if "champion" in df.columns and df["champion"].any():
        champ = df[df["champion"]]
        ax.scatter(
            champ["kernel 1"], champ["kernel 2"], champ["kernel 3"],
            s=300,  # Larger size
            c='red',
            marker='*',
            edgecolor='gold',
            linewidth=1.5,
            label='Champion'
        )

    ax.set_xlabel('Kernel 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('Kernel 2', fontsize=12, fontweight='bold')
    ax.set_zlabel('Kernel 3', fontsize=12, fontweight='bold')
    ax.set_title(f'Kernel-space with Fitness\nExperiment {experiment_id}', fontsize=16, pad=15)

    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Fitness Score', fontsize=12, fontweight='bold')

    min_fit, max_fit = df["fitness"].min(), df["fitness"].max()
    ax.text2D(0.05, 0.95, 
            f"Marker size represents fitness\n(min: {min_fit:.3f}, max: {max_fit:.3f})",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8))

    ax.grid(True, linestyle='--', alpha=0.7)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.view_init(elev=25, azim=45)

    #plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
