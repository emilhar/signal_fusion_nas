import pandas as pd
import matplotlib.pyplot as plt
import ast
import numpy as np
from matplotlib import colormaps

# GLOBAL SETTINGS (adjust these as needed)
COLORMAP_NAME = 'plasma'  # Change this to any valid colormap name ('viridis', 'plasma', etc.)
LINE_WIDTH = 2.5         # Default line width
DEFAULT_METRIC = 'mean_loss'  # Default aggregation metric
NEW_LAYER_LINE_STYLE = '--'  # Style for new layer indicator lines
NEW_LAYER_ALPHA = 0.75        # Transparency for new layer lines
ST = "train_loss"


def analyze_by_metric(experiment_ids, ID, metric=DEFAULT_METRIC, linewidth=LINE_WIDTH, st = ST):
    """
    Advanced by_metric visualization with customizable metrics
    
    Parameters:
        experiment_ids (list): List of experiment IDs to analyze (e.g., [0, 1, 2])
        metric (str): 'mean_loss', 'max_loss', 'min_loss', 'median_loss'
        linewidth (float): Thickness of plot lines
    """
    df = pd.read_csv(f'Logs/{ID}Logs/IndividualLog.csv')
    df = df[df['experiment_id'].isin(experiment_ids)]
    gen_df = pd.read_csv(f"./Logs/{ID}Logs/GenerationStatsLog.csv")
    gen_df = gen_df[gen_df["experiment_id"] == experiment_ids[0]]
    newlayer_gens = [0] # Ignore this, not really relevant
    idx = 1
    counts = gen_df["individual_count_per_layer"]
    for i, x in enumerate(counts):
        if idx >= len(ast.literal_eval(x)):
            break
        if ast.literal_eval(x)[idx] != 0:
            newlayer_gens.append(i)
            idx += 1
        
            

    # 2. Parse JSON and extract by_metric
    def safe_parse(perf_str):
        try:
            return abs(ast.literal_eval(perf_str)[st])  # Absolute value
        except:
            return np.nan
    
    df[st] = df['model_performance'].apply(safe_parse)
    df = df.dropna(subset=[st])

    # 3. Calculate specified metric
    METRICS = {
        'mean_loss': 'mean',
        'max_loss': 'max',
        'min_loss': 'min', 
        'median_loss': 'median'
    }
    
    if metric not in METRICS:
        print(f"Invalid metric. Using {DEFAULT_METRIC}. Options: {list(METRICS.keys())}")
        metric = DEFAULT_METRIC
        
    loss_data = df.groupby(['experiment_id', 'generation', 'layer'])[st].agg(METRICS[metric]).reset_index()

    # 4. Visualization with modern colormap API
    plt.figure(figsize=(12, 7))
    
    # Get colormap - modern API
    try:
        cmap = colormaps[COLORMAP_NAME]
    except KeyError:
        print(f"Colormap '{COLORMAP_NAME}' not found. Using 'viridis' instead.")
        cmap = colormaps['viridis']
    
    # Color and style setup
    layers = sorted(loss_data['layer'].unique())
    experiments = sorted(loss_data['experiment_id'].unique())
    
    colors = cmap(np.linspace(0, 1, len(layers)))
    linestyles = ['-', '--', '-.', ':'] * 2  # Supports up to 8 experiments
    

    # Plot each combination
    for exp_idx, exp in enumerate(experiments):
        for layer_idx, layer in enumerate(layers):
            subset = loss_data[(loss_data['experiment_id'] == exp) & 
                             (loss_data['layer'] == layer)]
            
            if not subset.empty:
                plt.plot(subset['generation'], subset[st],
                        color=colors[layer_idx],
                        linestyle=linestyles[exp_idx],
                        linewidth=linewidth,
                        label=f'Exp {exp} - Layer {layer}')

    ymin, ymax = plt.ylim()
    for gen in newlayer_gens:
        if gen > 0:  # Skip generation 0
            plt.axvline(x=gen, color='gray', 
                       linestyle=NEW_LAYER_LINE_STYLE,
                       alpha=NEW_LAYER_ALPHA,
                       linewidth=1.5)
            plt.text(gen, ymax*1.03, f'Layer {newlayer_gens.index(gen)}', 
                    rotation=0, va='top', ha='right',
                    alpha=0.7, fontsize=10)

    # 5. Perfect styling
    plt.title(f'Train Loss ({metric.replace("_", " ").title()})', fontsize=14, pad=20)
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    plt.grid(True, alpha=0.15)
    
    legend = plt.legend(bbox_to_anchor=(1.05, 1), 
                       loc='upper left',
                       frameon=True,
                       title="Experiments & Layers")
    legend.get_frame().set_edgecolor('black')
    
    plt.tight_layout()
    plt.show()

# Example usage:
ID = input("id please: [T/O]: ")
ID = ID.upper()
while True:
    ids = int(input("id: "))
    analyze_by_metric([6, 8], ID)  # Uses global settings
    # To change colormap: modify COLORMAP_NAME at the top