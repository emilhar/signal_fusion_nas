import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from Globals import Signal, Sleepstage, DataManager
from GridSearch.KRNL import KRNL_GridSearch, KRNL_BayesianSearch

choices = [
    "train_loss",
    "test_loss",
    "precision",
    "recall",
    "accuracy",
    "branches",
    "best_f1",
    "best_auc",
    "time"
]

signal = Signal.EEG.Fpz_Cz
sleep_stage = Sleepstage.N1
n_samples = 3000 if signal != Signal.EMG.SUBMENTAL else 30
krnl = KRNL_GridSearch(signal, sleep_stage, DataManager.DatasetNames.EDF_78, 0.20, n_samples)
# grid = krnl.grid
# print(grid[0, 0, 0])
#krnl = KRNL_BayesianSearch(signal, sleep_stage, DataManager.DatasetNames.EDF_78, 0.25, n_samples)
#krnl.run_optimization()



def plot_krnl_grid(krnl, choice='best_f1', figsize=(14, 10), cmap='viridis_r', elev=25, azim=-45):
    """
    Visualize KRNL grid search results in 3D space with proper axis labeling
    
    Args:
        krnl: KRNL_GridSearch instance
        choice: Metric to visualize (default: 'best_f1')
        figsize: Figure dimensions
        cmap: Colormap for metric values
        elev: Elevation viewing angle
        azim: Azimuth viewing angle
    """
    # Extract grid and hyperparameters
    grid = krnl.grid
    kernels = krnl.kernels
    layers = krnl.layers
    reductions = krnl.reductions
    
    # Prepare data arrays
    X, Y, Z = [], [], []  # Coordinates
    C = []                # Metric values
    sizes = []            # Point sizes (for visual clarity)
    labels = []           # Point labels
    

    # Collect data from grid
    for x in range(len(kernels)):
        for y in range(len(reductions)):
            for z in range(len(layers)):
                result = grid[x, y, z]
                X.append(kernels[x])
                Y.append(krnl.reduction_to_name[reductions[y]])
                Z.append(layers[z])
                C.append(result[choice])
                
                # Dynamic point sizing for better visibility
                sizes.append(120 - (layers[z] * 15))
                labels.append(f"K:{kernels[x]} L:{layers[z]} R:{Y[-1]}")

    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Convert reduction names to numeric values for plotting
    reduction_map = {name: i for i, name in enumerate(set(Y))}
    Y_numeric = [reduction_map[name] for name in Y]
    
    # Create scatter plot with color mapping
    sc = ax.scatter(X, Y_numeric, Z, c=C, s=sizes, cmap=cmap, 
                   alpha=0.85, edgecolor='k', linewidth=0.5)
    
    # Configure axes
    ax.set_xlabel('Starting Kernel Size', labelpad=15, fontsize=12)
    ax.set_ylabel('Reduction Function', labelpad=15, fontsize=12)
    ax.set_zlabel('Layer Count', labelpad=15, fontsize=12)
    
    # Set axis ticks
    ax.set_xticks(np.linspace(min(kernels), max(kernels), 5))
    ax.set_yticks(list(reduction_map.values()))
    ax.set_yticklabels(list(reduction_map.keys()))
    ax.set_zticks(layers)
    
    # Add colorbar
    cbar = fig.colorbar(sc, pad=0.1, shrink=0.7)
    cbar.set_label(choice.upper(), fontsize=12)
    
    # Set viewing angle
    ax.view_init(elev=elev, azim=azim)
    
    # Add title and grid
    plt.title(f'KRNL Grid Search: {choice.upper()} Results\n'
             f'Signal: {krnl.signal}, Stage: {krnl.sleep_stage}', 
             fontsize=14, pad=20)
    ax.grid(True, alpha=0.25)
    
    # Add hover annotations (optional - for interactive environments)
    try:
        from mpl_toolkits.mplot3d import proj3d
        from matplotlib.text import Annotation
        
        class HoverAnnotation(Annotation):
            def __init__(self, text, xyz, *args, **kwargs):
                super().__init__(text, xy=(0,0), *args, **kwargs)
                self.xyz = xyz
            
            def draw(self, renderer):
                x2, y2, z2 = proj3d.proj_transform(*self.xyz, self.axes.M)
                self.xy = (x2, y2)
                super().draw(renderer)
        
        annot = ax.annotate("", xy=(0,0), textcoords="offset points",
                           bbox=dict(boxstyle="round", fc="w", alpha=0.8),
                           ha="center", fontsize=9)
        annot.set_visible(False)
        
        def hover(event):
            vis = annot.get_visible()
            if event.inaxes == ax:
                cont, ind = sc.contains(event)
                if cont:
                    idx = ind["ind"][0]
                    annot.xy = (event.xdata, event.ydata)
                    annot.set_text(labels[idx])
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    if vis:
                        annot.set_visible(False)
                        fig.canvas.draw_idle()
        
        fig.canvas.mpl_connect("motion_notify_event", hover)
    except ImportError:
        pass
    
    plt.tight_layout()
    plt.show()


def plot_krnl_2d(krnl, reduction, metric='best_f1', time_metric='time', 
                 figsize=(14, 8), color_map='viridis', layers=[1, 2, 3, 4]):
    """
    Plot 2D visualization of KRNL grid search results for a fixed reduction function.
    Shows metric performance and training time across kernel sizes and layer counts.
    
    Args:
        krnl: KRNL_GridSearch instance
        reduction: Reduction function name to fix (e.g., 'max', 'min', 'mean')
        metric: Performance metric to visualize (default: 'best_f1')
        time_metric: Time metric to visualize (default: 'time')
        figsize: Figure dimensions
        color_map: Colormap for layer differentiation
    """
    # Validate inputs
    if reduction not in krnl.reduction_to_name.values():
        raise ValueError(f"Invalid reduction name. Available options: {list(krnl.reduction_to_name.values())}")
    
    # Get hyperparameters
    kernels = krnl.kernels
    reductions = krnl.reductions
    
    # Find reduction index
    reduction_idx = None
    for idx, func in enumerate(reductions):
        if krnl.reduction_to_name[func] == reduction:
            reduction_idx = idx
            break
    
    if reduction_idx is None:
        raise ValueError("Reduction function not found in grid")
    
    # Prepare data storage
    metric_data = {layer: [] for layer in layers}
    time_data = {layer: [] for layer in layers}
    
    # Extract data from grid
    for z, layer in enumerate(layers):
        for x, kernel in enumerate(kernels):
            result = krnl.grid[x, reduction_idx, z]
            metric_data[layer].append(result[metric])
            time_data[layer].append(result[time_metric])
    
    # Create figure with dual axes
    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()
    
    # Generate colormap
    colors = plt.get_cmap(color_map)(np.linspace(0, 1, len(layers)))
    
    # Create equidistant positions for kernel sizes
    x_positions = np.arange(len(kernels))
    
    # Plot metric and time for each layer
    for (layer, color) in zip(layers, colors):
        # Metric plot (primary axis)
        metric_line = ax1.plot(
            x_positions, 
            metric_data[layer], 
            marker='o', 
            markersize=8,
            linestyle='-', 
            linewidth=2.5,
            color=color,
            label=f'L={layer} ({metric})'
        )
        
        # Time plot (secondary axis)
        time_line = ax2.plot(
            x_positions, 
            time_data[layer], 
            marker='s', 
            markersize=6,
            linestyle='--', 
            linewidth=1.5,
            color=color,
            alpha=0.7,
            label=f'L={layer} ({time_metric})'
        )
    
    # Configure plot aesthetics
    ax1.set_xlabel('Kernel Size', fontsize=12)
    ax1.set_ylabel(metric.upper(), fontsize=12)
    ax2.set_ylabel(f'{time_metric.upper()} (seconds)', fontsize=12)
    
    # Set equidistant x-ticks with kernel size labels
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(kernels)
    
    ax1.set_title(
        f'KRNL Performance: {reduction} Reduction\n'
        f'Signal: {krnl.signal}, Stage: {krnl.sleep_stage}',
        fontsize=14,
        pad=15
    )
    
    # Add gridlines
    ax1.grid(True, linestyle='--', alpha=0.6, axis='y')
    
    # Add vertical gridlines at kernel positions
    for pos in x_positions:
        ax1.axvline(x=pos, color='gray', linestyle=':', alpha=0.3)
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2, 
        labels1 + labels2, 
        loc='upper left' if metric == 'train_loss' else 'lower right',
        frameon=True,
        framealpha=0.9,
        ncol=min(2, len(layers))  # Use 2 columns if many layers
    )
    
    plt.tight_layout()
    plt.show()

import json
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_optimization_results(signal, sleep_stage):
    # Load results
    results_dir = "./Data/bayesian_results/"
    filename = f"{results_dir}{signal}_{sleep_stage}.json"
    
    try:
        with open(filename, 'r') as f:
            history = json.load(f)
    except FileNotFoundError:
        print(f"Results file not found: {filename}")
        return
    
    if not history:
        print("No results found in the file")
        return
    
    # Extract data for plotting
    iterations = list(range(1, len(history) + 1))
    f1_scores = [entry['result']['test_f1'] for entry in history]
    cumulative_max = np.maximum.accumulate(f1_scores)
    
    # Create figure with subplots
    plt.figure(figsize=(15, 10))
    
    # Plot 1: F1 score progression
    plt.subplot(2, 2, 1)
    plt.plot(iterations, f1_scores, 'bo-', alpha=0.6, label='F1 Score')
    plt.plot(iterations, cumulative_max, 'r-', linewidth=2, label='Cumulative Max')
    plt.xlabel('Iteration')
    plt.ylabel('F1 Score')
    plt.title('F1 Score Progression')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Layer count histogram
    plt.subplot(2, 2, 2)
    layer_counts = [entry['params'][0] for entry in history]
    plt.hist(layer_counts, bins=[1, 2, 3, 4, 5], align='left', rwidth=0.8)
    plt.xlabel('Number of Layers')
    plt.ylabel('Frequency')
    plt.title('Layer Count Distribution')
    plt.xticks([1, 2, 3, 4])
    plt.grid(axis='y')
    
    # Plot 3: Kernel size distribution
    plt.subplot(2, 2, 3)
    all_kernels = []
    for entry in history:
        all_kernels.extend(entry['branch'])
    plt.hist(all_kernels, bins=50, alpha=0.7)
    plt.xlabel('Kernel Size')
    plt.ylabel('Frequency')
    plt.title('Kernel Size Distribution')
    plt.grid(True)
    
    # Plot 4: Best configuration
    best_idx = np.argmax(f1_scores)
    best_entry = history[best_idx]
    best_branch = best_entry['branch']
    
    plt.subplot(2, 2, 4)
    plt.bar(range(1, len(best_branch) + 1), best_branch, color='green')
    plt.xlabel('Layer Position')
    plt.ylabel('Kernel Size')
    plt.title(f'Best Configuration (F1={f1_scores[best_idx]:.4f})')
    plt.xticks(range(1, len(best_branch) + 1))
    
    # Main title and layout
    plt.suptitle(f'Bayesian Optimization Results: {signal} - {sleep_stage}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save and show
    plt.savefig(f"{results_dir}{signal}_{sleep_stage}_plot.png")
    plt.show()
    
    # Print best configuration details
    print("\nBest Configuration Found:")
    print(f"F1 Score: {f1_scores[best_idx]:.4f}")
    print(f"Number of Layers: {len(best_branch)}")
    print(f"Kernel Sizes: {best_branch}")
    print(f"Iteration: {best_idx + 1}")




#plot_krnl_grid(krnl, choice='train_loss')
plot_krnl_2d(krnl, reduction='halve', metric='train_loss', layers=[2, 3, 4])
