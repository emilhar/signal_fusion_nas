import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from Globals import Signal, Sleepstage
from GridSearch.KRNL import KRNL_GridSearch

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
sleep_stage = Sleepstage.REM
n_samples = 3000 if signal != Signal.EMG.SUBMENTAL else 30
krnl = KRNL_GridSearch(signal, sleep_stage, n_samples)
grid = krnl.grid
print(grid[0, 0, 0])



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

plot_krnl_grid(krnl, choice='time')
