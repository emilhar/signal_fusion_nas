import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
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

krnl = KRNL_GridSearch(Signal.EOG.HORIZONTAL, Sleepstage.REM)
grid = krnl.grid
print(grid[0, 0, 0])


def plot_3d_grid(krnl, choice='best_f1'):
    """
    Creates a 3D visualization of grid search results
    
    Args:
        krnl: KRNL_GridSearch instance
        choice: Metric to visualize (default: 'best_f1')
    """
    grid = krnl.grid
    n_kernels, n_reductions, n_layers = grid.shape
    
    # Extract metric values
    metric_values = np.zeros((n_kernels, n_reductions, n_layers))
    for i in range(n_kernels):
        for j in range(n_reductions):
            for k in range(n_layers):
                metric_values[i, j, k] = grid[i, j, k][choice]
    
    # Create coordinate grids
    X, Y, Z = np.meshgrid(
        np.arange(n_kernels),
        np.arange(n_reductions),
        np.arange(n_layers),
        indexing='ij'
    )
    
    # Setup plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create scatter plot
    sc = ax.scatter(
        X.flatten(), Y.flatten(), Z.flatten(),
        c=metric_values.flatten(),
        cmap='viridis',
        s=100,
        alpha=0.8,
        marker='o',
        depthshade=False
    )
    
    # Add colorbar
    cbar = plt.colorbar(sc, pad=0.1)
    cbar.set_label(choice.upper(), fontsize=12)
    
    # Label axes
    ax.set_xlabel('Kernel Size', fontsize=12, labelpad=15)
    ax.set_ylabel('Reduction Function', fontsize=12, labelpad=15)
    ax.set_zlabel('Layer Count', fontsize=12, labelpad=15)
    
    # Adjust layout
    plt.title(f'3D Grid Search: {choice.upper()} Visualization', fontsize=16)
    plt.tight_layout()
    plt.show()

# Example usage
plot_3d_grid(krnl, choice='best_f1')



