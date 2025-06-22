import numpy as np
import matplotlib.pyplot as plt

# Use Computer Modern serif MathText fonts
plt.rcParams['text.usetex'] = False
plt.rc('font', family='serif')
plt.rc('mathtext', fontset='cm')

# Settings
generations = np.arange(1, 51)
col_titles = ['Wake', 'N1', 'N2', 'N3', 'REM']
row_labels = ['EEG1', 'EEG2', 'EOG', 'EMG']

# Create subplots with a 4x4 inch figure size
fig, axes = plt.subplots(4, 5, figsize=(4, 4), sharex=True, sharey=True)

# Uniform font sizes
title_fs = 10
label_fs = 10
tick_fs = 8

for i in range(4):
    for j in range(5):
        ax = axes[i, j]
        # Simulate data
        avg_accuracy = 68 + 25 * (1 - np.exp(-generations / 15)) + np.random.randn(50) * 0.7
        best_limited = avg_accuracy + np.linspace(3, 6, 50) + np.random.randn(50) * 0.4
        full_gains = np.repeat([3.5, 5.5, 6.5, 7.5, 9], 10) + np.random.randn(50) * 0.3
        best_full = avg_accuracy + full_gains
        
        # Plot: avg dotted, best solid, full as x marker, all in black, thin lines
        ax.plot(generations, avg_accuracy, linestyle=':', color='black', linewidth=0.4)
        ax.plot(generations, best_limited, linestyle='-', color='black', linewidth=0.6)
        ax.scatter(generations[9::10], best_full[9::10], marker='x', color='black', s=20, linewidths=0.8)
        
        # Remove gridlines
        ax.grid(False)
        
        # Titles and row labels
        if i == 0:
            ax.set_title(col_titles[j], fontsize=title_fs, pad=2)
        if j == 0:
            ax.set_ylabel(row_labels[i], fontsize=label_fs, labelpad=2)
        
        # Only bottom row shows x-axis
        if i < 3:
            ax.xaxis.set_visible(False)
        # Set tick label size
        ax.tick_params(labelsize=tick_fs)

# Shared x-axis label
fig.supxlabel('Generations', fontsize=label_fs, x=0.45, y=0.03)

# Remove legend and tighten layout
plt.tight_layout(pad=1, w_pad=0.5, h_pad=0.5)

plt.savefig('enas_plots_psg.pdf', format='pdf', bbox_inches='tight')
plt.show()

