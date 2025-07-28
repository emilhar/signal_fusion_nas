import os

from Globals import Signal, Sleepstage, DataManager
from GridSearch.KRNL import QKernel_GridSearch, GridSearch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec



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


# for _ in range(9999999):
#     for signal in Signal.ALL_SIGNALS:
#         n_samples = 3000 if signal != Signal.EMG.SUBMENTAL else 30
#         qgrid = QKernel_GridSearch(signal, Sleepstage.N2, DataManager.DatasetNames.EDF_78, GridSearch._RunType.no_k0, 0.20, 1, n_samples)
#         qgrid.plot_qkernel_3d_vstime(metric="best_auc")
#         qgrid.plot_qkernel_slice_vstime(metric="best_auc", grid_steps=1000)

def main():
    # while True:
    #     for signal in (Signal.EEG.Fpz_Cz, Signal.EEG.Pz_Oz):
    #         for sleep_stage in (Sleepstage.WAKE, Sleepstage.N1, Sleepstage.N2, Sleepstage.N3, Sleepstage.REM):
    #             for runtype in (GridSearch._RunType.no_k0_1_filter_full,):
    #                 perc = 1.00 if runtype == GridSearch._RunType.no_k0_1_filter_full else 0.20
    #                 epochs = 10 if runtype == GridSearch._RunType.no_k0_1_filter_full else 5
    #                 qgrid = QKernel_GridSearch(
    #                     signal,
    #                     sleep_stage, 
    #                     DataManager.DatasetNames.EDF_78, 
    #                     runtype, 
    #                     perc, 
    #                     epochs, 
    #                     3000
    #                 )
    #                 print("\n\nNew Grid:")
    #                 print(qgrid.runtype)
    #                 print(qgrid.signal)
    #                 print(qgrid.sleep_stage)
    #                 qgrid.plot_qkernel_3d_vstime()
    #                 qgrid.plot_qkernel_3d_vstime(metric="best_auc")

    qgrid = QKernel_GridSearch(Signal.EEG.Fpz_Cz, Sleepstage.N2, DataManager.DatasetNames.EDF_78, GridSearch.RunType.any, 1.00, 40, 3000)
    qgrid.compute_grid()
    if qgrid:
        qgrid.plot_qkernel_3d_vstime(metric="best_auc")
        qgrid.plot_qkernel_slice_vstime(fixed_kernel="k1", fixed_value=50, metric="train_loss")
    # plot_all_head2head_results(qgrid, (4, 4, 4), (4, 4, 4), name="")#, name="40epoch-nodropout-same-v1", save_dir="1-layer", save=True)
    #plot_all_head2head_results(qgrid, (4, 4, 2), (4, 4, 1))#, name="40epoch-nodropout-similar-v3", save_dir="1-layer", save=True)
    #plot_all_head2head_results(qgrid, (4, 4, 4), (5, 1, 2))#, name="40epoch-nodropout-different-v2", save_dir="1-layer", save=True)

def plot_probability_differences(results, **kwargs):
    raise DeprecationWarning()
    save_dir, prefix = kwargs.get("save_dir", None), kwargs.get("prefix", None)

    prob_diffs = [x['prob_diff'] for x in results['prob_analysis']['all_probs']]

    fig = plt.figure(figsize=(16, 5))
    gs = GridSpec(1, 3, width_ratios=[1, 1, 0.95])
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    
    # Histogram with custom bins
    bins = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ax1.hist(prob_diffs, bins=bins, edgecolor='black', color='#4e79a7')
    ax1.set_title('Distribution of Probability Differences\n(in all infernece)')
    ax1.set_xlabel('Absolute Probability Difference')
    ax1.set_ylabel('Count')
    
    # Add count labels to each bar
    for rect in ax1.patches:
        height = rect.get_height()
        if height > 0:
            ax1.annotate(f'{int(height)}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    # Boxplot
    ax2.boxplot(prob_diffs, vert=False, patch_artist=True,
               boxprops=dict(facecolor='#f28e2b'))
    ax2.set_title('Boxplot of Probability Differences\n(in all infernece)')
    ax2.set_xlabel('Absolute Probability Difference')
    
    # Violin plot
    parts = ax3.violinplot(prob_diffs, vert=False, showmedians=True)
    for pc in parts['bodies']:
        pc.set_facecolor('#59a14f')
        pc.set_edgecolor('black')
        pc.set_alpha(1)
    
    ax3.set_title('Violin Plot of Probability Differences\n(in all infernece)')
    ax3.set_xlabel('Absolute Probability Difference')
    ax3.set_yticks([])
    
    # Add summary statistics
    stats_text = (
        f"Total inferences: {len(prob_diffs)}\n"
        f"Mean difference: {np.mean(prob_diffs):.3f}\n"
        f"Median difference: {np.median(prob_diffs):.3f}\n"
        f"Max difference: {np.max(prob_diffs):.3f}"
    )
    fig.text(0.92, 0.5, stats_text, 
             bbox=dict(facecolor='white', alpha=0.8),
             va='center', ha='left')
    
    plt.tight_layout()
    if save_dir is not None:
        base_dir = f"./_misc/{save_dir}"
        os.makedirs(base_dir, exist_ok=True)
        if prefix:
            prefix_dir = f"{base_dir}/{prefix}"
            os.makedirs(prefix_dir, exist_ok=True)
            save_path = f"{prefix_dir}/prob_diffs"
        else:
            save_path = f"{base_dir}/prob_diffs"
            
        plt.savefig(save_path)
    else:
        plt.show()


def plot_disagreement_probability_differences(results, **kwargs):
    raise DeprecationWarning()
    save_dir, prefix = kwargs.get("save_dir", None), kwargs.get("prefix", None)

    disagreements = results['prob_analysis']['disagreement_probs']
    prob_diffs = [x['prob_diff'] for x in disagreements]

    fig = plt.figure(figsize=(16, 5))
    gs = GridSpec(1, 3, width_ratios=[1, 1, 0.95])
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    
    bins = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ax1.hist(prob_diffs, bins=bins, edgecolor='black', color='#4e79a7')
    ax1.set_title('Distribution of Probability Differences\n(in Disagreements)')
    ax1.set_xlabel('Absolute Probability Difference')
    ax1.set_ylabel('Count')
    
    # Add count labels to each bar
    for rect in ax1.patches:
        height = rect.get_height()
        if height > 0:  # Only label bars with data
            ax1.annotate(f'{int(height)}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    # Boxplot
    ax2.boxplot(prob_diffs, vert=False, patch_artist=True,
               boxprops=dict(facecolor='#f28e2b'))
    ax2.set_title('Boxplot of Probability Differences\n(in Disagreements)')
    ax2.set_xlabel('Absolute Probability Difference')
    
    # Violin plot
    parts = ax3.violinplot(prob_diffs, vert=False, showmedians=True)
    for pc in parts['bodies']:
        pc.set_facecolor('#59a14f')
        pc.set_edgecolor('black')
        pc.set_alpha(1)
    
    ax3.set_title('Violin Plot of Probability Differences\n(in Disagreements)')
    ax3.set_xlabel('Absolute Probability Difference')
    ax3.set_yticks([])
    
    # Add summary statistics
    stats_text = (
        f"Total disagreements: {len(disagreements)}\n"
        f"Mean difference: {np.mean(prob_diffs):.3f}\n"
        f"Median difference: {np.median(prob_diffs):.3f}\n"
        f"Max difference: {np.max(prob_diffs):.3f}"
    )
    fig.text(0.92, 0.5, stats_text, 
             bbox=dict(facecolor='white', alpha=0.8),
             va='center', ha='left')
    
    plt.tight_layout()
    if save_dir is not None:
        base_dir = f"./_misc/{save_dir}"
        os.makedirs(base_dir, exist_ok=True)
        if prefix:
            prefix_dir = f"{base_dir}/{prefix}"
            os.makedirs(prefix_dir, exist_ok=True)
            save_path = f"{prefix_dir}/disag_prob_diffs"
        else:
            save_path = f"{base_dir}/disag_prob_diffs"
            
        plt.savefig(save_path)
    else:
        plt.show()


def plot_case_analysis(results, **kwargs):
    raise DeprecationWarning()
    save_dir, prefix = kwargs.get("save_dir", None), kwargs.get("prefix", None)
    disagreements = results['prob_analysis']['disagreement_probs']
    
    bins = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
    bin_labels = [
        '0-0.05', 
        '0.05-0.1', 
        '0.1-0.2', 
        '0.2-0.3', 
        '0.3-0.4', 
        '0.4-0.5', 
        '0.5+'
    ]
    
    prob_diffs = [x['prob_diff'] for x in disagreements]
    counts, _ = np.histogram(prob_diffs, bins=bins)
    
    prob_diff_data = dict(zip(bin_labels, counts))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    case_counts = {
        'Model 1 Correct\nModel 2 Wrong': len(results['examples']['model1_correct_model2_wrong']),
        'Model 2 Correct\nModel 1 Wrong': len(results['examples']['model2_correct_model1_wrong']),
    }
    
    colors_bar = ['#1f77b4', '#ff7f0e']
    ax1.bar(case_counts.keys(), case_counts.values(), color=colors_bar)
    ax1.set_title('Disagreement Cases')
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=45)
    
    colors_pie = plt.cm.viridis(np.linspace(0, 1, len(bin_labels)))
    
    pie_labels = [label for label, count in prob_diff_data.items() if count > 0]
    pie_counts = [count for count in prob_diff_data.values() if count > 0]
    pie_colors = [colors_pie[i] for i, count in enumerate(counts) if count > 0]
    
    wedges, texts, autotexts = ax2.pie(
        pie_counts,
        labels=pie_labels,
        autopct='%1.1f%%',
        colors=pie_colors,
        startangle=90,
        pctdistance=0.85
    )
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_weight('bold')
    
    centre_circle = plt.Circle((0,0), 0.70, fc='white')
    ax2.add_artist(centre_circle)
    
    ax2.set_title('Probability Differences in Disagreements')
    
    ax2.legend(
        wedges,
        [f"{label} ({count})" for label, count in zip(pie_labels, pie_counts)],
        title="Probability Difference",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1)
    )
    
    plt.tight_layout()
    if save_dir is not None:
        base_dir = f"./_misc/{save_dir}"
        os.makedirs(base_dir, exist_ok=True)
        if prefix:
            prefix_dir = f"{base_dir}/{prefix}"
            os.makedirs(prefix_dir, exist_ok=True)
            save_path = f"{prefix_dir}/case_analysis"
        else:
            save_path = f"{base_dir}/case_analysis"
            
        plt.savefig(save_path)
    else:
        plt.show()


def plot_probability_scatter(results, **kwargs):
    raise DeprecationWarning()
    save_dir, prefix = kwargs.get("save_dir", None), kwargs.get("prefix", None)
    model1_probs = [x['model1_prob'] for x in results['prob_analysis']['all_probs']]
    model2_probs = [x['model2_prob'] for x in results['prob_analysis']['all_probs']]
    targets = [x['target'] for x in results['prob_analysis']['all_probs']]
    
    plt.figure(figsize=(8, 8))
    
    # Color points by ground truth
    colors = ['red' if t == 1 else 'blue' for t in targets]
    plt.scatter(model1_probs, model2_probs, c=colors, alpha=0.05)
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)  # Diagonal line
    plt.axhline(0.5, color='gray', linestyle=':', alpha=0.3)
    plt.axvline(0.5, color='gray', linestyle=':', alpha=0.3)
    
    plt.xlabel(f'{results["model1"]} Probability')
    plt.ylabel(f'{results["model2"]} Probability')
    plt.title('Model Probability Comparison')
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Positive Class', 
               markerfacecolor='red', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Negative Class', 
               markerfacecolor='blue', markersize=10)
    ]
    plt.legend(handles=legend_elements)
    
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.tight_layout()
    if save_dir is not None:
        base_dir = f"./_misc/{save_dir}"
        os.makedirs(base_dir, exist_ok=True)
        if prefix:
            prefix_dir = f"{base_dir}/{prefix}"
            os.makedirs(prefix_dir, exist_ok=True)
            save_path = f"{prefix_dir}/prob_scatter"
        else:
            save_path = f"{base_dir}/prob_scatter"
            
        plt.savefig(save_path)
    else:
        plt.show()


def plot_disagreement_analysis(results, **kwargs):
    raise DeprecationWarning()
    save_dir, prefix = kwargs.get("save_dir", None), kwargs.get("prefix", None)
    disagreements = results['prob_analysis']['disagreement_probs']
    if not disagreements:
        print("No disagreements to plot")
        return
    
    model1_disagree_probs = [x['model1_prob'] for x in disagreements]
    model2_disagree_probs = [x['model2_prob'] for x in disagreements]
    targets = [x['target'] for x in disagreements]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = ['red' if t == 1 else 'blue' for t in targets]
    ax1.scatter(model1_disagree_probs, model2_disagree_probs, c=colors, alpha=0.05)
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax1.set_xlabel(f'{results["model1"]} Probability')
    ax1.set_ylabel(f'{results["model2"]} Probability')
    ax1.set_title('Disagreement Cases')
    
    disagree_diffs = [x['prob_diff'] for x in disagreements]
    ax2.hist(disagree_diffs, bins=20, edgecolor='black')
    ax2.set_title('Probability Differences in Disagreements')
    ax2.set_xlabel('Absolute Probability Difference')
    ax2.set_ylabel('Count')
    
    plt.tight_layout()
    if save_dir is not None:
        base_dir = f"./_misc/{save_dir}"
        os.makedirs(base_dir, exist_ok=True)
        if prefix:
            prefix_dir = f"{base_dir}/{prefix}"
            os.makedirs(prefix_dir, exist_ok=True)
            save_path = f"{prefix_dir}/disag_analysis"
        else:
            save_path = f"{base_dir}/disag_analysis"
            
        plt.savefig(save_path)
    else:
        plt.show()

def plot_confidence_comparison(results, **kwargs):
    raise DeprecationWarning()
    save_dir, prefix = kwargs.get("save_dir", None), kwargs.get("prefix", None)
    correct1 = []
    correct2 = []
    wrong1 = []
    wrong2 = []
    
    for entry in results['prob_analysis']['all_probs']:
        t = entry['target']
        pb1 = entry['model1_prob']
        pb2 = entry['model2_prob']
        pd1 = entry['model1_pred']
        pd2 = entry['model2_pred']
        
        conf1 = pb1 if pd1 == 1 else 1 - pb1
        if pd1 == t:
            correct1.append(conf1)
        else:
            wrong1.append(conf1)
            
        conf2 = pb2 if pd2 == 1 else 1 - pb2
        if pd2 == t:
            correct2.append(conf2)
        else:
            wrong2.append(conf2)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.hist([correct1, wrong1], bins=20, label=['Correct', 'Wrong'], stacked=True)
    ax1.set_title(f'{results["model1"]} Confidence')
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Count')
    ax1.legend()

    ax2.hist([correct2, wrong2], bins=20, label=['Correct', 'Wrong'], stacked=True)
    ax2.set_title(f'{results["model2"]} Confidence')
    ax2.set_xlabel('Confidence')
    ax2.legend()
    
    plt.tight_layout()
    if save_dir is not None:
        base_dir = f"./_misc/{save_dir}"
        os.makedirs(base_dir, exist_ok=True)
        if prefix:
            prefix_dir = f"{base_dir}/{prefix}"
            os.makedirs(prefix_dir, exist_ok=True)
            save_path = f"{prefix_dir}/conf_comp"
        else:
            save_path = f"{base_dir}/conf_comp"
            
        plt.savefig(save_path)
    else:
        plt.show()

def plot_all_head2head_results(qgrid: QKernel_GridSearch, _indi1, _indi2, name, save_dir=None, save=False):
    raise DeprecationWarning()
    
    if not save:
        save = input("Save? (y/*) ") == "y"
    if save and save_dir is None:
        folders = os.listdir("./_misc")
        for i, folder in enumerate(folders, start=1):
            print(f"{i}.", folder)
        print()
        save_dir = folders[int(input("")) - 1]

    #indi1 = qgrid.grid[6, 1, 2][0]["branches"][0] # 1792 total features into FC
    #indi2 = qgrid.grid[4, 4, 4][0]["branches"][0] # 1792 total features into FC

    indi1 = qgrid.grid[*_indi1][0]["branches"][0]
    indi2 = qgrid.grid[*_indi2][0]["branches"][0]


    qgrid.epochs = 5
    results = qgrid.head2head(indi1, indi2)
    plot_probability_differences(results, save_dir=save_dir, prefix=name)
    plot_disagreement_probability_differences(results, save_dir=save_dir, prefix=name)
    plot_case_analysis(results, save_dir=save_dir, prefix=name)
    plot_probability_scatter(results, save_dir=save_dir, prefix=name)
    plot_disagreement_analysis(results, save_dir=save_dir, prefix=name)
    plot_confidence_comparison(results, save_dir=save_dir, prefix=name)


if __name__ == "__main__":
    main()
