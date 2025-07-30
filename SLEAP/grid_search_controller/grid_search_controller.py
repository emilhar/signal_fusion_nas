import random
import time

import numpy as np
import torch
import torch.nn as nn
import matplotlib as mpl
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.proj3d as proj3d

from Globals import Signal, Targets, DataManager
from data.data_loader import SDataLoader
from models.cnn_binary_classifier import CNN_BinaryClassifier


class GridSearch:
    def __init__(self, signal: Signal, sleep_stage: Targets, dataset: DataManager.DatasetNames, runtype: RunType, dataset_percentage, epochs, n_samples=3000):
        DataManager.MAX_MEMORY = 2048
        DataManager.DATASET = dataset
        DataManager.dataset_percentage = dataset_percentage

        self.runtype = runtype
        self.signal = signal
        self.sleep_stage = sleep_stage
        self.n_samples = n_samples
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loader = SDataLoader(self.signal, self.sleep_stage)

        self._train_loader, self._test_loader, n_samples, self._pos_weight = self.loader.get_random_subset()

        self.__run_type_validation()


    def load_grid(self):
        try:
            print("Loading grid from data...")
            print()
            self.grid = np.load(f"./data/grids/{self.runtype}/{self.signal}/{self.sleep_stage}.npy", allow_pickle=True)
            print("Loaded grid.")
        except FileNotFoundError as e:
            print(e)
            print(f"No grid found for {self.signal}+{self.sleep_stage} with runtype {self.runtype}")
            self.grid = None

    def save_grid(self):
        path = f"./data/grids/{self.runtype}/{self.signal}/{self.sleep_stage}"
        print(f"Saving grid to {path}")
        np.save(path, self.grid)
        print("Grid has been saved.")

    def __run_type_validation(self):
        if self.runtype == self.RunType.any:
            return

        if self.runtype == self.RunType.k1000_approx:
            if DataManager.dataset_percentage != 0.20:
                raise ValueError(f"{self.RunType.k1000_approx} expects 20% of the dataset.")
            if self.epochs != 1:
                raise ValueError(f"{self.RunType.k1000_approx} expects 1 epoch per model")
            
        if self.runtype == self.RunType.no_k0_1_filter:
            if DataManager.dataset_percentage != 0.20:
                raise ValueError(f"{self.RunType.no_k0_1_filter} expects 20% of the dataset.")
            if self.epochs != 5:
                raise ValueError(f"{self.RunType.no_k0_1_filter} expects 5 epoch per model")
            
        if self.runtype == self.RunType.k1000_full:
            if DataManager.dataset_percentage != 1.00:
                raise ValueError(f"{self.RunType.k1000_full} expects 100% of the dataset")
            if self.epochs != 10:
                raise ValueError(f"{self.RunType.k1000_full} expects 10 epochs per model.")
            
        if self.runtype == self.RunType.no_k0_full:
            if DataManager.dataset_percentage != 1.00:
                raise ValueError(f"{self.RunType.no_k0_full} expects 100% of the dataset")
            if self.epochs != 10:
                raise ValueError(f"{self.RunType.no_k0_full} expects 10 epochs per model.")

        if self.runtype == self.RunType.no_k0_1_filter_full:
            if DataManager.dataset_percentage != 1.00:
                raise ValueError(f"{self.RunType.no_k0_1_filter_full} expects 100% of the dataset")
            if self.epochs != 10:
                raise ValueError(f"{self.RunType.no_k0_1_filter_full} expects 10 epochs per model.")
    
    def __bool__(self):
        return self.grid is not None



class QKernel_GridSearch(GridSearch):
    def __init__(self, signal: Signal, sleep_stage: Targets, dataset: DataManager.DatasetNames, runtype: GridSearch.RunType, dataset_percentage, epochs, n_samples=3000):
        super().__init__(signal, sleep_stage, dataset, runtype, dataset_percentage, epochs, n_samples)
        self.primordial_kernel = 1000 if signal != Signal.EMG.SUBMENTAL else 15
        self.kernels = self.__get_kernels()
        self.load_grid()


    def theta(self) -> list[list[int]]:
        """debugging purposes"""
        opt = [[19, 18], [420, 120, 8], [1000, 1000, 1000], [1, 1, 1], [1000], [900, 500, 500]]


        k = random.choice(opt)
        for i, x in enumerate(k):
            k[i] *= random.uniform(0.5, 1.5)
            k[i] = min(1500, max(1, int(k[i])))

        return k

    
    def compute_grid(self):
        start_time = time.time()
        total_iterations = len(self.kernels) ** 4
        completed = 0
        
        if self.grid is None:
            self.grid = np.empty((len(self.kernels), len(self.kernels), len(self.kernels), len(self.kernels)), dtype=list)
        for x, k1 in enumerate(self.kernels):
            for y, k2 in enumerate(self.kernels):
                for z, k3 in enumerate(self.kernels):
                    for a, k4 in enumerate(self.kernels):
                        if not self.grid[x, y, z, a]:
                            self.grid[x, y, z, a] = []
                        self.grid[x, y, z, a].append(self.new_model(k1, k2, k3, k4))
                        completed += 1
                        
                        elapsed = time.time() - start_time
                        percent = (completed / total_iterations) * 100
                        
                        if completed > 0:
                            iter_time = elapsed / completed
                            remaining = max(0, (total_iterations - completed) * iter_time)
                            eta_sec = int(remaining)
                            eta_str = f"{eta_sec//3600:02d}:{eta_sec%3600//60:02d}:{eta_sec%60:02d}"
                        else:
                            eta_str = "--:--:--"
                        
                        print(f"\rProgress: [{'='*int(percent/5)}{' '*(20-int(percent/5))}] {percent:.1f}% • ETA: {eta_str}", end="\r")
        
        total_time = time.time() - start_time
        print(f"\nCompleted {total_iterations} models in {total_time:.1f} seconds")
    
    
    def new_model(self, k1, k2, k3, k4):
        train_loader, test_loader, n_samples, pos_weight = self.loader.get_random_subset() 

        branch = [k1, k2, k3, k4]

        model_args = get_branch_configs([branch], self.n_samples)
        model_args["batch_size"] = 128
        model = CNN_BinaryClassifier(**model_args).to(self.device)
        
        model_performance = CNN_BinaryClassifier.train(
            model,
            train_loader,
            test_loader,
            pos_weight,
            epochs=self.epochs,
        )

        model_performance = {
            k: v for k, v in model_performance.items()
            if k != "true_labels" and k != "best_scores" and k != "state_dict"
        }

        return model_performance
    
    def __head2head(self, indi1, indi2):
        models = []

        DataManager.DATASET = DataManager.DatasetNames.EDF_78       
        DataManager.dataset_percentage = 1.00
        
        for individual in (indi1, indi2):
            model_args = get_branch_configs([individual], self.n_samples)
            model_args["batch_size"] = 128
            model = CNN_BinaryClassifier(**model_args).to(self.device)
            
            model_performance = CNN_BinaryClassifier.train(
                model,
                self._train_loader,
                self._test_loader,
                self._pos_weight,
                epochs=100,
                output_period=10,
                verbose=True,
                lr=2.5e-4,
                p=101,
            )

            model.load_state_dict(model_performance["recent_state_dict"])
            model.to(self.device)
            models.append(model)
            print()

        model1 = models[0]
        model2 = models[1]

        results = {
            'model1': str(indi1),
            'model2': str(indi2),
            'disagreements': 0,
            'prob_analysis': {
                'all_probs': [],
                'disagreement_probs': [],
                'average_prob_fidd': 0,
                'max_prob_diff': 0,
                'min_prob_diff': 0,
                'prob_deviation_stats': {}
            },
            'examples': {
                'model1_correct_model2_wrong': [],
                'model2_correct_model1_wrong': [],
                'both_wrong': [],
                'both_correct': [],
                'high_disagreement': []
            },
            'metrics': {
                'model1': {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'loss': 0},
                'model2': {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'loss': 0}
            }
        }

        true_positives1 = true_positives2 = 0
        false_positives1 = false_positives2 = 0
        false_negatives1 = false_negatives2 = 0
        correct1 = correct2 = 0
        total = 0
        test_loss1 = test_loss2 = 0
        sum_prob_diff = 0
        max_prob_diff = -float('inf')
        min_prob_diff = float('inf')
        
        criterion = nn.BCEWithLogitsLoss(pos_weight=self._pos_weight.to(self.device))
        
        model1.eval()
        model2.eval()
        
        with torch.inference_mode():
            for batch_idx, (data, target) in enumerate(self._test_loader):
                data, target = data.to(self.device), target.to(self.device).float()
                
                logits1 = model1(data)
                logits2 = model2(data)
                
                test_loss1 += criterion(logits1, target.unsqueeze(1)).item()
                test_loss2 += criterion(logits2, target.unsqueeze(1)).item()
                
                prob1 = torch.sigmoid(logits1)
                prob2 = torch.sigmoid(logits2)
                pred1 = prob1.round()
                pred2 = prob2.round()
                
                for i in range(len(target)):
                    total += 1
                    t = target[i].item()
                    pb1 = prob1[i].item()
                    pb2 = prob2[i].item()
                    pd1 = pred1[i].item()
                    pd2 = pred2[i].item()
                    
                    prob_diff = abs(pb1 - pb2)
                    sum_prob_diff += prob_diff
                    max_prob_diff = max(max_prob_diff, prob_diff)
                    min_prob_diff = min(min_prob_diff, prob_diff)
                    
                    prob_entry = {
                        'target': t,
                        'model1_prob': pb1,
                        'model2_prob': pb2,
                        'model1_pred': pd1,
                        'model2_pred': pd2,
                        'prob_diff': prob_diff
                    }
                    results['prob_analysis']['all_probs'].append(prob_entry)
                    
                    if pd1 != pd2:
                        results['disagreements'] += 1
                        results['prob_analysis']['disagreement_probs'].append(prob_entry)
                        

                        if prob_diff >= 0.25:
                            results['examples']['high_disagreement'].append(prob_entry)
                    
                    if pd1 == t:
                        correct1 += 1
                        if t == 1:
                            true_positives1 += 1
                    else:
                        if pd1 == 1:
                            false_positives1 += 1
                        else:
                            false_negatives1 += 1
                    
                    if pd2 == t:
                        correct2 += 1
                        if t == 1:
                            true_positives2 += 1
                    else:
                        if pd2 == 1:
                            false_positives2 += 1
                        else:
                            false_negatives2 += 1
                    
                    case_entry = {
                        'data': data[i].cpu().numpy(),
                        'target': t,
                        'model1_prob': pb1,
                        'model2_prob': pb2,
                        'model1_pred': pd1,
                        'model2_pred': pd2,
                        'prob_diff': prob_diff
                    }
                    
                    if pd1 == t and pd2 != t:
                        results['examples']['model1_correct_model2_wrong'].append(case_entry)
                    elif pd2 == t and pd1 != t:
                        results['examples']['model2_correct_model1_wrong'].append(case_entry)
                    elif pd1 != t and pd2 != t:
                        results['examples']['both_wrong'].append(case_entry)
                    else:
                        results['examples']['both_correct'].append(case_entry)
        
        test_loss1 /= len(self._test_loader)
        test_loss2 /= len(self._test_loader)
        
        results['prob_analysis']['average_prob_diff'] = sum_prob_diff / total
        results['prob_analysis']['max_prob_diff'] = max_prob_diff
        results['prob_analysis']['min_prob_diff'] = min_prob_diff
        
        all_diffs = [x['prob_diff'] for x in results['prob_analysis']['all_probs']]
        results['prob_analysis']['prob_deviation_stats'] = {
            'mean': np.mean(all_diffs),
            'median': np.median(all_diffs),
            'std': np.std(all_diffs),
            'q1': np.percentile(all_diffs, 25),
            'q3': np.percentile(all_diffs, 75)
        }
        
        precision1 = true_positives1 / (true_positives1 + false_positives1) if (true_positives1 + false_positives1) > 0 else 0
        recall1 = true_positives1 / (true_positives1 + false_negatives1) if (true_positives1 + false_negatives1) > 0 else 0
        f1_1 = 2 * (precision1 * recall1) / (precision1 + recall1) if (precision1 + recall1) > 0 else 0
        
        precision2 = true_positives2 / (true_positives2 + false_positives2) if (true_positives2 + false_positives2) > 0 else 0
        recall2 = true_positives2 / (true_positives2 + false_negatives2) if (true_positives2 + false_negatives2) > 0 else 0
        f1_2 = 2 * (precision2 * recall2) / (precision2 + recall2) if (precision2 + recall2) > 0 else 0
        
        results['metrics']['model1'].update({
            'accuracy': 100. * correct1 / total,
            'precision': precision1,
            'recall': recall1,
            'f1': f1_1,
            'loss': test_loss1
        })
        
        results['metrics']['model2'].update({
            'accuracy': 100. * correct2 / total,
            'precision': precision2,
            'recall': recall2,
            'f1': f1_2,
            'loss': test_loss2
        })
        
        results['disagreement_percentage'] = 100. * results['disagreements'] / total
        
        print(f"\n=== Head-to-Head Comparison ===")
        print(f"Model 1: {results['model1']}")
        print(f"Model 2: {results['model2']}")
        print(f"\nDisagreements: {results['disagreements']} ({results['disagreement_percentage']:.2f}%)")
        
        print("\nProbability Analysis:")
        print(f"Average probablility difference: {results['prob_analysis']['average_prob_diff']:.4f}")
        print(f"Max probability difference: {results['prob_analysis']['max_prob_diff']:.4f}")
        print(f"Min probability difference: {results['prob_analysis']['min_prob_diff']:.4f}")
        print("\nProbability Deviation Statistics:")
        for stat, value in results['prob_analysis']['prob_deviation_stats'].items():
            print(f"{stat}: {value:.4f}")
        
        print("\nPerformance Metrics:")
        print(f"{'Metric':<10} {'Model 1':<10} {'Model 2':<10}")
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'loss']:
            print(f"{metric:<10} {results['metrics']['model1'][metric]:<10.4f} {results['metrics']['model2'][metric]:<10.4f}")
        
        print("\nCase Analysis:")
        print(f"Model 1 correct & Model 2 wrong: {len(results['examples']['model1_correct_model2_wrong'])}")
        print(f"Model 2 correct & Model 1 wrong: {len(results['examples']['model2_correct_model1_wrong'])}")
        print(f"Both wrong: {len(results['examples']['both_wrong'])}")
        print(f"Both correct: {len(results['examples']['both_correct'])}")
        print(f"High disagreement cases: {len(results['examples']['high_disagreement'])}")
        
        return results

        
            

    def plot_qkernel_3d(self, metric='train_loss', figsize=(14, 10), 
                        color_map='viridis', marker_size=50, alpha=0.7,
                        elev=25, azim=45):
        if self.grid is None:
            raise ValueError("Grid is empty. Compute the grid first.")
        
        # Prepare data
        xs, ys, zs, cs, labels = [], [], [], [], []
        display_kernels = sorted(self.kernels)
        
        # Create logarithmic positions for equal spacing
        log_kernels = np.log10(display_kernels)
        pos_mapping = {k: i for i, k in enumerate(log_kernels)}
        
        for x, k1 in enumerate(self.kernels):
            for y, k2 in enumerate(self.kernels):
                for z, k3 in enumerate(self.kernels):
                    results = self.grid[x, y, z]
                    if not results or not isinstance(results, list):
                        continue
                    
                    values = []
                    for res in results:
                        if isinstance(res, dict) and metric in res:
                            values.append(res[metric])
                    
                    if not values:
                        continue
                        
                    avg_value = np.mean(values)
                    # Use mapped positions for equal spacing
                    xs.append(pos_mapping[np.log10(k1)])
                    ys.append(pos_mapping[np.log10(k2)])
                    zs.append(pos_mapping[np.log10(k3)])
                    cs.append(avg_value)
                    labels.append(
                        f"k1: {k1}\nk2: {k2}\nk3: {k3}\n"
                        f"{metric}: {avg_value:.4f}\n"
                        f"Runs: {len(values)}"
                    )
        
        if not xs:
            raise ValueError(f"No valid data found for metric '{metric}'")
        
        fig = plt.figure(figsize=figsize, dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=elev, azim=azim)
        
        scatter = ax.scatter3D(
            xs, ys, zs, 
            c=cs, 
            cmap=color_map,
            s=marker_size,
            alpha=alpha,
            edgecolor='k',
            linewidth=0.5,
            picker=True
        )
        
        # Configure axes with original kernel labels
        ax.set_xlabel('k1', labelpad=15, fontsize=12)
        ax.set_ylabel('k2', labelpad=15, fontsize=12)
        ax.set_zlabel('k3', labelpad=15, fontsize=12)
        
        # Set ticks at mapped positions with original labels
        tick_positions = range(len(display_kernels))
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_zticks(tick_positions)
        
        ax.set_xticklabels(display_kernels, rotation=45)
        ax.set_yticklabels(display_kernels, rotation=45)
        ax.set_zticklabels(display_kernels, rotation=45)
        
        # Adjust plot limits for better spacing
        ax.set_xlim(-0.5, len(display_kernels)-0.5)
        ax.set_ylim(-0.5, len(display_kernels)-0.5)
        ax.set_zlim(-0.5, len(display_kernels)-0.5)
        
        cbar = fig.colorbar(scatter, pad=0.1)
        cbar.set_label(metric.upper(), rotation=270, labelpad=25, fontsize=12)
        
        ax.set_title(
            f'QKernel Performance: {metric.upper()}\n'
            f'Signal: {self.signal}, Stage: {self.sleep_stage}\n'
            'Logarithmically spaced kernel sizes',
            fontsize=14,
            pad=20
        )
        
        # Annotation handling (same as before)
        annot = ax.annotate(
            "", xy=(0, 0), xytext=(20, 20),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w", alpha=0.9),
            arrowprops=dict(arrowstyle="->")
        )
        annot.set_visible(False)
        
        def update_annot(ind):
            x, y, z = scatter._offsets3d
            pos = (x[ind["ind"][0]], y[ind["ind"][0]], z[ind["ind"][0]])
            x2, y2, _ = proj3d.proj_transform(pos[0], pos[1], pos[2], ax.get_proj())
            annot.xy = (x2, y2)
            annot.set_text(labels[ind["ind"][0]])
            annot.get_bbox_patch().set_alpha(0.9)
        
        def hover(event):
            vis = annot.get_visible()
            if event.inaxes == ax:
                cont, ind = scatter.contains(event)
                if cont:
                    update_annot(ind)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    if vis:
                        annot.set_visible(False)
                        fig.canvas.draw_idle()
        
        fig.canvas.mpl_connect("motion_notify_event", hover)
        
        plt.tight_layout()
        plt.show()


    def plot_qkernel_3d_vstime(self, metric='train_loss', time_metric='time', 
                          figsize=(14, 10), color_map='viridis', marker_size=50, 
                          alpha=0.7, elev=25, azim=45, size_by_time=True,
                          time_normalization='max'):
        """
        3D plot comparing any metric against time, with optional size scaling by time.
        
        Parameters:
        -----------
        metric : str
            Primary metric to color the markers ('train_loss', 'test_loss', etc.)
        time_metric : str
            Time metric to compare against ('time' or other time-related metric)
        figsize : tuple
            Figure size
        color_map : str
            Colormap for the primary metric
        marker_size : int
            Base size of the markers in the scatter plot
        alpha : float
            Transparency of the markers
        elev : float
            Elevation angle for 3D view
        azim : float
            Azimuth angle for 3D view
        size_by_time : bool
            Whether to scale marker sizes by time (smaller = faster)
        time_normalization : str
            How to normalize time for sizing ('max', 'min', or 'mean')
        """
        if self.grid is None:
            raise ValueError("Grid is empty. Compute the grid first.")
        
        # Prepare data
        xs, ys, zs, cs, ts, sizes, labels = [], [], [], [], [], [], []
        display_kernels = sorted(self.kernels)
        
        # Create logarithmic positions for equal spacing
        log_kernels = np.log10(display_kernels)
        pos_mapping = {k: i for i, k in enumerate(log_kernels)}
        
        # First pass to collect all time values for normalization
        all_times = []
        for x, k1 in enumerate(self.kernels):
            for y, k2 in enumerate(self.kernels):
                for z, k3 in enumerate(self.kernels):
                    results = self.grid[x, y, z]
                    if not results or not isinstance(results, list):
                        continue
                    
                    time_values = []
                    for res in results:
                        if isinstance(res, dict) and time_metric in res:
                            time_values.append(res[time_metric])
                    
                    if time_values:
                        all_times.extend(time_values)
        
        if not all_times:
            raise ValueError(f"No valid data found for time metric '{time_metric}'")
        
        # Calculate normalization factor for time
        if time_normalization == 'max':
            time_norm = max(all_times)
        elif time_normalization == 'min':
            time_norm = min(all_times)
        else:  # 'mean'
            time_norm = np.mean(all_times)
        
        # Second pass to collect data for plotting
        for x, k1 in enumerate(self.kernels):
            for y, k2 in enumerate(self.kernels):
                for z, k3 in enumerate(self.kernels):
                    results = self.grid[x, y, z]
                    if not results or not isinstance(results, list):
                        continue
                    
                    metric_values = []
                    time_values = []
                    for res in results:
                        if isinstance(res, dict):
                            if metric in res:
                                metric_values.append(res[metric])
                            if time_metric in res:
                                time_values.append(res[time_metric])
                    
                    if not metric_values or not time_values:
                        continue
                    
                    avg_metric = np.mean(metric_values)
                    avg_time = np.mean(time_values)
                    
                    # Use mapped positions for equal spacing
                    xs.append(pos_mapping[np.log10(k1)])
                    ys.append(pos_mapping[np.log10(k2)])
                    zs.append(pos_mapping[np.log10(k3)])
                    cs.append(avg_metric)
                    ts.append(avg_time)
                    
                    # Calculate marker size based on time (smaller time = larger marker)
                    if size_by_time:
                        size_factor = (time_norm / avg_time)# ** 0.5  # Square root for better visual scaling
                        sizes.append(marker_size * size_factor)
                    else:
                        sizes.append(marker_size)
                    
                    labels.append(
                        f"k1: {k1}\nk2: {k2}\nk3: {k3}\n"
                        f"{metric}: {avg_metric:.4f}\n"
                        f"{time_metric}: {avg_time:.2f}s\n"
                        f"Runs: {len(metric_values)}"
                    )
        
        if not xs:
            raise ValueError(f"No valid data found for metrics '{metric}' and '{time_metric}'")
        
        fig = plt.figure(figsize=figsize, dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=elev, azim=azim)
        
        scatter = ax.scatter3D(
            xs, ys, zs, 
            c=cs, 
            cmap=color_map,
            norm=mpl.colors.LogNorm(vmin=min(cs), vmax=max(cs)),
            s=sizes,
            alpha=alpha,
            edgecolor='k',
            linewidth=0.5,
            picker=True
        )
        
        # Configure axes with original kernel labels
        ax.set_xlabel('k1', labelpad=15, fontsize=12)
        ax.set_ylabel('k2', labelpad=15, fontsize=12)
        ax.set_zlabel('k3', labelpad=15, fontsize=12)
        
        # Set ticks at mapped positions with original labels
        tick_positions = range(len(display_kernels))
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_zticks(tick_positions)
        
        ax.set_xticklabels(display_kernels, rotation=45)
        ax.set_yticklabels(display_kernels, rotation=45)
        ax.set_zticklabels(display_kernels, rotation=45)
        
        # Adjust plot limits for better spacing
        ax.set_xlim(-0.5, len(display_kernels)-0.5)
        ax.set_ylim(-0.5, len(display_kernels)-0.5)
        ax.set_zlim(-0.5, len(display_kernels)-0.5)
        
        # Create colorbar for the primary metric
        cbar = fig.colorbar(scatter, pad=0.1)
        cbar.set_label(metric.upper(), rotation=270, labelpad=25, fontsize=12)
        
        # Add time information to title
        title = (f'QKernel Performance: {metric.upper()} vs {time_metric.upper()}\n'
                f'Signal: {self.signal}, Stage: {self.sleep_stage}\n'
                'Logarithmically spaced kernel sizes')
        
        if size_by_time:
            title += f"\nMarker size ∝ 1/{time_metric} (normalized by {time_normalization})"
        
        ax.set_title(title, fontsize=14, pad=20)
        
        # Annotation handling
        annot = ax.annotate(
            "", xy=(0, 0), xytext=(20, 20),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w", alpha=0.9),
            arrowprops=dict(arrowstyle="->")
        )
        annot.set_visible(False)
        
        def update_annot(ind):
            x, y, z = scatter._offsets3d
            pos = (x[ind["ind"][0]], y[ind["ind"][0]], z[ind["ind"][0]])
            x2, y2, _ = proj3d.proj_transform(pos[0], pos[1], pos[2], ax.get_proj())
            annot.xy = (x2, y2)
            annot.set_text(labels[ind["ind"][0]])
            annot.get_bbox_patch().set_alpha(0.9)
        
        def hover(event):
            vis = annot.get_visible()
            if event.inaxes == ax:
                cont, ind = scatter.contains(event)
                if cont:
                    update_annot(ind)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    if vis:
                        annot.set_visible(False)
                        fig.canvas.draw_idle()
        
        fig.canvas.mpl_connect("motion_notify_event", hover)
        
        #plt.tight_layout()
        plt.show()


    def plot_qkernel_slice(self, fixed_kernel='k1', fixed_value=None, metric='train_loss', 
                      figsize=(12, 8), color_map='viridis', marker_size=50, alpha=0.7):
        """
        Plot a 2D slice of the 3D grid by fixing one kernel dimension.
        
        Parameters:
        -----------
        fixed_kernel : str
            Which kernel to fix ('k1', 'k2', or 'k3')
        fixed_value : float or None
            Value of the fixed kernel. If None, uses the middle value from available kernels
        metric : str
            Which metric to plot ('train_loss', 'test_loss', etc.)
        figsize : tuple
            Figure size
        color_map : str
            Colormap for the plot
        marker_size : int
            Size of the markers in the scatter plot
        alpha : float
            Transparency of the markers
        """
        if self.grid is None:
            raise ValueError("Grid is empty. Compute the grid first.")
        
        # Validate fixed_kernel parameter
        if fixed_kernel not in ['k1', 'k2', 'k3']:
            raise ValueError("fixed_kernel must be 'k1', 'k2', or 'k3'")
        
        # Prepare data
        display_kernels = sorted(self.kernels)
        
        # If fixed_value is not specified, use the middle kernel
        if fixed_value is None:
            fixed_value = display_kernels[len(display_kernels)//2]
        elif fixed_value not in display_kernels:
            # Find the closest available kernel
            fixed_value = min(display_kernels, key=lambda x: abs(x - fixed_value))
            print(f"Warning: Using closest available kernel value: {fixed_value}")
        
        # Get the index of the fixed kernel
        fixed_idx = display_kernels.index(fixed_value)
        
        # Prepare data for plotting
        xs, ys, cs, labels = [], [], [], []
        
        for i, k1 in enumerate(display_kernels):
            for j, k2 in enumerate(display_kernels):
                for k, k3 in enumerate(display_kernels):
                    # Skip points that don't match our fixed kernel condition
                    if fixed_kernel == 'k1' and i != fixed_idx:
                        continue
                    if fixed_kernel == 'k2' and j != fixed_idx:
                        continue
                    if fixed_kernel == 'k3' and k != fixed_idx:
                        continue
                    
                    results = self.grid[i, j, k]
                    if not results or not isinstance(results, list):
                        continue
                    
                    values = []
                    for res in results:
                        if isinstance(res, dict) and metric in res:
                            values.append(res[metric])
                    
                    if not values:
                        continue
                    
                    avg_value = np.mean(values)
                    
                    # Determine which axes to plot based on fixed kernel
                    if fixed_kernel == 'k1':
                        xs.append(j)  # k2 on x-axis
                        ys.append(k)  # k3 on y-axis
                    elif fixed_kernel == 'k2':
                        xs.append(i)  # k1 on x-axis
                        ys.append(k)  # k3 on y-axis
                    else:  # fixed_kernel == 'k3'
                        xs.append(i)  # k1 on x-axis
                        ys.append(j)  # k2 on y-axis
                    
                    cs.append(avg_value)
                    labels.append(
                        f"k1: {k1}\nk2: {k2}\nk3: {k3}\n"
                        f"{metric}: {avg_value:.4f}\n"
                        f"Runs: {len(values)}"
                    )
        
        if not xs:
            raise ValueError(f"No valid data found for metric '{metric}'")
        
        fig, ax = plt.subplots(figsize=figsize, dpi=100)
        
        scatter = ax.scatter(
            xs, ys, 
            c=cs, 
            cmap=color_map,
            s=marker_size,
            alpha=alpha,
            edgecolor='k',
            linewidth=0.5,
            picker=True
        )
        
        # Set axis labels based on which kernel is fixed
        if fixed_kernel == 'k1':
            ax.set_xlabel('k2', fontsize=12)
            ax.set_ylabel('k3', fontsize=12)
            title_kernel = f'k1 = {fixed_value}'
        elif fixed_kernel == 'k2':
            ax.set_xlabel('k1', fontsize=12)
            ax.set_ylabel('k3', fontsize=12)
            title_kernel = f'k2 = {fixed_value}'
        else:
            ax.set_xlabel('k1', fontsize=12)
            ax.set_ylabel('k2', fontsize=12)
            title_kernel = f'k3 = {fixed_value}'
        
        # Set ticks and labels
        tick_positions = range(len(display_kernels))
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_xticklabels(display_kernels, rotation=45)
        ax.set_yticklabels(display_kernels, rotation=45)
        
        # Adjust plot limits
        ax.set_xlim(-0.5, len(display_kernels)-0.5)
        ax.set_ylim(-0.5, len(display_kernels)-0.5)
        
        cbar = fig.colorbar(scatter)
        cbar.set_label(metric.upper(), rotation=270, labelpad=25, fontsize=12)
        
        ax.set_title(
            f'QKernel Performance: {metric.upper()}\n'
            f'Fixed {title_kernel}, Signal: {self.signal}, Stage: {self.sleep_stage}',
            fontsize=14,
            pad=20
        )
        
        # Annotation handling
        annot = ax.annotate(
            "", xy=(0, 0), xytext=(20, 20),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w", alpha=0.9),
            arrowprops=dict(arrowstyle="->")
        )
        annot.set_visible(False)
        
        def update_annot(ind):
            pos = scatter.get_offsets()[ind["ind"][0]]
            annot.xy = pos
            annot.set_text(labels[ind["ind"][0]])
            annot.get_bbox_patch().set_alpha(0.9)
        
        def hover(event):
            vis = annot.get_visible()
            if event.inaxes == ax:
                cont, ind = scatter.contains(event)
                if cont:
                    update_annot(ind)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    if vis:
                        annot.set_visible(False)
                        fig.canvas.draw_idle()
        
        fig.canvas.mpl_connect("motion_notify_event", hover)
        
        plt.tight_layout()
        plt.show()


    def plot_qkernel_slice_vstime(self, fixed_kernel='k1', fixed_value=None, 
                                metric='train_loss', time_metric='time',
                                figsize=(12, 8), metric_cmap='viridis', 
                                time_cmap='plasma', marker_size=100, alpha=0.8,
                                time_normalization='max', grid_steps=20):
        """
        2D slice plot with background colored by time and points colored by primary metric.
        
        Parameters:
        -----------
        fixed_kernel : str
            Which kernel to fix ('k1', 'k2', or 'k3')
        fixed_value : float or None
            Value of the fixed kernel. If None, uses the middle value from available kernels
        metric : str
            Primary metric to color the markers ('train_loss', 'test_loss', etc.)
        time_metric : str
            Time metric for background coloring ('time' or other time-related metric)
        figsize : tuple
            Figure size
        metric_cmap : str
            Colormap for the primary metric points
        time_cmap : str
            Colormap for the background time values
        marker_size : int
            Size of the markers in the scatter plot
        alpha : float
            Transparency of the markers
        time_normalization : str
            How to normalize time for coloring ('max', 'min', or 'mean')
        grid_steps : int
            Number of steps for background grid interpolation
        """
        if self.grid is None:
            raise ValueError("Grid is empty. Compute the grid first.")
        
        # Validate fixed_kernel parameter
        if fixed_kernel not in ['k1', 'k2', 'k3']:
            raise ValueError("fixed_kernel must be 'k1', 'k2', or 'k3'")
        
        # Prepare data
        display_kernels = sorted(self.kernels)
        
        # If fixed_value is not specified, use the middle kernel
        if fixed_value is None:
            fixed_value = display_kernels[len(display_kernels)//2]
        elif fixed_value not in display_kernels:
            # Find the closest available kernel
            fixed_value = min(display_kernels, key=lambda x: abs(x - fixed_value))
            print(f"Warning: Using closest available kernel value: {fixed_value}")
        
        # Get the index of the fixed kernel
        fixed_idx = display_kernels.index(fixed_value)
        
        # Collect all data points for interpolation
        x_coords, y_coords, metric_values, time_values = [], [], [], []
        
        for i, k1 in enumerate(display_kernels):
            for j, k2 in enumerate(display_kernels):
                for k, k3 in enumerate(display_kernels):
                    # Skip points that don't match our fixed kernel condition
                    if fixed_kernel == 'k1' and i != fixed_idx:
                        continue
                    if fixed_kernel == 'k2' and j != fixed_idx:
                        continue
                    if fixed_kernel == 'k3' and k != fixed_idx:
                        continue
                    
                    results = self.grid[i, j, k]
                    if not results or not isinstance(results, list):
                        continue
                    
                    m_values, t_values = [], []
                    for res in results:
                        if isinstance(res, dict):
                            if metric in res:
                                m_values.append(res[metric])
                            if time_metric in res:
                                t_values.append(res[time_metric])
                    
                    if not m_values or not t_values:
                        continue
                    
                    # Determine which axes to plot based on fixed kernel
                    if fixed_kernel == 'k1':
                        x_coords.append(j)  # k2 on x-axis
                        y_coords.append(k)  # k3 on y-axis
                    elif fixed_kernel == 'k2':
                        x_coords.append(i)  # k1 on x-axis
                        y_coords.append(k)  # k3 on y-axis
                    else:  # fixed_kernel == 'k3'
                        x_coords.append(i)  # k1 on x-axis
                        y_coords.append(j)  # k2 on y-axis
                    
                    metric_values.append(np.mean(m_values))
                    time_values.append(np.mean(t_values))
        
        if not x_coords:
            raise ValueError(f"No valid data found for metrics '{metric}' and '{time_metric}'")
        
        # Convert to numpy arrays for processing
        x_coords = np.array(x_coords)
        y_coords = np.array(y_coords)
        metric_values = np.array(metric_values)
        time_values = np.array(time_values)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize, dpi=100)
        
        # Create grid for background coloring
        xi = np.linspace(-0.5, len(display_kernels)-0.5, grid_steps)
        yi = np.linspace(-0.5, len(display_kernels)-0.5, grid_steps)
        xi, yi = np.meshgrid(xi, yi)
        
        # Interpolate time values for background
        from scipy.interpolate import griddata
        zi = griddata((x_coords, y_coords), time_values, (xi, yi), method='cubic')
        
        # Normalize time values for coloring
        if time_normalization == 'max':
            norm_time = zi / np.max(time_values)
        elif time_normalization == 'min':
            norm_time = (zi - np.min(time_values)) / (np.max(time_values) - np.min(time_values))
        else:  # 'mean'
            norm_time = zi / np.mean(time_values)
        
        # Plot background time colors
        background = ax.imshow(
            norm_time, 
            extent=[-0.5, len(display_kernels)-0.5, -0.5, len(display_kernels)-0.5],
            origin='lower', 
            cmap=time_cmap, 
            alpha=0.3,
            aspect='auto'
        )
        
        # Add colorbar for time background
        time_cbar = fig.colorbar(background, ax=ax, pad=0.15)
        time_cbar.set_label(f"Normalized {time_metric.upper()}", rotation=270, labelpad=25, fontsize=10)
        
        # Plot the actual data points
        scatter = ax.scatter(
            x_coords, y_coords, 
            c=metric_values, 
            cmap=metric_cmap,
            s=marker_size,
            alpha=alpha,
            edgecolor='k',
            linewidth=0.5,
            picker=True
        )
        
        # Add colorbar for metric points
        metric_cbar = fig.colorbar(scatter, ax=ax, pad=0.01)
        metric_cbar.set_label(metric.upper(), rotation=270, labelpad=25, fontsize=10)
        
        # Set axis labels based on which kernel is fixed
        if fixed_kernel == 'k1':
            ax.set_xlabel('k2', fontsize=12)
            ax.set_ylabel('k3', fontsize=12)
            title_kernel = f'k1 = {fixed_value}'
        elif fixed_kernel == 'k2':
            ax.set_xlabel('k1', fontsize=12)
            ax.set_ylabel('k3', fontsize=12)
            title_kernel = f'k2 = {fixed_value}'
        else:
            ax.set_xlabel('k1', fontsize=12)
            ax.set_ylabel('k2', fontsize=12)
            title_kernel = f'k3 = {fixed_value}'
        
        # Set ticks and labels
        tick_positions = range(len(display_kernels))
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_xticklabels(display_kernels, rotation=45)
        ax.set_yticklabels(display_kernels, rotation=45)
        
        # Create labels for hover tooltips
        labels = []
        for i in range(len(x_coords)):
            if fixed_kernel == 'k1':
                k1 = display_kernels[fixed_idx]
                k2 = display_kernels[int(x_coords[i])]
                k3 = display_kernels[int(y_coords[i])]
            elif fixed_kernel == 'k2':
                k1 = display_kernels[int(x_coords[i])]
                k2 = display_kernels[fixed_idx]
                k3 = display_kernels[int(y_coords[i])]
            else:
                k1 = display_kernels[int(x_coords[i])]
                k2 = display_kernels[int(y_coords[i])]
                k3 = display_kernels[fixed_idx]
            
            labels.append(
                f"k1: {k1}\nk2: {k2}\nk3: {k3}\n"
                f"{metric}: {metric_values[i]:.4f}\n"
                f"{time_metric}: {time_values[i]:.2f}s"
            )
        
        # Set title
        ax.set_title(
            f'QKernel Performance: {metric.upper()} (points) vs {time_metric.upper()} (background)\n'
            f'Fixed {title_kernel}, Signal: {self.signal}, Stage: {self.sleep_stage}\n'
            f'Time normalized by {time_normalization}',
            fontsize=12,
            pad=20
        )
        
        # Annotation handling
        annot = ax.annotate(
            "", xy=(0, 0), xytext=(-20, 20),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w", alpha=0.9),
            arrowprops=dict(arrowstyle="->")
        )
        annot.set_visible(False)
        
        def update_annot(ind):
            pos = scatter.get_offsets()[ind["ind"][0]]
            annot.xy = pos
            annot.set_text(labels[ind["ind"][0]])
            annot.get_bbox_patch().set_alpha(0.9)
        
        def hover(event):
            vis = annot.get_visible()
            if event.inaxes == ax:
                cont, ind = scatter.contains(event)
                if cont:
                    update_annot(ind)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    if vis:
                        annot.set_visible(False)
                        fig.canvas.draw_idle()
        
        fig.canvas.mpl_connect("motion_notify_event", hover)
        
        plt.tight_layout()
        plt.show()


    def indidual_box_plot():
        ...

    def __get_kernels(self) -> list[int]:
        if self.signal == Signal.EMG.SUBMENTAL:
            k = [1, 2, 4, 5, 6, 8, 10]
        else:
            k = [5, 50, 500] # TODO: finicky stuff

        return k

