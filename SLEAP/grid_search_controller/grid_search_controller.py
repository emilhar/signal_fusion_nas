import time
import math

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.proj3d as proj3d

from Globals import DataManager
from datahelpers.signal import Signal
from datahelpers.target import Target
from datahelpers.data import Data
from dataloaders.data_loader import SDataLoader
from utils.trained_model_maker import TrainedModelMaker


class GridSearchTimeoutError(Exception):
    def __init__(self, *args):
        super().__init__(*args)


class GridSearchController:
    def __init__(self, signal: Signal, target: Target, dataset_percentage, epochs):
        self.signal = signal
        self.target = target
        self.n_samples = signal.n_samples
        self.grid = None
        self.__epochs = epochs
        self.__kernels = self.__get_kernels()
        self.__loader = SDataLoader(self.signal, self.target, 32)

        self.__train_loader, self.__test_loader, _, self.__pos_weight = self.__loader.get_random_subset()

        print(self.__kernels)
        DataManager.dataset_percentage = dataset_percentage
    
    def __bool__(self):
        return self.grid is not None


    def __get_kernels(self) -> list[int]:
        def next_prime(num):
            num = math.ceil(num)
            if num <= 1:
                return 2
            
            while True:
                is_prime = True
                for i in range(2, int(math.sqrt(num)) + 1):
                    if num % i == 0:
                        is_prime = False
                        break
                if is_prime:
                    return num
                num += 1
    
        divisors = [5, 10, 30, 90, 270, 810, 2430]
        kernels = []
        for div in divisors:
            k = max(next_prime(self.signal.n_samples // div), 3)
            if k not in kernels:
                kernels.append(max(next_prime(self.signal.n_samples // div), 3))
        
        return kernels
    

    def load_grid(self) -> bool:
        try:
            self.grid = np.load(
                f"./data/grids/{self.signal.name}/{self.target.name}.npy",
                allow_pickle=True
            )
            return True
        
        except FileNotFoundError:
            self.grid = None
            return False
        

    def save_grid(self):
        print("WARNING: SAVE KERNEL IS UNDER CONSTRUCTION! :)")
        raise NotImplementedError("WARNING: SAVE KERNEL IS UNDER CONSTRUCTION! :)")
        return
        path = f"./data/grids/{self.runtype}/{self.signal}/{self.sleep_stage}"
        print(f"Saving grid to {path}")
        np.save(path, self.grid)
        print("Grid has been saved.")

    
    def compute_grid(self):
        start_time = time.time()
        total_iterations = len(self.__kernels) ** 3
        completed = 0

        self.load_grid()
        
        if self.grid is None:
            self.grid = np.empty((len(self.__kernels), len(self.__kernels), len(self.__kernels)), dtype=list)
        for x, k1 in enumerate(self.__kernels):
            for y, k2 in enumerate(self.__kernels):
                for z, k3 in enumerate(self.__kernels):
                    if not self.grid[x, y, z]:
                        self.grid[x, y, z] = []
                    self.grid[x, y, z].append(self.__new_model(k1, k2, k3))
                    completed += 1
                    
                    elapsed = time.time() - start_time
                    percent = (completed / total_iterations) * 100
                    
                    iter_time = elapsed / completed
                    remaining = max(0, (total_iterations - completed) * iter_time)
                    eta_sec = int(remaining)
                    if eta_sec // 3600 > 1 and completed == 1:
                        raise GridSearchTimeoutError(f"Estimated grid search completion time is too long: {eta_sec//3600} hours")
                        

                    eta_str = f"{eta_sec//3600:02d}:{eta_sec%3600//60:02d}:{eta_sec%60:02d}"

                    
                    print(f"\rProgress: [{'='*int(percent/5)}{' '*(20-int(percent/5))}] {percent:.1f}% • ETA: {eta_str}", end="\r")
        
        total_time = time.time() - start_time
        print(f"\nCompleted {total_iterations} models in {total_time:.1f} seconds")


    def grid_to_list(self):
        l = []
        if not self.grid:
            raise ValueError(f"Grid for {self.signal.name} predicting {self.target.name} does not exist.")
        
        # TODO: Selection criteria
        for x in range(self.__kernels):
            for y in range(self.__kernels):
                for z in range(self.__kernels):
                    l.append(self.grid[x, y, z][0])

        return l

    
    def __new_model(self, k1, k2, k3):
        #train_loader, test_loader, n_samples, pos_weight = self.loader.get_random_subset() 

        branch = [k1, k2, k3]

        model_performance = TrainedModelMaker(
            [branch], 
            self.signal.n_samples, 
            self.__pos_weight,
            self.__train_loader,
            self.__test_loader,
            self.__epochs,
            32,
        )

        model_performance = {
            k: v for k, v in model_performance.items()
            if k != "true_labels" and k != "best_scores" and k != "state_dict"
        }

        return model_performance
    

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


