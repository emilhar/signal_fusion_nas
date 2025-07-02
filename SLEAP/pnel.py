import os
import random
from datetime import datetime

import torch
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.cluster import KMeans
from scipy.spatial import distance

from Globals import LoggingSettings
from ModelController.ModelMaker import CNN_BinaryClassifier
from ModelController._Trainer import train_model
from ModelController.BranchSettings import get_branch_configs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    if device.type == "cpu":
        raise ValueError("Device is CPU, not GPU")
    
    if input("Plot only? (y/n) ").lower().endswith("y"):
        plot()
        return
    
    # signal = input("Signal: ")
    # if signal not in ["Fpz-Cz", "Pz-Oz", "EMG submental", "EOG horizontal"]:
    #     raise ValueError("Not valid signal")

    # sleep_stage = input("Sleep stage: ")
    # if sleep_stage not in ["W", "N1", "N2", "N3", "REM"]:
    #     raise ValueError("Not valid sleep stage")

    if r := input("Random branches? (y/n) ").lower().endswith("y"):
        branches = get_branches(gen_random=True, size=24, seed=42)
        branches.append([[300, 50, 10, 10], [100, 20, 20, 10], [20, 4, 4, 4]]) # Four convolutional behemoth
    else:
        branches = get_branches()

    for epochs in [1]:
        for signal in ["Fpz-Cz", "Pz-Oz", "EMG_submental", "EOG_horizontal"]:
            for sleep_stage in ["W", "N1", "N2", "N3", "REM"]:
                PnEL, FtF1, FtEL = penelope(branches, epochs, signal, sleep_stage)

                title = f"Fully-trained F1 vs. Partial {epochs} Epoch Loss (train)"
                PnEL_FtMetric(
                    PnEL, FtF1,
                    "FtF1",
                    title,
                    [str(x) for x in branches], 
                    epochs, 
                    f"{"random_" if r else ""}{epochs}_{sleep_stage}_{signal}_f1", 
                    save=True,
                    n_labels=20,
                    reverse=True
                )
                title = f"Fully-trained Loss vs. Partial {epochs} Epoch Loss (train)"
                PnEL_FtMetric(
                    PnEL, FtEL,
                    "FtEL",
                    title,
                    [str(x) for x in branches], 
                    epochs, 
                    f"{"random_" if r else ""}{epochs}_{sleep_stage}_{signal}_test_loss", 
                    save=True,
                    n_labels=20,
                    reverse=False
                )
    

def penelope(branches: list[list[list[int]]], epochs: int, signal: str, sleep_stage: str) -> tuple[list[float], list[float]]:
    PnEL = []
    FtF1 = []
    FtEL = []

    temp = []
    if not branches:
        for log_id in LoggingSettings.LOG_IDS:
            with open(f"Logs/{log_id}_fully_trained_models.csv") as f:
                df = pd.read_csv(f)
                filtered_df = df[(df["signal"] == signal) & (df["sleep_stage"] == sleep_stage)]
                for b in filtered_df["name"]:
                    temp.append(eval(b))

        branches = temp

    for branch in branches:
        f1, train_loss, _ = fully_train(branch, signal, sleep_stage, 5e-5)
        FtF1.append(f1)
        FtEL.append(train_loss)
        PnEL.append(partial_train(branch, epochs, signal, sleep_stage, 5e-4))

    return PnEL, FtF1, FtEL



def fully_train(branch: list[list[int]], signal: str, sleep_stage: str, lr: float) -> float:
    title = f"Training Full model with kernels {branch}"
    print(f"\n{"="*len(title)}")
    print(title)
    print(f"{"="*len(title)}\n")
    for log_id in LoggingSettings.LOG_IDS:
        with open(f"./Logs/{log_id}_fully_trained_models.csv") as f:
            df = pd.read_csv(f)
            mask = (df["name"] == str(branch)) & (df["signal"] == signal) & (df["sleep_stage"] == sleep_stage)
            if mask.any():
                print(f"\nModel {branch} already in {LoggingSettings.LOGGER_ID}_fully_train_models.csv, skipping...")
                f1 = df.loc[mask, "F1"].values[0]
                train_loss = df.loc[mask, "loss"].values[0]
                return f1, train_loss, 0.0
    
    with open(f"./Logs/{LoggingSettings.LOGGER_ID}_fully_trained_models.csv") as f:

        model_args = get_branch_configs(branch, "", 3000)
        model = CNN_BinaryClassifier(**model_args).to(device)

        train_loader, test_loader, pos_weight = load_data(128, signal, sleep_stage)

        start_time = datetime.now()
        res = train_model(
            model, 
            device, 
            train_loader, 
            test_loader, 
            pos_weight, 
            lr=lr, 
            wd=0.0001, 
            p=5, 
            f=0.5, 
            epochs=40, 
            output_period=5, 
            verbose=True,
            have_time_limit=False # Time limit switch
        )
        end_time = datetime.now()
        diff = end_time - start_time

        new_row = pd.DataFrame([{
            "name": str(branch), 
            "F1": res["Best F1"],
            "loss": res["Train Loss"],
            "signal": signal,
            "sleep_stage": sleep_stage,
            "train_time": diff
        }])
        df = pd.concat([df, new_row], ignore_index=True)
        new_row.to_csv(f"./Logs/{LoggingSettings.LOGGER_ID}_fully_trained_models.csv", mode="a", header=False, index=False)
    
    return res["Best F1"], res["Train Loss"], res["Test Loss"]



def partial_train(branch: list[int], epochs: int, signal: str, sleep_stage: str, lr: float) -> float:
    model_args = get_branch_configs(branch, "", 3000)
    model = CNN_BinaryClassifier(**model_args).to(device)

    train_loader, test_loader, pos_weight = load_data(32, signal, sleep_stage, 0.1)

    title = f"Training Partial model with kernels {branch}"
    print(f"\n{"-"*len(title)}")
    print(title)
    print(f"{"-"*len(title)}\n")
    res = train_model(
        model, 
        device, 
        train_loader, 
        test_loader, 
        pos_weight, 
        lr=lr, 
        wd=0.0001, 
        p=5, 
        f=0.5, 
        epochs=epochs, 
        output_period=1, 
        verbose=True,
        have_time_limit=False # Time limit switch
    )

    return res["Train Loss"]


def get_branches(gen_random=False, size=0, seed=None) -> list[list[int]]:
    if gen_random:
        if seed is not None:
            random.seed(seed)

        first_numbers = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 400, 500, 750, 1000]
        result = []
        for _ in range(size):
            num_branches = random.randint(1, 4)
            branch_lenght = random.choice([1, 2, 2, 2, 3, 3])
            group = []
            for _ in range(num_branches):
                branch = []
                curr = random.choice(first_numbers)
                branch.append(curr)
                for _ in range(branch_lenght - 1):
                    curr //= random.choice([1, 2, 2, 2, 3, 3])
                    if curr < 1:
                        curr = 1
                    branch.append(curr)

                group.append(branch)

            result.append(group)

        return result
    
    return [
        [[1]],
        [[1], [1]],
        [[1], [1], [1], [1]],
        [[1, 1]],
        [[1, 1], [1, 1]],
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
        [[50]],
        [[50], [100]],
        [[250, 125], [100, 50]],
        [[500, 250], [250, 125], [100, 50]],
        [[400, 200, 100], [20, 10, 5]],
        [[100, 50, 25], [30, 15, 7]],
        [[400, 6, 6], [22, 8, 8]], # Original model kernel-sizes
        [[90, 60, 30], [90, 30, 10], [30, 15, 7], [20, 10, 5]]
        #[[1500, 750, 375], [1000, 500, 250], [100, 25, 5], [200, 100, 50]], # TOO SLOW!!!!
    ]   



def load_data(batch_size: int, signal: str, sleep_stage: str, fraction: float | None = None) -> tuple[DataLoader, DataLoader, torch.Tensor]:
    STAGE_MAP = {
        0: 1 if sleep_stage == "W" else 0,
        1: 1 if sleep_stage == "N1" else 0,
        2: 1 if sleep_stage == "N2" else 0,
        3: 1 if sleep_stage == "N3" else 0,
        4: 1 if sleep_stage == "REM" else 0
    }

    
    with np.load(f"./Data/telemetry/TrainingData/telemetry_EEG_{signal}_train.npz") as data:
        Xtrain = data["X"].astype(np.float32)
        ytrain = data["y"]
        if fraction is not None:
            Xtrain = Xtrain[:int(len(Xtrain) * fraction)]
            ytrain = ytrain[:int(len(ytrain) * fraction)]

        Xtrain = np.expand_dims(Xtrain, 1)

        Xtrain = torch.tensor(Xtrain)
        ytrain = np.vectorize(STAGE_MAP.get)(ytrain)
        pos_weight = torch.tensor([(1 - ytrain.mean()) / ytrain.mean()]).to(device)
        ytrain = torch.tensor(ytrain)

        dataset = TensorDataset(Xtrain, ytrain)
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True
        )

    with np.load(f"./Data/telemetry/TestingData/telemetry_EEG_{signal}_test.npz") as data:
        Xtest = data["X"].astype(np.float32)
        ytest = data["y"]

        Xtest = np.expand_dims(Xtest, 1)

        Xtest = torch.tensor(Xtest)
        ytest = np.vectorize(STAGE_MAP.get)(ytest)
        ytest = torch.tensor(ytest)

        dataset = TensorDataset(Xtest, ytest)
        test_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True
        )

    return train_loader, test_loader, pos_weight


def plot():
    plot_paths = sorted(os.listdir("./Logs/penelope_plots/"))
    plot_paths.remove(".gitignore")
    for i, path in enumerate(plot_paths):
        print(f"{i+1}: {path}")
    plot_paths = ["./Logs/penelope_plots/" + p for p in plot_paths]

    print()
    plots_idx = input("Plot: ").split(",")
    plots_idx = [int(x) - 1 for x in plots_idx]
    plot_paths = [plot_paths[idx] for idx in plots_idx]

    num_rows, num_cols = len(plot_paths) // 2, 2
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 8))
    axes = axes.flatten()

    for i, path in enumerate(plot_paths):
        img = mpimg.imread(path)
        axes[i].imshow(img)
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def PnEL_FtMetric(PnEL, FtMetric, metric_name, plot_title, labels, epochs, name="pen_plot.png", save=True, n_labels=5, reverse=True):
    fig, ax = plt.subplots(figsize=(10, 6))
    points = np.column_stack((PnEL, FtMetric))
    
    scatter = ax.scatter(PnEL, FtMetric, color="blue", marker="o")
    
    ax.set_title(plot_title)
    ax.set_ylabel(metric_name)
    ax.set_xlabel(f"P-{epochs}-EL")
    ax.grid(True, linestyle="--", alpha=0.7)
    
    max_idx = np.argmax(FtMetric)
    min_idx = np.argmin(FtMetric)
    selected_indices = [max_idx, min_idx]
    
    if n_labels > 2:
        remaining_indices = [i for i in range(len(points)) if i not in selected_indices]
        remaining_points = points[remaining_indices]
        
        n_clusters = n_labels - 2
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(remaining_points)
        cluster_centers = kmeans.cluster_centers_
        
        for center in cluster_centers:
            dists = distance.cdist([center], remaining_points, 'euclidean')[0]
            selected_indices.append(remaining_indices[np.argmin(dists)])

    
    selected_indices = selected_indices[:min(n_labels, len(points))]
    sorted_indices = sorted(selected_indices, key=lambda i: FtMetric[i], reverse=reverse)
    
    legend_handles = []
    legend_labels = []
    
    markers = {
        "max": ('D', 'gold'),
        "min": ('v', 'lime'),
        "other": ('s', 'red')
    }
    
    for order, idx in enumerate(sorted_indices, start=1):
        x, y = PnEL[idx], FtMetric[idx]
        
        if idx == np.argmax(FtMetric):
            marker, color = markers['max']
            label_type = " (Best)"
        elif idx == np.argmin(FtMetric):
            marker, color = markers['min']
            label_type = " (Worst)"
        else:
            marker, color = markers['other']
            label_type = ""
        
        ax.annotate(
            str(order), 
            (x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7)
        )
        
        h = ax.scatter(
            [x], [y], 
            color=color, 
            marker=marker, 
            s=60,
            edgecolors='black',
            linewidths=0.7
        )
        legend_handles.append(h)
        legend_labels.append(f"{order}: {labels[idx]}{label_type}")
    
    slope, intercept = np.polyfit(PnEL, FtMetric, 1)
    x_line = np.linspace(min(PnEL), max(PnEL), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color="red", linestyle='--', alpha=0.7)
    
    ax.legend(
        handles=legend_handles,
        labels=legend_labels,
        title="Selected models",
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        borderaxespad=0.,
        fontsize=9,
        title_fontsize=10
    )
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    if "." not in name:
        name += ".png"
    else:
        n, e = name.split(".")
        if e not in ["png", "pdf", "svg", "jpeg"]:
            name = n + ".png"
    
    if save:
        plt.savefig(f"./Logs/penelope_plots/{name}", dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


if __name__ == "__main__":
    while True:
        print("\n",LoggingSettings.LOG_IDS)
        potential_log_id = input("Enter logging ID: ").upper().strip()
        if potential_log_id in LoggingSettings.LOG_IDS:
            LoggingSettings.LOGGER_ID = potential_log_id
            break
        else:
            print("❌ Please enter valid ID\n")
    
    main()