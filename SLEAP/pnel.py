import torch
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Globals import LoggingSettings
from ModelController.ModelMaker import CNN_BinaryClassifier
from ModelController._Trainer import train_model
from ModelController.BranchSettings import get_branch_configs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    if device.type == "cpu":
        raise ValueError("Device is CPU, not GPU")
    
    signal = input("Signal: ")
    if signal not in ["Fpz-Cz", "Pz-Oz", "EMG submental", "EOG horizontal"]:
        raise ValueError("Not valid signal")

    sleep_stage = input("Sleep stage: ")
    if sleep_stage not in ["W", "N1", "N2", "N3", "REM"]:
        raise ValueError("Not valid sleep stage")

    branches = [
    [1, 1500, 1],
    [17, 999, 42],
    [5, 5, 1500],
    [123, 456, 789],
    [10, 1000, 10],
    [1500, 1, 1500],
    [2, 4, 8, 16, 32],
    [777, 777, 777],
    [64, 128, 256],
    [1, 2, 3,]
    ]

    epochs = 3

    PnEL, FtF1 = penelope(branches, epochs, signal, sleep_stage)

    plot(PnEL, FtF1, [str(x) for x in branches], epochs)
    

def penelope(branches: list[list[int]], epochs: int, signal: str, sleep_stage: str) -> tuple[list[float], list[float]]:
    PnEL = []
    FtF1 = []

    temp = []
    if not branches:
        for log_id in LoggingSettings.LOG_IDS:
            with open(f"Logs/{log_id}_fully_trained_models.csv") as f:
                df = pd.read_csv(f)
                # Filter by signal and sleep stage
                filtered_df = df[(df["signal"] == signal) & (df["sleep_stage"] == sleep_stage)]
                for b in filtered_df["name"]:
                    temp.append(eval(b))

    branches = temp

    for branch in branches:
        FtF1.append(fully_train(branch, signal, sleep_stage))
        PnEL.append(partial_train(branch, epochs, signal, sleep_stage))

    return PnEL, FtF1

def fully_train(branch: list[int], signal: str, sleep_stage: str) -> float:
    title = f"Training Full model with kernels {branch}"
    for log_id in LoggingSettings.LOG_IDS:
        with open(f"./Logs/{log_id}_fully_trained_models.csv") as f:
            df = pd.read_csv(f)
            # Check for matching branch, signal, and sleep stage
            mask = (df["name"] == str(branch)) & (df["signal"] == signal) & (df["sleep_stage"] == sleep_stage)
            if mask.any():
                print(f"Model {branch} already in {LoggingSettings.LOGGER_ID}_fully_train_models.csv, skipping...")
                return df.loc[mask, "F1"].values[0]
    
    with open(f"./Logs/{LoggingSettings.LOGGER_ID}_fully_trained_models.csv") as f:
        print(f"\n{"="*len(title)}")
        print(title)
        print(f"{"="*len(title)}\n")

        model_args = get_branch_configs([branch], "Fully Trained", 3000)
        model = CNN_BinaryClassifier(**model_args).to(device)

        train_loader, test_loader, pos_weight = load_data(128, signal, sleep_stage)

        res = train_model(
            model, 
            device, 
            train_loader, 
            test_loader, 
            pos_weight, 
            5e-7, 
            wd=0.0001, 
            p=5, 
            f=0.5, 
            epochs=40, 
            output_period=5, 
            verbose=True,
            have_time_limit=False # Time limit switch
        )

        new_row = pd.DataFrame([{
            "name": str(branch), 
            "F1": res["F1"], 
            "signal": signal,
            "sleep_stage": sleep_stage
        }])
        df = pd.concat([df, new_row], ignore_index=True)
        new_row.to_csv(f"./Logs/{LoggingSettings.LOGGER_ID}_fully_trained_models.csv", mode="a", header=False, index=False)
    
    return res["Best F1"]

def partial_train(branch: list[int], epochs: int, signal: str, sleep_stage: str) -> float:
    model_args = get_branch_configs([branch], "Fully Trained", 3000)
    model = CNN_BinaryClassifier(**model_args).to(device)

    train_loader, test_loader, pos_weight = load_data(32, signal, sleep_stage, 0.1)

    title = f"Training Partial model with kernels {branch}"
    print(f"\n{"="*len(title)}")
    print(title)
    print(f"{"="*len(title)}\n")
    res = train_model(
        model, 
        device, 
        train_loader, 
        test_loader, 
        pos_weight, 
        5e-5, 
        wd=0.0001, 
        p=5, 
        f=0.5, 
        epochs=epochs, 
        output_period=1, 
        verbose=True,
        have_time_limit=False # Time limit switch
    )

    return res["Train Loss"]

def load_data(batch_size: int, signal: str, sleep_stage: str, fraction: float | None = None) -> tuple[DataLoader, DataLoader, torch.Tensor]:
    # Map sleep stage to binary classification
    STAGE_MAP = {
        "W": 0,
        "N1": 0,
        "N2": 0,
        "N3": 1,
        "REM": 0
    }
    
    # Load training data
    with np.load(f"./Data/telemetry/TrainingData/telemetry_EEG_{signal}_train.npz") as data:
        Xtrain = data["X"].astype(np.float32)
        ytrain = data["y"]
        if fraction is not None:
            Xtrain = Xtrain[:int(len(Xtrain) * fraction)]
            ytrain = ytrain[:int(len(ytrain) * fraction)]

        Xtrain = np.expand_dims(Xtrain, 1)

        Xtrain = torch.tensor(Xtrain)
        ytrain = np.array([STAGE_MAP[sleep_stage] if y == sleep_stage else 0 for y in ytrain])
        pos_weight = torch.tensor([(1 - ytrain.mean()) / ytrain.mean()]).to(device)
        ytrain = torch.tensor(ytrain)

        dataset = TensorDataset(Xtrain, ytrain)
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True
        )

    # Load testing data
    with np.load(f"./Data/telemetry/TestingData/telemetry_EEG_{signal}_test.npz") as data:
        Xtest = data["X"].astype(np.float32)
        ytest = data["y"]

        Xtest = np.expand_dims(Xtest, 1)

        Xtest = torch.tensor(Xtest)
        ytest = np.array([STAGE_MAP[sleep_stage] if y == sleep_stage else 0 for y in ytest])
        ytest = torch.tensor(ytest)

        dataset = TensorDataset(Xtest, ytest)
        test_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True
        )

    return train_loader, test_loader, pos_weight

def plot(PnEL, FtF1, labels, epochs):
    offset_range = 8
    plt.scatter(PnEL, FtF1, color="blue", marker="o")

    plt.title(f"Fully-trained F1 vs. Partial {epochs} Train Loss")
    plt.ylabel("FtF1")
    plt.xlabel(f"P{epochs}EL")
    plt.ylim(0.0, 1.0)
    plt.xlim(0.0, 1.5)

    plt.grid(True, linestyle="--", alpha=0.7)

    for i, label in enumerate(labels):
        plt.annotate(
            label, (PnEL[i], FtF1[i]),
            xytext=(0, 8),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="gray", alpha=0.5)
        )

    plt.show()

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