import torch
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from ModelController.ModelMaker import CNN_BinaryClassifier
from ModelController._Trainer import train_model
from ModelController.BranchSettings import get_branch_configs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    if device.type == "cpu":
        raise ValueError("Device is CPU, not GPU")
    
    branches = [
        [1, 1, 1],
        [50, 50, 50],
        [400, 400, 400],
        [1, 1, 1500],
        [50, 25, 13],
        [100, 50, 25],
        [150, 75, 38], 
        [200, 100, 50],
        [250, 125, 63],
        [300, 150, 75],
        [350, 175, 88],
        [400, 200, 100],
        [450, 225, 113],
        [500, 250, 125],
        [400, 8, 8],
        [22, 6, 6],
        [270, 90, 30]
    ]
    epochs = 1

    PnEL, FtF1 = penelope(branches, epochs)

    plot(PnEL, FtF1, [str(x) for x in branches], epochs)
    


def penelope(branches: list[list[int]], epochs: int) -> tuple[list[float], list[float]]:
    PnEL = []
    FtF1 = []
    for branch in branches:
        FtF1.append(fully_train(branch))
        PnEL.append(partial_train(branch, epochs))

    return PnEL, FtF1

def fully_train(branch: list[int]) -> float:
    with open("./Logs/fully_trained_models.csv") as f:
        title = f"Training Full model with kernels {branch}"
        print(f"\n{"="*len(title)}")
        print(title)
        print(f"{"="*len(title)}\n")
        df = pd.read_csv(f)
        if (df["name"] == str(branch)).any():
            print(f"Model {branch} already in fully_train_models.csv, skipping...")
            return df.loc[df["name"] == str(branch), "F1"].values[0]

        model_args = get_branch_configs([branch], "Fully Trained", 3000)
        model = CNN_BinaryClassifier(**model_args).to(device)

        train_loader, test_loader, pos_weight = load_data(128)

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
            champion=True # Time limit switch
        )

        new_row = pd.DataFrame([{"name": str(branch), "F1": res["F1"]}])
        df = pd.concat([df, new_row], ignore_index=True)
        new_row.to_csv("./Logs/fully_trained_models.csv", mode="a", header=False, index=False)
    
    return res["Best F1"]

def partial_train(branch: list[int], epochs: int) -> float:
    model_args = get_branch_configs([branch], "Fully Trained", 3000)
    model = CNN_BinaryClassifier(**model_args).to(device)

    train_loader, test_loader, pos_weight = load_data(32, 0.1)

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
        champion=True # Time limit switch
    )

    return res["Train Loss"]

def load_data(batch_size: int, fraction: float | None = None) -> tuple[DataLoader, DataLoader, torch.Tensor]:
    with np.load("./Data/telemetry/TrainingData/telemetry_EEG_Pz-Oz_train.npz") as data:
        STAGE_MAP = {
            0: 0,
            1: 0,
            2: 0,
            3: 1,
            4: 0
        }
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

    with np.load("./Data/telemetry/TestingData/telemetry_EEG_Pz-Oz_test.npz") as data:
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
    main()