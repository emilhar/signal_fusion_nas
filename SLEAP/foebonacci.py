import torch
import matplotlib.pyplot as plt
from ModelController.ModelMaker import CNN_BinaryClassifier
from ModelController._Trainer import train_model
from ModelController.BranchSettings import get_branch_configs
import pandas as pd
from pnel import load_data

"""
FOEbonacci

F1
Over
Epocs
-bonacci (taken from Fibonacci, due to the fact that we are using his famous sequence)
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_for_epochs(branch, signal, sleep_stage, epoch_list, lr=5e-5):
    f1_scores = []

    for epochs in epoch_list:
        print(f"\nTraining for {epochs} epochs...\n{'-'*40}")

        model_args = get_branch_configs(branch, "", 3000)
        model = CNN_BinaryClassifier(**model_args).to(device)

        train_loader, test_loader, pos_weight = load_data(
            batch_size=128, signal=signal, sleep_stage=sleep_stage
        )

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
            have_time_limit=False
        )

        f1_scores.append(res["Best F1"])

        log_ts(res, branch)

    return res

def plot_f1_curve(epoch_list, f1_scores, branch):
    plt.figure(figsize=(8, 6))
    plt.scatter(epoch_list, f1_scores, color='blue')
    plt.plot(epoch_list, f1_scores, linestyle='--', alpha=0.7)
    plt.title(f"F1 Score vs Epochs\nBranch: {branch}")
    plt.xlabel("Epochs")
    plt.ylabel("F1 Score")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f"Logs/Foebonacci_plots/{branch}_FOEBonacci.png", dpi=300, bbox_inches='tight')

def log_ts(res:dict, branch):
    # Convert the branch to its string representation for the 'Name' column
    branch_str = str(branch)
    res["Name"] = branch_str
    
    # Define the path for the CSV file
    csv_path = "Logs/OLogs/Foebonacci_Log.csv"

    res.pop("Branches")
    
    # Convert the res dictionary to a DataFrame
    df = pd.DataFrame([res])
    
    # Write to CSV, append if file exists, include headers if file doesn't exist
    df.to_csv(csv_path, mode='a', header=False, index=False)

if __name__ == "__main__":
    branches = [ [[300, 150, 75], [60, 30, 15]], [[3, 1, 1]], [[400, 8, 8], [22, 6, 6]] ]

    data = {
            55: 5,
            34: 5,
            21: 5,
            10: 10,
            7: 20,
            5: 30,
            4: 30,
            3: 40,
            2: 60,
            1: 100,
        }

    for branch in branches:
        signal = "Pz-Oz"
        sleep_stage = "N3"


        print("TRAINING NEW MODEL:", branch)
        training_list = []

        for key, value in data.items():
            for i in range(value):
                training_list.append(key)

        res = train_for_epochs(branch, signal, sleep_stage, training_list)