import torch
import matplotlib.pyplot as plt
from ModelController.ModelMaker import CNN_BinaryClassifier
from ModelController._Trainer import train_model
from ModelController.BranchSettings import get_branch_configs
import pandas as pd
from pnel import load_data
import os

"""
FOEbonacci

F1
Over
Epocs
-bonacci (taken from Fibonacci, due to the fact that we are using his famous sequence)
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_PATH = "Logs/TLogs/Foebonacci_Log.csv"
PLOT_DIR = "Logs/Foebonacci_plots"

def train_one_run(branch, signal, sleep_stage, epochs, lr=5e-5):
    """Train a single model run and return results"""
    model_args = get_branch_configs(branch, "", 3000)
    model = CNN_BinaryClassifier(**model_args).to(device)

    train_loader, test_loader, pos_weight = load_data(
        batch_size=128, signal=signal, sleep_stage=sleep_stage
    )

    return train_model(
        model,
        device,
        train_loader,
        test_loader,
        pos_weight,
        lr=lr,
        wd=0.0001,
        p=2,
        f=0.5,
        epochs=epochs,
        output_period=1,
        verbose=True,
        have_time_limit=False,
    )

def log_result(res, branch):
    res = res.copy()
    res["Name"] = str(branch)
    res.pop("Branches", None)
    
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    df = pd.DataFrame([res])
    df.to_csv(LOG_PATH, mode='a', header=not os.path.exists(LOG_PATH), index=False)

def plot_f1_boxplot(df, branch, save=False):
    branch_df = df[df['Name'] == str(branch)].copy()
    if branch_df.empty:
        print(f"No data for branch {branch}")
        return
    
    branch_df.loc[:, 'Epoch'] = branch_df['Epoch'] + 1
    
    grouped = branch_df.groupby('Epoch')['Best F1'].apply(list)
    if grouped.empty:
        print(f"No valid F1 scores for branch {branch}")
        return
    
    epochs = sorted(grouped.index)
    data = [grouped[epoch] for epoch in epochs]
    positions = range(1, len(epochs) + 1)
    
    plt.figure(figsize=(12, 7))
    plt.boxplot(data, positions=positions)
    
    plt.title(f"F1 Score Distribution by Epoch\nBranch: {branch}")
    plt.xlabel("Epochs")
    plt.ylabel("F1 Score")
    plt.xticks(positions, epochs, rotation=45)
    plt.grid(True, axis='y')
    plt.tight_layout()

    if save:
        os.makedirs(PLOT_DIR, exist_ok=True)
        plt.savefig(f"{PLOT_DIR}/{branch}_FOEBonacci.png", dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def load_or_create_log():
    if os.path.exists(LOG_PATH):
        return pd.read_csv(LOG_PATH)
    return pd.DataFrame()

def needs_training(branch, epoch, df_log, required_count):
    branch_str = str(branch)
    existing_count = len(df_log[(df_log['Name'] == branch_str) & 
                               (df_log['Epoch'] == epoch)])
    return max(0, required_count - existing_count)

if __name__ == "__main__":
    if input("Generate plots? (y/n) ").lower() == "y":
        df = pd.read_csv(LOG_PATH) if os.path.exists(LOG_PATH) else pd.DataFrame()
        if df.empty:
            print("No log data available for plotting")
        else:
            for branch in df['Name'].unique():
                save = True if input("Save? (y/n) ") == "y" else False
                plot_f1_boxplot(df, branch)
        exit()

    BRANCHES = [[[300, 150, 75]], [[400, 8, 8], [22, 6, 6]], [[3, 1, 1], [4, 1, 1]], [[22, 4], [11, 2], [10, 5]]]
    
    EPOCH_RUNS = {
        1: 100, 
        2: 80, 
        3: 75, 
        4: 70, 
        5: 60,
        7: 100, 
        10: 50, 
        20: 25, 
        40: 40, 
    }

    df_log = load_or_create_log()
    SIGNAL = "EEG_Fpz-Cz"
    SLEEP_STAGE = "W"

    quit()

    for signal in ["EEG_Fpz-Cz", "EEG_Pz-Oz", "EMG_submental", "EOG_horizontal"]:
        for sleep_stage in ["W", "N1", "N2", "N3", "REM"]:
            for branch in BRANCHES:
                print(f"\n{'='*40}\nProcessing branch: {branch}\n{'='*40}")
                
                for epochs, required_count in EPOCH_RUNS.items():
                    runs_needed = needs_training(branch, epochs, df_log, required_count)
                    if runs_needed == 0:
                        print(f"Already have {required_count} runs for {epochs} epochs. Skipping.")
                        continue

                    print(f"Training {runs_needed} model(s) for {epochs} epochs...")
                    for _ in range(runs_needed):
                        result = train_one_run(branch, SIGNAL, SLEEP_STAGE, epochs)
                        log_result(result, branch)
                    
                    df_log = load_or_create_log()