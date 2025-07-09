import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve
from ModelController.ModelMaker import CNN_BinaryClassifier
from ModelController._Trainer import train_model
from ModelController.BranchManager import get_branch_configs
from pnel import load_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_PATH = "./Logs/TLogs/roc_auc.csv"
FIG_PATH = "./Logs/TLogs/roc_plots/"

def main():
    if input("Plot? (y/*) ").lower() == "y":
        return
    
    save = input("Save plots? (y/*)").lower() == "y"
    train_all(save)


def train_all(save):
    num_branches = input("num branches: ")
    if num_branches == "":
        branch = [[300, 100, 20], [40, 20, 10], [22, 6, 6]]
    else:
        branch = [list(map(int, input("Enter branch: ").split())) for _ in range(int(num_branches))]
    
    i = 1
    for signal in ["EEG_Fpz-Cz", "EEG_Pz-Oz", "EMG_submental", "EOG_horizontal"]:
        for sleep_stage in ["W", "N1", "N2", "N3", "REM"]:
            model_args = get_branch_configs(branch, "", 3000)
            train_loader, test_loader, pos_weight = load_data(batch_size=128, signal=signal, sleep_stage=sleep_stage)
            model = CNN_BinaryClassifier(**model_args).to(device)
            res = train_model(
                model,
                device,
                train_loader,
                test_loader,
                pos_weight,
                lr=5e-6,
                wd=0.0001,
                p=5,
                f=0.5,
                epochs=40,
                output_period=1,
                verbose=True,
                have_time_limit=False,
            )

            fpr, tpr, thresholds = roc_curve(res["True Labels"], res["Best Scores"])
            
            best_f1, best_threshold = find_best_f1_threshold(res["True Labels"], res["Best Scores"])
            print(f"Best F1-score: {best_f1:.4f} at threshold: {best_threshold:.4f}")    
            plot_roc_curve(fpr, tpr, res["Best AUC"], str(res["Branches"]), signal, sleep_stage, i, save=save)
            i += 1


def find_best_f1_threshold(y_true, y_scores):
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)  # Add small epsilon to avoid division by zero
    
    best_idx = np.argmax(f1_scores[:-1]) # Exclude last value (added by precision_recall_curve)
    best_f1 = f1_scores[best_idx]
    best_threshold = thresholds[best_idx]
    
    return best_f1, best_threshold


def plot_roc_curve(fpr, tpr, roc_auc, title, signal, sleep_stage, i, save=False):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title + " " + signal + " " + sleep_stage)
    plt.legend(loc="lower right")
    if save:
        plt.savefig(FIG_PATH + f"{i}.png")
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    main()