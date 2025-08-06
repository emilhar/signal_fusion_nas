import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from Globals import *
from datetime import datetime
from datahelpers.signal import Signal
from datahelpers.target import Target


def plot_sample_predictions(model, X, y, sample_idx, class_names, true_label=None, pred_label=None):
    signals = {}

    for sig in Signal.ALL_SIGNALS:
        signals[sig] = X[sig][sample_idx]

    if true_label is None:
        true_label = y[sample_idx]

    if pred_label is None:
        model.eval()
        with torch.inference_mode():
            input_dict = {}
            for ch, signal in signals.items():
                input_dict[ch] = signal.clone().detach().float().transpose(0, 1).unsqueeze(0).to(device)

            output = model(input_dict)
            probs = torch.softmax(output, dim=1)
            pred_label = torch.argmax(probs, dim=1).item()

    fig, axs = plt.subplots(4, 1, figsize=(14, 10))
    fig.suptitle(f"Sample {sample_idx} - True: {class_names[true_label]}, Pred: {class_names[pred_label]}",
                 fontsize=16, fontweight='bold')

    for i, (ch, signal) in enumerate(signals.items()):
        axs[i].plot(signal, color='royalblue')
        axs[i].set_title(f"{ch} ({len(signal)} samples)", fontsize=12)
        axs[i].set_ylabel("Amplitude", fontsize=10)
        axs[i].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.show()

    return true_label, pred_label

def analyze_predictions(all_true, all_preds, targets:list[Target]):
    class_names = [target.given_name for target in targets]

    cm = confusion_matrix(all_true, all_preds, labels=range(len(class_names)), normalize="true")

    plt.figure(figsize=(10, 8))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )
    disp.plot(cmap="Blues", values_format=".2f")

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    a = str(datetime.now().replace(microsecond=0)).replace(" ", "_").replace(":", "-")
    d = Globals.confusion_matrix_folder_name
    fig_path = f"_misc/confusion_matrices/{d}/Experiment_{a}.png"
    os.makedirs(f"_misc/confusion_matrices/{d}", exist_ok=True)
    plt.savefig(fig_path)
    print(f"Ensemble model plot saved at: {fig_path}")

    # for class_idx, class_name in enumerate(class_names):
    #     print(f"\n===== {class_name} Analysis =====")

    #     # True Positives (TP): Correctly predicted as this class
    #     tp_indices = np.where((all_true == class_idx) & (all_preds == class_idx))[0]
    #     print(f"True Positives: {len(tp_indices)} samples")

    #     # False Positives (FP): Predicted as this class but actually not
    #     fp_indices = np.where((all_true != class_idx) & (all_preds == class_idx))[0]
    #     print(f"False Positives: {len(fp_indices)} samples")

    #     # False Negatives (FN): Actually this class but predicted as something else
    #     fn_indices = np.where((all_true == class_idx) & (all_preds != class_idx))[0]
    #     print(f"False Negatives: {len(fn_indices)} samples")

    #     # True Negatives (TN): Not this class and not predicted as that class
    #     tn_indices = np.where((all_true != class_idx) & (all_preds == all_true))[0]
    #     print(f"True Negatives: {len(tn_indices)} samples")