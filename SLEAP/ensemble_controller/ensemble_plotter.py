import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
from Globals import device, Signal, Targets, LoggingSettings
from log_manager.log_manager import LogManager
from datetime import datetime

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

def analyze_predictions(all_true, all_preds, model_marker=None, filters=None): # TODO: remove filters=None
    class_names = Classes.All_CLASSES

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

    if model_marker:
        a = str(datetime.now().replace(microsecond=0)).replace(" ", "_").replace(":", "-") + f"_{filters}-filters"
        fig_path =f"_misc/confusion_matrices/{model_marker}_{a}.png"
        plt.savefig(fig_path)
        print(f"Ensemble model plot saved at: {fig_path}")
    else:
        id_helper = LogManager()
        fig_path = f"_misc/confusion_matrices/Experiment_{id_helper.experiment_id-20}.png"
        plt.savefig(fig_path)
        print(f"Ensemble model plot saved at: {fig_path}")

    for class_idx, class_name in enumerate(class_names):
        print(f"\n===== {class_name} Analysis =====")

        # True Positives (TP): Correctly predicted as this class
        tp_indices = np.where((all_true == class_idx) & (all_preds == class_idx))[0]
        print(f"True Positives: {len(tp_indices)} samples")

        # False Positives (FP): Predicted as this class but actually not
        fp_indices = np.where((all_true != class_idx) & (all_preds == class_idx))[0]
        print(f"False Positives: {len(fp_indices)} samples")

        # False Negatives (FN): Actually this class but predicted as something else
        fn_indices = np.where((all_true == class_idx) & (all_preds != class_idx))[0]
        print(f"False Negatives: {len(fn_indices)} samples")

        # True Negatives (TN): Not this class and not predicted as that class
        tn_indices = np.where((all_true != class_idx) & (all_preds == all_true))[0]
        print(f"True Negatives: {len(tn_indices)} samples")