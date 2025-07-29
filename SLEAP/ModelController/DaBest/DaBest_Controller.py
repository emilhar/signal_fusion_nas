from ModelController.ModelMaker import CNN_BinaryClassifier
from ModelController.DaBest.DaBest_DataLoader import get_dataloaders_with_multimodal_datasets
from ModelController.DaBest.DaBest_Maker import EnsembleModel
from ModelController.DaBest.DaBest_Trainer import train_model
from ModelController.DaBest.DaBest_Plotter import analyze_predictions
from Globals import Classes, Signal, LoggingSettings, device
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from Logs.LogManager import LogManager

import os
import torch
import numpy as np

LoggingSettings.LOGGER_ID = "O"
def superMain(given_folder=None, model_marker=None):
    print("📦 Loading Data...")
    train_loader, test_loader = get_dataloaders_with_multimodal_datasets()

    print("🧠 Loading Models...")
    models_dict = load_each_model(given_folder)

    print("🚀 Training Model...")
    model = EnsembleModel(models_dict)
    trained_state = ensomnia(models_dict, train_loader, test_loader)
    assert device.type == "cuda", "Device is not CUDA"
    model.to(device).load_state_dict(trained_state)

    print("📊 Plotting Results...")
    plot(model, test_loader, model_marker)

def load_each_model(given_folder):
        id_helper = LogManager()

        if given_folder:
            data_dir = given_folder
        else:
            data_dir = f"Logs/{LoggingSettings.LOGGER_ID}Logs/ModelStateDicts/{id_helper.Experiment_ID-20}"

        all_model_files = [f for f in os.listdir(data_dir)]
        models_dict = {}
        for signal in Signal.ALL_SIGNALS:
            models_dict[signal] =[load_model(os.path.join(data_dir, model_path)) for model_path in all_model_files if signal in model_path]

        return models_dict

def load_model(model_path):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        checkpoint = torch.load(model_path, map_location=device)

    model_args = checkpoint["model_args"]

    model = CNN_BinaryClassifier(
        n_samples=model_args["n_samples"],
        branch_configs=model_args["branch_configs"],
        batch_size=model_args["batch_size"]
    )

    model.load_state_dict(checkpoint["state_dict"])
    return model

def ensomnia(models_dict, train_loader, test_loader):
    model = EnsembleModel(models_dict).to(device)

    trained_state = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=1,
        lr=5e-5,
        wd=1e-4,
    )
    
    return trained_state


def plot(model: torch.nn.Module, test_loader: DataLoader, model_marker):
    all_true = []
    all_preds = []

    model.eval()
    model.to(device)  # Ensure model is on the right device

    with torch.inference_mode():
        for (datas, labels) in test_loader:
            # Move data to the same device as model
            datas = {k: v.to(device) for k, v in datas.items()}
            labels = labels.to(device)
            
            output = model(datas)
            pred_label = torch.argmax(output, dim=1).cpu()
            all_true.extend(labels.cpu().numpy())
            all_preds.extend(pred_label)

    all_true = np.array(all_true)
    all_preds = np.array(all_preds)

    print("\nTest Classification Report:")
    print(classification_report(
        all_true,
        all_preds,
        target_names=Classes.All_CLASSES,
        digits=4,
        zero_division=0
    ))

    print("\n\n\n\n")

    analyze_predictions(all_true, all_preds, model_marker)