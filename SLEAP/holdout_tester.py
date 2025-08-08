import os

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from datahelpers.data import Data
from dataloaders.multimodal_dataset import MultimodalLazyDataset
from models.cnn_binary_classifier import CNN_BinaryClassifier
from models.ensemble_model import EnsembleModel
from ensemble_controller.ensemble_plotter import analyze_predictions
from dataloaders.multimodal_dataset import get_dataloaders_with_multimodal_datasets

device = "cuda"
d = Data()

def load_each_model():
    models_dict = {}
    
    if os.path.exists("saved_models"):
        saved_model_files = [
            f for f in os.listdir("saved_models") 
            if f.endswith('.pt')
        ]
        
        for signal in d.signal_objects:
            signal_name = signal.name
            if signal_name not in models_dict:
                models_dict[signal_name] = []
                
            for model_file in saved_model_files:
                if signal_name in model_file:
                    full_path = os.path.join("saved_models", model_file)
                    models_dict[signal_name].append(load_model(full_path))
    
    return models_dict

def load_model(model_path):
    checkpoint = torch.load(model_path, map_location=device, weights_only=True, mmap=True)

    model_args = checkpoint["model_args"]

    model = CNN_BinaryClassifier(
        n_samples=model_args["n_samples"],
        branch_configs=model_args["branch_configs"],
        batch_size=model_args["batch_size"]
    )

    model.load_state_dict(checkpoint["state_dict"])
    return model


def load_ensemble(self, path):
    data = torch.load(path, map_location=device)
    
    models = {}
    for signal, configs in data["binary_classifier_configs"].items():
        models[signal] = [CNN_BinaryClassifier(**config) for config in configs]
    
    ensemble = EnsembleModel(models)
    ensemble.mlp.load_state_dict(data["mlp_state_dict"])
    return ensemble



def get_confusion_matrix(model, test_loader):
    all_true = []
    all_preds = []

    model.eval()
    model.to(device)

    with torch.inference_mode():
        for (datas, labels) in test_loader:
            datas = {k: v.to(device) for k, v in datas.items()}
            labels = labels.to(device)
            
            output = model(datas)
            pred_label = torch.argmax(output, dim=1).cpu()
            all_true.extend(labels.cpu().numpy())
            all_preds.extend(pred_label)

    all_true = np.array(all_true)
    all_preds = np.array(all_preds)

    analyze_predictions(all_true, all_preds, d.target_objects)
    return confusion_matrix(all_true, all_preds, labels=range(len(Data.get_all_target_names())), normalize="true")

def make_holdout_loader():
    
    # Initialize datasets to hold all signals
    loader = MultimodalLazyDataset()
    
    for signal in d.signal_objects:
        signal = signal.name
        data_directory = f"{d.DIRECTORY}/{d.dataset}/{signal}"
        all_files = sorted([f for f in os.listdir(data_directory) if f.endswith('.npz')])
        print(data_directory)

        loader.add_data(all_files, signal, data_directory, len(d.signal_objects))

    loader = DataLoader(
        loader,
        batch_size=128,
        shuffle=True,
        pin_memory=True if torch.cuda.is_available() else False,
        num_workers=1,
        persistent_workers=True,
    )

    return loader

def analyze_predictions(all_true, all_preds, targets):
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

    os.makedirs(f"_misc/confusion_matrices/{d}", exist_ok=True)
    plt.show()


def main():
    train_loader, test_loader = get_dataloaders_with_multimodal_datasets(d.target_objects, d.signal_objects)
    models_dict = load_each_model()
    print("Training Model...")
    model = EnsembleModel(models_dict)
    model.load_state_dict(EnsembleModel.train_model(model, train_loader, test_loader, epochs=10))

    holdout_set = "_misc/holdout_sleep-EDF-78"
    Data.DIRECTORY = "_misc"
    d.dataset = "holdout_sleep-EDF-78"

    holdout = make_holdout_loader()

    cm = get_confusion_matrix(model, holdout)
    print(cm)


if __name__ == "__main__":
    main()
