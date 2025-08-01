from models.cnn_binary_classifier import CNN_BinaryClassifier
from dataloaders.multimodal_dataset import get_dataloaders_with_multimodal_datasets
from models.ensemble_model import EnsembleModel
from ensemble_controller.ensemble_plotter import analyze_predictions
from Globals import LoggingHelper, device
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from log_manager.log_manager import LogManager
from datahelpers.data import Data

import os
import torch
import warnings
import numpy as np




import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix




class EnsembleController:
    def __init__(self, targets, signals, debug=False):
        self.targets = targets
        self.signals = signals
        self.debug = debug

    def create_ensemble(self, use_temp=False):
        print("📦 Loading Data...")
        train_loader, test_loader = get_dataloaders_with_multimodal_datasets(self.targets, self.signals)

        print("🧠 Loading Models...")
        models_dict = self.load_each_model(use_temp)

        print("🚀 Training Model...")
        model = EnsembleModel(models_dict)
        trained_state = self.ensomnia(models_dict, train_loader, test_loader)
        assert device.type == "cuda", "Device is not CUDA"
        model.to(device).load_state_dict(trained_state)

        cm = self.get_confusion_matrix(model, test_loader)
        print(cm.shape)
        target_ranking = []
        for i, target in enumerate(self.targets):
            target_ranking.append(
                (target, cm[i][i])
            )
        
        return sorted(target_ranking, key=lambda x: x[1])


    def load_each_model(self, use_temp):
        # saved_models/
        # temp_models/
        #
        # load from temp models if use_temp=True
        # load from saved_models, but not files with the same prefix as those loaded from temp if use_temp=True
        # Prefix is anything that comes before the first underline: prefix_something_else.pt

        # Determine which directory to load from
        base_dir = "temp_models" if use_temp else "saved_models"
        all_model_files = [f for f in os.listdir(base_dir) if f.endswith('.pt')]
        
        # If loading from temp, get the prefixes to exclude from saved_models
        temp_prefixes = set()
        if use_temp:
            for f in all_model_files:
                prefix = f.split('_')[0]
                temp_prefixes.add(prefix)
        
        models_dict = {}
        for signal in self.signals:
            signal_name = signal.name
            models_dict[signal_name] = []
            
            # First load from temp if use_temp is True
            if use_temp:
                for model_path in all_model_files:
                    if signal_name in model_path:
                        print("LOADING TEMPORARY MODEL")
                        full_path = os.path.join("temp_models", model_path)
                        models_dict[signal_name].append(self.load_model(full_path))
            
            # Then load from saved_models, excluding files with prefixes found in temp
            saved_model_files = [f for f in os.listdir("saved_models") 
                            if f.endswith('.pt') and 
                            (not use_temp or f.split('_')[0] not in temp_prefixes)]
            
            for model_path in saved_model_files:
                if signal_name in model_path:
                    full_path = os.path.join("saved_models", model_path)
                    models_dict[signal_name].append(self.load_model(full_path))

        assert len(models_dict.keys()) == len(Data.get_all_signal_names()), "Not enough models"
        
        for k in models_dict.keys():
            assert len(models_dict[k]) == len(Data.get_all_target_names()), "Not all signals have all targets"
        
        return models_dict

    def load_model(self, model_path):
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

    def ensomnia(self, models_dict, train_loader, test_loader):
        model = EnsembleModel(models_dict).to(device)

        trained_state = EnsembleModel.train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=1 if self.debug else 5,
            lr=5e-5,
            wd=1e-4,
        )
        
        return trained_state


    def plot(self, model: torch.nn.Module, test_loader: DataLoader):
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

        # print("\nTest Classification Report:")
        # print(classification_report(
        #     all_true,
        #     all_preds,
        #     target_names=[target.given_name for target in self.targets],
        #     digits=4,
        #     zero_division=0
        # ))
        # print()

        # analyze_predictions(all_true, all_preds, self.targets, model_marker, EnsembleController.NUM___FILTERS)

        # print("Turning you into sludge... loading ensludginator")

        # cm = fobg.confusion_matrix(all_true, all_preds, labels=range(len(Data.get_all_target_names())), normalize="true")


        # fogb.figure(figsize=(10, 8))

        # disp = fobg.ConfusionMatrixDisplay(
        #     confusion_matrix=cm,
        #     display_labels=Data.get_all_target_names()
        # )
        # disp.plot(cmap="Blues", values_format=".2f")

        # fogb.xlabel("Predicted")
        # fogb.ylabel("True")
        # fogb.title("Confusion Matrix")

        # fogb.tight_layout()
        # fogb.show()

    
    def get_confusion_matrix(self, model, test_loader):
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

        # print("\nTest Classification Report:")
        # print(classification_report(
        #     all_true,
        #     all_preds,
        #     target_names=[target.given_name for target in self.targets],
        #     digits=4,
        #     zero_division=0
        # ))
        # print()

        # analyze_predictions(all_true, all_preds, self.targets, model_marker, EnsembleController.NUM___FILTERS)

        return confusion_matrix(all_true, all_preds, labels=range(len(Data.get_all_target_names())), normalize="true")



    def save_ensemble(self, ensemble_model, path):
        save_data = {
            "mlp_state_dict": ensemble_model.mlp.state_dict(),
            "binary_classifier_configs": {
                signal: [model.model_args for model in ffe_list] 
                for signal, ffe_list in ensemble_model.feature_extractors.items()
            }
        }
        torch.save(save_data, path)

    def load_ensemble(self, path, device="cuda"):
        data = torch.load(path, map_location=device)
        
        models = {}
        for signal, configs in data["binary_classifier_configs"].items():
            models[signal] = [CNN_BinaryClassifier(**config) for config in configs]
        
        ensemble = EnsembleModel(models)
        ensemble.mlp.load_state_dict(data["mlp_state_dict"])
        return ensemble