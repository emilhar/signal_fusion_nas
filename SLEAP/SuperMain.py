from ModelController.SuperModelController.MultimodalDataLoader import get_dataloaders_with_multimodal_datasets
from ModelController.ModelMaker import CNN_BinaryClassifier
from ModelController.SuperModelController.SuperModelMaker import EnsembleModel
from ModelController.SuperModelController.SuperTrainer import train_model
from Globals import Sleepstage, Signal, LoggingSettings, device
from Logs.LogManager import LogManager

import os
import torch

LoggingSettings.LOGGER_ID = "O"
def superMain():
    print("Loading Data")
    train_loader, test_loader = get_dataloaders_with_multimodal_datasets()
    print("Loading Models")
    models_dict = load_each_model()
    print("Training Model")
    ensomnia(models_dict, train_loader, test_loader)

def load_each_model():
        id_helper = LogManager()
        data_dir = f"Logs/{LoggingSettings.LOGGER_ID}Logs/ModelStateDicts/{id_helper.Experiment_ID-20}"
        all_model_files = [f for f in os.listdir(data_dir)]

        models_dict = {}

        for signal in Signal.ALL_SIGNALS:
            models_dict[signal] =[load_model(os.path.join(data_dir, model_path)) for model_path in all_model_files if signal in model_path]

        return models_dict

def load_model(model_path):
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
        epochs=5,
        lr=5e-5,
        wd=1e-4,
    )
    print("FART")
    print(model.load_state_dict(trained_state))

superMain()