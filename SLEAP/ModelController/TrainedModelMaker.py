# Base Imports
import torch
from torch.utils.data import DataLoader

# Model and Training imports
from ModelController._Trainer import train_model
from ModelController.ModelMaker import CNN_BinaryClassifier
from ModelController.BranchSettings import get_branch_configs
from Globals import EvolutionManager, ModelManager, LoggingSettings

class TrainedModelMaker:

    def __init__(self, 
                 branches:list[list[int]],
                 N_SAMPLES:int, 
                 pos_weight:torch.FloatTensor, 
                 train_loader:DataLoader, 
                 test_loader:DataLoader
        ):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model_args = get_branch_configs(branches, N_SAMPLES) # See ModelManager
        model_args["batch_size"] = ModelManager.BATCH_SIZE
        model = CNN_BinaryClassifier(**model_args).to(device)

        if EvolutionManager.VERY_VERBOSE:
            print(f"\n\nTraining model: {branches=}, Generation: {LoggingSettings.current_generation_id}/{EvolutionManager.GENERATIONS}, Generation Completeness: {LoggingSettings.current_individual_id}/{LoggingSettings.population_size}")

        self.model_performance = train_model(
            model, 
            device, 
            train_loader, 
            test_loader, 
            pos_weight, 
            verbose=EvolutionManager.VERY_VERBOSE, 
            lr=ModelManager.LEARNING_RATE, 
            epochs=ModelManager.TRAINING_EPOCHS_PER_INDIVIDUAL)