"""
Most important file, this connects to main.
"""

# Base Imports
import torch
from torch.utils.data import DataLoader

# Model and Training imports
from ModelController._Trainer import train_model
from ModelController.ModelMaker import CNN_BinaryClassifier
from ModelController.BranchSettings import get_branch_configs
from Globals import EvolutionManager, LoggingManager

class TrainedModelMaker:

    def __init__(self, 
                 branches:list[list[int]],
                 name:str, 
                 
                 sleepstage:str, 
                 signal_type:str, 

                 N_SAMPLES:int, 
                 pos_weight:torch.FloatTensor, 
                 train_loader:DataLoader, 
                 test_loader:DataLoader,

                 epochs:int,
                 batch_size:int,
                 learning_rate:int,

                 have_time_limit:bool = False,
                 verbose:bool = EvolutionManager.VERBOSE
        ):
        
        self.STAGE = sleepstage
        self.EXG_SIGNAL = signal_type
        self.lr = learning_rate

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_samples = N_SAMPLES
        self.pos_weight = pos_weight
        self.train_loader = train_loader
        self.test_loader = test_loader

        model_args = get_branch_configs(branches, name, self.n_samples) # See ModelManager
        model_args["batch_size"] = batch_size

        model = CNN_BinaryClassifier(**model_args).to(self.device)

        if verbose:
            print(f"\n\nTraining model: {branches=}, Generation: {LoggingManager.current_generation_id}/{EvolutionManager.GENERATIONS}, Generation Completeness: {LoggingManager.current_individual_id}/{LoggingManager.population_size}")

        self.model_performance = train_model(model, self.device, self.train_loader, self.test_loader, self.pos_weight, self.lr, epochs=epochs, verbose=verbose, have_time_limit=have_time_limit)

