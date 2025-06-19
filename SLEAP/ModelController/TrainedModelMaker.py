"""
Most important file, this connects to main.
"""

# Base Imports
import torch
from torch.utils.data import DataLoader
from Globals import ModelSettings

# Model and Training imports
from ModelController._Trainer import train_model
from ModelController.ModelMaker import CNN_BinaryClassifier
from ModelController.BranchSettings import get_branch_configs

class TrainedModelMaker:

    def __init__(self, 
                 left_kernel_sizes:list [int], 
                 right_kernel_sizes:list[int],
                 name:str, 
                 
                 sleepstage:str, 
                 signal_type:str, 

                 N_SAMPLES:int, 
                 pos_weight:torch.FloatTensor, 
                 train_loader:DataLoader, 
                 test_loader:DataLoader,

                 batch_size:int = ModelSettings.BATCH_SIZE,
                 epochs:int = ModelSettings.TRAINING_EPOCHS_PER_INDIVIDUAL,
                 dataset_fraction:float = ModelSettings.DATASET_FRACTION,

                 champion:bool = False,
                 verbose:bool = ModelSettings.VERBOSE
        ):
        
        self.STAGE = sleepstage
        self.EXG_SIGNAL = signal_type
        self.DATASET_FRACTION = dataset_fraction

        # Get Stage Map
        self.BATCH_SIZE = batch_size

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_samples = N_SAMPLES
        self.pos_weight = pos_weight
        self.train_loader = train_loader
        self.test_loader = test_loader

        model_args = get_branch_configs(left_kernel_sizes, right_kernel_sizes, name, self.n_samples) # See ModelSettings

        model = CNN_BinaryClassifier(**model_args).to(self.device)

        if verbose: 
            print(f"\n\nTraining model: {left_kernel_sizes=}, {right_kernel_sizes=}")

        self.model_performance = train_model(model, self.device, self.train_loader, self.test_loader, self.pos_weight, lr=5e-5, epochs=epochs, verbose=verbose, champion=champion)

