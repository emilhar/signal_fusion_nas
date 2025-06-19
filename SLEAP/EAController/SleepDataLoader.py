import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch
import gc
from random import sample

from ModelController.ModelMaker import CNN_BinaryClassifier
from Globals import Sleepstage, EvolutionSettings, DataSettings


class SleepDataLoader:
    def __init__(self, verbose, signal_type, sleepstage, dataset_fraction, batch_size):
        self.sleepstage = sleepstage
        self.signal_type = signal_type

        if DataSettings.DATASET == DataSettings.DatasetNames.TELEMETRY:
            self.signal_type = f"telemetry_{signal_type}"

        self.dataset_fraction = dataset_fraction
        self.batch_size = batch_size

        self.verbose = verbose
        
        try:
            train_file_path = self.get_filepath(SLEAP=True, data_type="Training")            
            self.train_loader, self.pos_weight, self.n_samples = self._load_data(filepath=train_file_path, training=True) # could fail if filepath is wrong

            test_file_path = self.get_filepath(SLEAP=True, data_type="Testing")

        except FileNotFoundError:
            train_file_path = self.get_filepath(SLEAP=False, data_type="Training") # try other filepath
            self.train_loader, self.pos_weight, self.n_samples = self._load_data(filepath=train_file_path, training=True)
            
            test_file_path = self.get_filepath(SLEAP=False, data_type="Testing")
        
        self.test_loader, _, _ = self._load_data(filepath=test_file_path, training=False)



    def get_filepath(self, SLEAP, data_type):
        
        if SLEAP:
            beginning = "SLEAP/"
        else:
            beginning = ""

        if data_type == "Training":
            ending = "train"
        elif data_type == "Testing":
            ending = "test"

        filepath = f"{beginning}Data/{DataSettings.DATASET}/{data_type}Data/{self.signal_type}_{ending}.npz"

        return filepath
        
    def _load_data(self, filepath, training):

        if self.verbose: 
            if training: print("Loading Training data")
            else: print("Loading Testing Data")

        with np.load(filepath) as data:
            X = (data['X']).astype(np.float32)
            y = data['y']

            if self.verbose: print("Data split.")

            loader, pos_weight, n_samples = self._prepare(X, y, training)

        del data
        gc.collect()

        return loader, pos_weight, n_samples

    def _prepare(self, X, y, training):
            
            X = np.expand_dims(X, 1)
            _, _, n_samples = X.shape

            X_tensor = torch.tensor(X)
            y = np.vectorize(self._get_stage_map().get)(y)
            y_tensor = torch.tensor(y)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            pos_weight = torch.tensor([(1 - y.mean()) / y.mean()]).to(device)

            dataset = TensorDataset(X_tensor, y_tensor)
            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=training,
                pin_memory=True
            )

            del X, y, X_tensor, y_tensor, dataset
            gc.collect()

            return loader, pos_weight, n_samples
    
    def _get_stage_map(self):
        STAGE_MAP = {
            CNN_BinaryClassifier.WAKE: 1 if self.sleepstage == Sleepstage.WAKE else 0,
            CNN_BinaryClassifier.LIGHT_SLEEP: 1 if self.sleepstage == Sleepstage.LIGHT_SLEEP else 0,
            CNN_BinaryClassifier.DEEP_SLEEP: 1 if self.sleepstage == Sleepstage.DEEP_SLEEP else 0,
            CNN_BinaryClassifier.REM: 1 if self.sleepstage == Sleepstage.REM else 0
        }

        return STAGE_MAP
    
    def get_random_subset(self):
        train_dataset = self.train_loader.dataset
        test_dataset = self.test_loader.dataset

        training_subset = sample(list(train_dataset), EvolutionSettings.DATA_POINTS_PER_INDIVIUAL)
        testing_subset = sample(list(test_dataset), EvolutionSettings.DATA_POINTS_PER_INDIVIUAL)

        train_loader_subset = DataLoader(training_subset, batch_size=self.batch_size, shuffle=True)
        test_loader_subset = DataLoader(testing_subset, batch_size=self.batch_size, shuffle=False)

        return train_loader_subset, test_loader_subset, self.n_samples, self.pos_weight
    
    def get_full_dataset(self):
        return self.train_loader, self.test_loader, self.n_samples, self.pos_weight