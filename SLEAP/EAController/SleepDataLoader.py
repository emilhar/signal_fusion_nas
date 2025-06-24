import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
import torch
import gc

from math import ceil

from ModelController.ModelMaker import CNN_BinaryClassifier
from Globals import Sleepstage, EvolutionSettings, DataSettings


class SleepDataLoader:
    def __init__(self, verbose, signal_type, sleepstage, batch_size):
        self.sleepstage = sleepstage
        self.signal_type = signal_type

        if DataSettings.DATASET == DataSettings.DatasetNames.TELEMETRY:
            self.signal_type = f"telemetry_{signal_type}"

        self.batch_size = batch_size

        self.verbose = verbose

        if verbose: print("Loading Training data")
        try:
            try_sleap=True
            train_file_path = self.get_filepath(SLEAP=try_sleap, data_type="Training")            
            self.train_loader, self.pos_weight, self.n_samples = self._load_data(filepath=train_file_path, training=True) # could fail if filepath is wrong

        except FileNotFoundError:
            try_sleap=False
            train_file_path = self.get_filepath(SLEAP=False, data_type="Training") # try other filepath
            self.train_loader, self.pos_weight, self.n_samples = self._load_data(filepath=train_file_path, training=True)
            
        if verbose: print("Loading Testing data")
        test_file_path = self.get_filepath(SLEAP=try_sleap, data_type="Testing")
        
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
        
        with np.load(filepath) as data:
            X = (data['X']).astype(np.float32)
            y = data['y']

            if self.verbose: print("Data split.")

            loader, pos_weight, n_samples = self._prepare(X, y, training)

        del data
        gc.collect()

        if self.verbose: print("Getting targets")

        if training:
            self.training_targets = [label for _, label in loader.dataset]
        else:
            self.testing_targets = [label for _, label in loader.dataset]

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

        if not EvolutionSettings.VALID_DATA_SPLIT:
            raise (ValueError, f"Invalid data split. {EvolutionSettings.DATA_SPLIT_TRAINING} + {EvolutionSettings.DATA_SPLIT_TESTING} != 1")

        training_subset = self.get_stratified_subset(train_dataset, training=True)
        testing_subset = self.get_stratified_subset(test_dataset, training=False)

        train_loader_subset = DataLoader(training_subset, batch_size=self.batch_size, shuffle=True)
        test_loader_subset = DataLoader(testing_subset, batch_size=self.batch_size, shuffle=False)

        return train_loader_subset, test_loader_subset, self.n_samples, self.pos_weight
    

    def get_stratified_subset(self, dataset, training:bool):

        if training:
            targets = self.training_targets
            data_amount = ceil(EvolutionSettings.DATA_POINTS_PER_INDIVIUAL * EvolutionSettings.DATA_SPLIT_TRAINING)
        else:
            targets = self.testing_targets
            data_amount = ceil(EvolutionSettings.DATA_POINTS_PER_INDIVIUAL * EvolutionSettings.DATA_SPLIT_TESTING)
    

        splitter = StratifiedShuffleSplit(test_size=data_amount)

        # Perform stratified split
        _, subset_idx = next(splitter.split(np.zeros(len(dataset)), targets))
        a =  Subset(dataset, subset_idx)

        #self.tfunc(a.dataset)

        return a
    
    def tfunc(self, dataset):
        labels = [dataset[i][1] for i in range(len(dataset))]  # Collect all y values

        s = {}
        ylen = len(labels)
        for label in labels:
            label = int(label)
            if label not in s:
                s[label] = 0
            s[label] += 1

        for label in s:
            print(f"{label}: {round(s[label]/ylen * 100, 2)}%")

    def get_full_dataset(self):
        return self.train_loader, self.test_loader, self.n_samples, self.pos_weight