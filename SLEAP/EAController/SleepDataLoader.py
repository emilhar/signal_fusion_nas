import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Subset
import torch
import gc
from random import sample
from math import ceil

from ModelController.ModelMaker import CNN_BinaryClassifier
from Globals import Sleepstage, ModelSettings, EvolutionSettings, DataSettings


class SleepDataLoader:
    def __init__(self, signal_type, sleepstage):
        self.sleepstage = sleepstage
        self.signal_type = signal_type

        if DataSettings.DATASET == DataSettings.DatasetNames.TELEMETRY:
            self.signal_type = f"telemetry_{signal_type}"

        self.batch_size = ModelSettings.BATCH_SIZE

        if ModelSettings.VERBOSE: print("Loading Training data")
        try:
            try_sleap=True
            train_file_path = self.get_filepath(SLEAP=try_sleap, data_type="Training")            
            self.train_loader, self.pos_weight, self.n_samples = self._load_data(filepath=train_file_path, training=True)

        except FileNotFoundError:
            try_sleap=False
            train_file_path = self.get_filepath(SLEAP=False, data_type="Training") # try other filepath
            self.train_loader, self.pos_weight, self.n_samples = self._load_data(filepath=train_file_path, training=True)
            
        if ModelSettings.VERBOSE: print("Loading Testing data")
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

            if ModelSettings.VERBOSE: print("Data split. Preparing data")

            loader, pos_weight, n_samples = self._prepare(X, y, training)
            
        if DataSettings.EVEN_DATA_SPLIT:

            if ModelSettings.VERBOSE: print("Preparing for even data split")
            if training:
                self.training_indices_class_0 = []
                self.training_indices_class_1 = []

                for i, (_, label) in enumerate(loader.dataset):
                    if label == 0:
                        self.training_indices_class_0.append(i)
                    elif label == 1:
                        self.training_indices_class_1.append(i)

            else:
                self.testing_indices_class_0 = []
                self.testing_indices_class_1 = []

                for i, (_, label) in enumerate(loader.dataset):
                    if label == 0:
                        self.testing_indices_class_0.append(i)
                    elif label == 1:
                        self.testing_indices_class_1.append(i)


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
            CNN_BinaryClassifier.N1: 1 if self.sleepstage == Sleepstage.N1 else 0,
            CNN_BinaryClassifier.N2: 1 if self.sleepstage == Sleepstage.N2 else 0,
            CNN_BinaryClassifier.N3: 1 if self.sleepstage == Sleepstage.N3 else 0,
            CNN_BinaryClassifier.REM: 1 if self.sleepstage == Sleepstage.REM else 0
        }

        return STAGE_MAP
    
    def get_random_subset(self):
        train_dataset = self.train_loader.dataset
        test_dataset = self.test_loader.dataset

        if not EvolutionSettings.VALID_DATA_SPLIT:
            raise (ValueError, f"Invalid data split. {EvolutionSettings.DATA_SPLIT_TRAINING} + {EvolutionSettings.DATA_SPLIT_TESTING} != 1")
        
        train_data_amount = ceil(EvolutionSettings.DATA_POINTS_PER_INDIVIUAL * EvolutionSettings.DATA_SPLIT_TRAINING)
        test_data_amount = ceil(EvolutionSettings.DATA_POINTS_PER_INDIVIUAL * EvolutionSettings.DATA_SPLIT_TESTING)

        if DataSettings.EVEN_DATA_SPLIT:
            training_subset = self.get_balanced_subset(train_dataset, total_data_points=train_data_amount, training=True)
            testing_subset = self.get_balanced_subset(test_dataset, total_data_points=test_data_amount, training=False)
        else:
            training_subset = sample(list(train_dataset), train_data_amount)
            testing_subset =sample(list(test_dataset), test_data_amount)
            
        train_loader_subset = DataLoader(training_subset, batch_size=self.batch_size, shuffle=True)
        test_loader_subset = DataLoader(testing_subset, batch_size=self.batch_size, shuffle=False)

        return train_loader_subset, test_loader_subset, self.n_samples, self.pos_weight

    def get_balanced_subset(self, dataset, total_data_points, training: bool):
        if training:
            indices_class_0 = self.training_indices_class_0
            indices_class_1 = self.training_indices_class_1
        else:
            indices_class_0 = self.testing_indices_class_0
            indices_class_1 = self.testing_indices_class_1

        samples_per_class = min(len(indices_class_0), len(indices_class_1), total_data_points // 2)

        if samples_per_class == 0:
            raise ValueError("Not enough samples in one or both classes to create a balanced subset.")

        subset_idx_class_0 = sample(indices_class_0, samples_per_class)
        subset_idx_class_1 = sample(indices_class_1, samples_per_class)

        combined_subset_idx = subset_idx_class_0 + subset_idx_class_1

        np.random.shuffle(combined_subset_idx)

        balanced_subset = Subset(dataset, combined_subset_idx)

        #self.see_dataset_breakdown(balanced_subset)

        return balanced_subset
    
    def see_dataset_breakdown(self, dataset):
        labels = [dataset[i][1] for i in range(len(dataset))]

        s = {}
        ylen = len(labels)
        for label in labels:
            label = int(label)
            if label not in s:
                s[label] = 0
            s[label] += 1

        for label in s:
            print(f"{label}: {round(s[label]/ylen * 100, 2)}%. {s[label]}")

    def get_full_dataset(self):
        return self.train_loader, self.test_loader, self.n_samples, self.pos_weight