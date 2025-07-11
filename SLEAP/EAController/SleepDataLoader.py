import numpy as np
from torch.utils.data import DataLoader, Subset
import torch
import os
from random import sample, shuffle
from math import ceil

from ModelController.ModelMaker import CNN_BinaryClassifier
from EAController.SleepEDF20LazyDataset import SleepEDF20LazyDataset
from Globals import Sleepstage, ModelManager, EvolutionManager, DataManager, SLEAPyException


class SleepDataLoader:
    def __init__(self, signal_type, sleepstage):
        self.sleepstage = sleepstage
        self.signal_type = signal_type
        self.stage_map = self._get_stage_map()

        self.batch_size = ModelManager.BATCH_SIZE

        if EvolutionManager.VERBOSE: print("Loading Data")
        self.train_loader, self.test_loader, self.n_samples, self.pos_weight = self.prepare_data()

    def _get_stage_map(self):
        STAGE_MAP = {
            CNN_BinaryClassifier.WAKE: 1 if self.sleepstage == Sleepstage.WAKE else 0,
            CNN_BinaryClassifier.N1: 1 if self.sleepstage == Sleepstage.N1 else 0,
            CNN_BinaryClassifier.N2: 1 if self.sleepstage == Sleepstage.N2 else 0,
            CNN_BinaryClassifier.N3: 1 if self.sleepstage == Sleepstage.N3 else 0,
            CNN_BinaryClassifier.REM: 1 if self.sleepstage == Sleepstage.REM else 0
        }

        return STAGE_MAP
    
    def prepare_data(self):

        data_dir = f"Data/{DataManager.DATASET}/{self.signal_type}"
        all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])
        subject_ids = sorted(set(f[:6] for f in all_files))
        shuffle(subject_ids)

        split_idx = int(len(subject_ids) * EvolutionManager.DATA_SPLIT_TRAINING)
        train_subjects = subject_ids[:split_idx]
        test_subjects = subject_ids[split_idx:]

        train_files = [f for f in all_files if f[:6] in train_subjects]
        test_files = [f for f in all_files if f[:6] in test_subjects]

        stage_map = self._get_stage_map()

        train_dataset = SleepEDF20LazyDataset(train_files, data_dir, stage_map)
        test_dataset = SleepEDF20LazyDataset(test_files, data_dir, stage_map)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)


        # data_dir = f"Data/{DataManager.DATASET}/{self.signal_type}"
        # x_files = sorted([f for f in os.listdir(data_dir) if f.endswith('_x.npy')])
        # y_files = sorted([f for f in os.listdir(data_dir) if f.endswith('_y.npy')])

        # if not (x_files and y_files):
        #     raise SLEAPyException(msg=f"No data inside Data/{DataManager.DATASET}/{self.signal_type}")
        
        # subject_ids = sorted(set(f[:6] for f in x_files))
        # shuffle(subject_ids)

        # split_idx = int(len(subject_ids) * EvolutionManager.DATA_SPLIT_TRAINING)
        # train_subjects = subject_ids[:split_idx]
        # test_subjects = subject_ids[split_idx:]

        # x_train_files = [f for f in x_files if f[:6] in train_subjects]
        # y_train_files = [f for f in y_files if f[:6] in train_subjects]
        # x_test_files = [f for f in x_files if f[:6] in test_subjects]
        # y_test_files = [f for f in y_files if f[:6] in test_subjects]

        # train_dataset = SleepEDF20LazyDataset(x_train_files, y_train_files, data_dir, self.stage_map)
        # test_dataset = SleepEDF20LazyDataset(x_test_files, y_test_files, data_dir, self.stage_map)

        # # During dataset initialization
        # self.train_indices = list(range(len(train_dataset)))
        # self.test_indices = list(range(len(test_dataset)))
        
        # # Only get first sample for shape info
        n_samples = train_dataset[0][0].shape[1]
        
        # # Estimate pos_weight using random sampling instead of full iteration
        pos_weight = self.estimate_pos_weight(train_dataset)
        
        return (
            #DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True),
            #DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True),
            train_loader,
            test_loader,
            n_samples,
            pos_weight
        )

    def estimate_pos_weight(self, dataset, sample_size=None):
        """Estimate positive weight using random sampling"""
        if not sample_size:
            sample_size = len(dataset)

        label_counts = {0: 0, 1: 0}
        indices = sample(range(len(dataset)), min(sample_size, len(dataset)))
        
        for i in indices:
            _, label = dataset[i]
            label_counts[int(label)] += 1
            
        return torch.tensor([label_counts[0] / max(1, label_counts[1])])

    def get_random_subset(self, dataset_percentage, batch_size):
        train_dataset = self.train_loader.dataset
        test_dataset = self.test_loader.dataset

        # Calculate subset sizes
        train_size = int(len(train_dataset) * dataset_percentage * EvolutionManager.DATA_SPLIT_TRAINING)
        test_size = int(len(test_dataset) * dataset_percentage * EvolutionManager.DATA_SPLIT_TESTING)
        
        # Create index-based subsets
        if DataManager.EVEN_DATA_SPLIT:
            train_subset = self.create_balanced_subset(train_dataset, train_size)
            test_subset = self.create_balanced_subset(test_dataset, test_size)
        else:
            train_subset = Subset(train_dataset, sample(range(len(train_dataset)), train_size))
            test_subset = Subset(test_dataset, sample(range(len(test_dataset)), test_size))
        


        return (
            DataLoader(train_subset, batch_size=batch_size, shuffle=True),
            DataLoader(test_subset, batch_size=batch_size, shuffle=False),
            self.n_samples,
            self.pos_weight
        )

    def create_balanced_subset(self, dataset, size):
        """Create balanced subset without full dataset iteration"""
        # Use precomputed indices if available
        indices = self.train_indices if dataset == self.train_loader.dataset else self.test_indices
        
        # Get random batch of indices
        batch_indices = sample(indices, min(10000, len(indices)))

        sample_size = min(10000, len(dataset))
        sampled_indices = sample(range(len(dataset)), sample_size)
        
        for idx in sampled_indices:
            _, label = dataset[idx]
            batch_indices[int(label)].append(idx)
        
        # Determine samples per class
        n_per_class = size // 2
        selected = []
        
        # Sample from each class
        for label in [0, 1]:
            if len(batch_indices[label]) > n_per_class:
                selected.extend(sample(batch_indices[label], n_per_class))
            else:
                selected.extend(batch_indices[label])
        
        return Subset(dataset, selected)

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