from torch.utils.data import DataLoader, Subset
import torch
import os
from random import sample, shuffle

from dataloaders.lazy_dataset import LazyDataset
from datahelpers.data import Data
from Globals import DATASET_PATH


class SDataLoader:
    TRAINING_SPLIT = 0.7
    def __init__(self, signal_type, classification_class, batch_size):
        self.signal_type = signal_type

        d = Data()
        self.stage_map = d.get_stage_map(classification_class)
        self.batch_size = batch_size

        self.train_loader, self.test_loader, self.n_samples, self.pos_weight = self.prepare_data()
    
    def prepare_data(self):

        data_dir = f"{DATASET_PATH}/{self.signal_type}"
        all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])
        #subject_ids = sorted(set(f[:5] for f in all_files))
        #shuffle(subject_ids)

        split_idx = int(len(all_files) * SDataLoader.TRAINING_SPLIT)
        #train_subjects = subject_ids[:split_idx]
        #test_subjects = subject_ids[split_idx:]

        train_files = all_files[:split_idx]
        test_files = all_files[split_idx:]

        train_dataset = LazyDataset(train_files, data_dir, self.stage_map, max_memory=Data.max_memory)
        test_dataset = LazyDataset(test_files, data_dir, self.stage_map, max_memory=Data.max_memory)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)

        n_samples = train_dataset[0][0].shape[1]
        
        # Estimate pos_weight using random sampling instead of full iteration
        pos_weight = self.estimate_pos_weight(train_dataset)
        
        return (
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

    def get_random_subset(self, datapoints_per_individual):
        train_dataset = self.train_loader.dataset
        test_dataset = self.test_loader.dataset

        # Calculate subset sizes
        train_size = int(datapoints_per_individual * SDataLoader.TRAINING_SPLIT)
        test_size = int(datapoints_per_individual * (1 - SDataLoader.TRAINING_SPLIT))
        
        train_subset = Subset(train_dataset, sample(range(len(train_dataset)), train_size))
        test_subset = Subset(test_dataset, sample(range(len(test_dataset)), test_size))
        
        return (
            DataLoader(train_subset, batch_size=self.batch_size, shuffle=True),
            DataLoader(test_subset, batch_size=self.batch_size, shuffle=False),
            self.n_samples,
            self.pos_weight
        )

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