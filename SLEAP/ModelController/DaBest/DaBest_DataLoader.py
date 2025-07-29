import torch
from torch.utils.data import Dataset, DataLoader
import os
from Globals import Signal, DataManager
from EAController.LazyDataset import LazyDataset

TRAIN_SPLIT = 0.8
MAX_MEMORY = 512

class MultimodalLazyDataset(Dataset):
    def __init__(self):
        self.data_dict = {}

    def __len__(self):
        if not self.data_dict:
            return 0
        
        first_key = list(self.data_dict.keys())[0]
        return len(self.data_dict[first_key])

    def add_data(self, files, signal, data_directory):
        if signal in self.data_dict:
            print(f"WARNING: Data from {signal} already in data dictionary")

        self.data_dict[signal] = LazyDataset(
            files, 
            data_directory, 
            stage_map=None, 
            max_memory = MAX_MEMORY // len(Signal.ALL_SIGNALS)
        )

    def __getitem__(self, idx):
        output_dict =  {}
        labels = set()
        for signal, lazy_data_loader in self.data_dict.items():
            x, y = lazy_data_loader[idx]
            output_dict[signal] = x
            labels.add(y.item())

            if len(labels) > 1:
                raise ValueError(f"Inconsistent labels for index {idx}: {labels}")

        return output_dict, labels.pop()
        
def get_dataloaders_with_multimodal_datasets() -> tuple[DataLoader, DataLoader]:
    # Load data
    train_dataset, test_dataset = make_training_and_testing_data()

    # Make dataloaders
    train_load, test_load = create_dataloaders(train_dataset, test_dataset)

    return train_load, test_load

def make_training_and_testing_data():
    
    # Initialize datasets to hold all signals
    train_dataset = MultimodalLazyDataset()
    test_dataset = MultimodalLazyDataset()
    
    for signal in Signal.ALL_SIGNALS:
        
        # Find and split data
        data_directory = f"Data/{DataManager.DATASET}/{signal}"
        all_files = sorted([f for f in os.listdir(data_directory) if f.endswith('.npz')])
        subject_ids = sorted(set(f[:5] for f in all_files))

        split_idx = int(len(subject_ids) * TRAIN_SPLIT)
        train_subjects = subject_ids[:split_idx]
        test_subjects = subject_ids[split_idx:]

        train_files = [f for f in all_files if f[:5] in train_subjects]
        test_files = [f for f in all_files if f[:5] in test_subjects]

        train_dataset.add_data(train_files, signal, data_directory)
        test_dataset.add_data(test_files, signal, data_directory)

    return train_dataset, test_dataset

def create_dataloaders(train_dataset, test_dataset):

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        pin_memory=True if torch.cuda.is_available() else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return train_loader, test_loader