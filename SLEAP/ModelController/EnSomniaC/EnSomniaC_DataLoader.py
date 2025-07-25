import torch
from torch.utils.data import Dataset, DataLoader
from EAController.LazyDataset import LazyDataset
import os
from Globals import Signal, DataManager

TRAIN_SPLIT = 0.8

class MultimodalDataset(Dataset):
    def __init__(self, data_dict, labels):
        self.length = len(labels)
        for signal, data in data_dict.items():
            assert len(data) == self.length, \
                f"Signal {signal} has {len(data)} samples, expected {self.length}"

        self.data_dict = data_dict
        self.labels = labels

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        dd = {}
        for signal, data in self.data_dict.items():
            dd[signal] = data[idx][0]

        return dd, self.labels[idx].type(torch.LongTensor)
    
def get_dataloaders_with_multimodal_datasets() -> tuple[DataLoader, DataLoader]:
    # Load data
    X_train, y_train, X_test, y_test = make_training_and_testing_data()

    # Make dataloaders
    train_load, test_load = create_dataloaders(X_train, y_train, X_test, y_test)

    return train_load, test_load

def make_training_and_testing_data():
    # Initialize dictionaries to hold all signals
    X_train = {signal: [] for signal in Signal.ALL_SIGNALS}
    X_test = {signal: [] for signal in Signal.ALL_SIGNALS}
    y_train = []
    y_test = []
    
    for signal in Signal.ALL_SIGNALS:
        data_dir = f"Data/{DataManager.DATASET}/{signal}"
        all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npz')])
        subject_ids = sorted(set(f[:5] for f in all_files))

        split_idx = int(len(subject_ids) * TRAIN_SPLIT)
        train_subjects = subject_ids[:split_idx]
        test_subjects = subject_ids[split_idx:]

        train_files = [f for f in all_files if f[:5] in train_subjects]
        test_files = [f for f in all_files if f[:5] in test_subjects]

        # Get data for current signal
        train_dataset = LazyDataset(train_files, data_dir, stage_map=None)
        test_dataset = LazyDataset(test_files, data_dir, stage_map=None)
        
        # Store datasets in dictionaries
        X_train[signal] = train_dataset
        X_test[signal] = test_dataset
        
        # For y, we only need to store once (assuming all signals have same y)
        if not y_train:
            # Get labels from first dataset (assuming all have same labels)
            y_train = [train_dataset[i][1].unsqueeze(0) for i in range(len(train_dataset))]
        if not y_test:
            y_test = [test_dataset[i][1].unsqueeze(0) for i in range(len(test_dataset))]


    # Convert lists to tensors
    y_train = torch.cat(y_train)
    y_test = torch.cat(y_test)


    return X_train, y_train, X_test, y_test

def create_dataloaders(X_train, y_train, X_test, y_test):
    # Create datasets
    train_dataset = MultimodalDataset(X_train, y_train)
    test_dataset = MultimodalDataset(X_test, y_test)

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
