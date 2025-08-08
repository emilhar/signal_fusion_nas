import torch
import datetime
import numpy as np
import torch.nn as nn
from Globals import device
import torch.optim as optim
from datahelpers.data import Data
from sklearn.metrics import precision_recall_fscore_support

class _FFE(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def forward(self, x):
        branch_outputs = [branch(x).flatten(1) for branch in self.model.branches.values()]
        combined = torch.cat(branch_outputs, dim=1)
        return self.model.fc(combined) # -> (batch_size, 32)


class EnsembleModel(nn.Module):
    WEIGHTS = None
    def __init__(self, models):
        super().__init__()
        self.signal_names = models.keys()

        self.feature_extractors = nn.ModuleDict()
        for signal, model_list in models.items():
            self.feature_extractors[signal] = nn.ModuleList([
                _FFE(model) for model in model_list
            ])

        assert Data.get_all_signal_names() is not None, "Haven't initialized Data"
        assert Data.get_all_target_names() is not None, "Haven't initialized Data"

        self.mlp = nn.Sequential(
            nn.Linear(len(Data.get_all_signal_names()) * len(Data.get_all_target_names()) * 32, 256), # (N binary models each outputting a 32 embedding) -> 256
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, len(Data.get_all_target_names()))
        )

    def forward(self, x_data):
        """
        x_data: Dictionary of tensors {signal_name: (batch, 1, n_samples)}
        Returns:
            logits: (batch, 5) sleep stage logits
        """
        all_features = []

        for signal in self.signal_names:
            signal_data = x_data[signal]
            signal_models = self.feature_extractors[signal]

            for ffe in signal_models:
                features = ffe(signal_data)
                all_features.append(features)

        combined = torch.cat(all_features, dim=1)

        return self.mlp(combined)
    

    @staticmethod
    def calculate_class_weights(train_loader):
        
        if EnsembleModel.WEIGHTS is not None:
            return EnsembleModel.WEIGHTS

        class_counts = {}
        for _, labels in train_loader:
            labels = labels.cpu().numpy()
            for label in labels:
                if label in class_counts:
                    class_counts[label] += 1
                else:
                    class_counts[label] = 1
        
        # Sort classes and get counts in order
        classes = sorted(class_counts.keys())
        counts = [class_counts[c] for c in classes]
        
        weights = 1. / torch.tensor(counts, dtype=torch.float32)
        weights = weights / weights.sum()
        weights = weights.to(device)
        EnsembleModel.WEIGHTS = weights
        return weights.to(device)

    @staticmethod
    def train_model(model, train_loader, test_loader, epochs=5):
        training_time_start = datetime.datetime.now()

        print(f"Running for {epochs} epochs...")
        model = model.to(device)
        if device.type == "cpu":
            raise ValueError("Training with CPU")
        
        # Calculate weights if not provided
        weights = EnsembleModel.calculate_class_weights(train_loader)
        print(f"Automatically calculated class weights: {weights.cpu().numpy()}")
        
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = optim.AdamW(model.parameters())

        best_test_f1 = 0.0
        best_model_state = model.state_dict()

        d = Data()
        class_names = [t.given_name for t in d.target_objects]

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            all_train_preds = []
            all_train_targets = []

            for batch in train_loader:
                x_dict, labels = batch
                x_dict = {key: val.to(device) for key, val in x_dict.items()}
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(x_dict).to(device)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * labels.size(0)

                _, preds = torch.max(outputs, 1)
                all_train_preds.extend(preds.cpu().numpy())
                all_train_targets.extend(labels.cpu().numpy())

            train_loss /= len(train_loader.dataset)
            train_acc = np.mean(np.array(all_train_preds) == np.array(all_train_targets))
            train_precision, train_recall, train_f1, train_support = precision_recall_fscore_support(
                all_train_targets, all_train_preds, zero_division=0
            )

            model.eval()
            test_loss = 0.0
            all_test_preds = []
            all_test_targets = []
            all_test_probs = []

            with torch.inference_mode():
                for batch in test_loader:
                    x_dict, labels = batch
                    x_dict = {key: val.to(device) for key, val in x_dict.items()}
                    labels = labels.to(device)

                    outputs = model(x_dict)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item() * labels.size(0)

                    probs = torch.softmax(outputs, dim=1)
                    _, preds = torch.max(outputs, 1)

                    all_test_preds.extend(preds.cpu().numpy())
                    all_test_targets.extend(labels.cpu().numpy())
                    all_test_probs.extend(probs.cpu().numpy())

            test_loss /= len(test_loader.dataset)
            test_acc = np.mean(np.array(all_test_preds) == np.array(all_test_targets))
            test_precision, test_recall, test_f1, test_support = precision_recall_fscore_support(
                all_test_targets, all_test_preds, zero_division=0
            )

            # DOES NOT WORK.
            # if test_f1 > best_test_f1:
            #     best_test_f1 = test_f1
            #     best_model_state = model.state_dict()

            elapsed = (datetime.datetime.now() - training_time_start).total_seconds()

            print("\nTraining Per-Class Metrics:")
            for i, (prec, rec, f1, supp) in enumerate(zip(train_precision, train_recall, train_f1, train_support)):
                print(f"Class {class_names[i]} ({i}):")
                print(f"  Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, Support: {supp}")

            print("\nTest Per-Class Metrics:")
            for i, (prec, rec, f1, supp) in enumerate(zip(test_precision, test_recall, test_f1, test_support)):
                print(f"Class {class_names[i]} ({i}):")
                print(f"  Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, Support: {supp}")

        return model.state_dict()

