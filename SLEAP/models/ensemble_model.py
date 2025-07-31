import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support, classification_report
import numpy as np

from Globals import device

class _FFE(nn.Module):
    """Frozen Feature Extractinator
    Freezes the Features and then Extractinates them
    """
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
    def __init__(self, models):
        """
        models = {
            "s1": [m1, m2, m3, m4],
            ...
            "s4": [m13, m14, m15, m16]
        }
        """
        super().__init__()
        self.signal_names = models.keys()

        self.feature_extractors = nn.ModuleDict()
        for signal, model_list in models.items():
            self.feature_extractors[signal] = nn.ModuleList([
                _FFE(model) for model in model_list
            ])


        self.mlp = nn.Sequential(
            nn.Linear(20 * 32, 256), # (20 binary each outputting a 32 embedding) -> 256
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 5)  # 5 sleep stages
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
    def train_model(model, train_loader, test_loader, epochs=5, lr=1e-4, wd=1e-4, class_names=None):
        training_time_start = datetime.datetime.now()

        print(f"Running for {epochs} epochs...")
        model = model.to(device)
        if device.type == "cpu":
            raise ValueError("Training with CPU")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters())

        best_test_f1 = 0.0
        best_model_state = model.state_dict()

        if class_names is None:
            class_names = [f"Class {i}" for i in range(5)]

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
            train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
                all_train_targets, all_train_preds, average='weighted', zero_division=0
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
            test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
                all_test_targets, all_test_preds, average='weighted', zero_division=0
            )

            if test_f1 > best_test_f1:
                best_test_f1 = test_f1
                best_model_state = model.state_dict()


            elapsed = (datetime.datetime.now() - training_time_start).total_seconds()

            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}, "
                f"Precision={train_precision:.4f}, Recall={train_recall:.4f}, F1={train_f1:.4f}")
            print(f"  Test:  Loss={test_loss:.4f}, Acc={test_acc:.4f}, "
                f"Precision={test_precision:.4f}, Recall={test_recall:.4f}, F1={test_f1:.4f}")

            print("\nTest Classification Report:")
            print(classification_report(
                all_test_targets,
                all_test_preds,
                target_names=class_names,
                digits=4,
                zero_division=0
                ))
            
            print(f"\nCumulative run time: {elapsed:.4f} seconds")

        return model.state_dict()