import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support, classification_report
from Globals import Sleepstage, Signal
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
from ModelController.ModelMaker import CNN_BinaryClassifier

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
        return {signal: data[idx] for signal, data in self.data_dict.items()}, self.labels[idx]

class _FFE(nn.Module):
    """Frozen Feature Extractinator"""
    def __init__(self, model):
        super().__init__()
        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def forward(self, x):
        branch_outputs = [branch(x).flatten(1) for branch in self.model.branches.values()]
        combined = torch.cat(branch_outputs, dim=1)
        return self.model.fc(combined)

class EnsembleModel(nn.Module):
    def __init__(self, models, signal_names):
        super().__init__()
        self.signal_names = signal_names
        self.feature_extractors = nn.ModuleDict()
        
        for signal, model_list in models.items():
            self.feature_extractors[signal] = nn.ModuleList([
                _FFE(model) for model in model_list
            ])

        self.mlp = nn.Sequential(
            nn.Linear(20 * 32, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, len(Sleepstage.ALL_STAGES))
        )

    def forward(self, x_data):
        all_features = []
        for signal in self.signal_names:
            signal_data = x_data[signal]
            signal_models = self.feature_extractors[signal]
            for ffe in signal_models:
                features = ffe(signal_data)
                all_features.append(features)
        combined = torch.cat(all_features, dim=1)
        return self.mlp(combined)

class EnsembleModelMaker:
    def __init__(self, model_save_dir, train_data_path, test_data_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_save_dir = model_save_dir
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.signal_names = [
            Signal.EEG.Fpz_Cz.replace('_', ' '),
            Signal.EEG.Pz_Oz.replace('_', ' '),
            Signal.EOG.HORIZONTAL.replace('_', ' '),
            Signal.EMG.SUBMENTAL.replace('_', ' ')
        ]
        self.class_names = Sleepstage.ALL_STAGES

    def load_data(self):
        X_train, y_train, X_test, y_test = {}, None, {}, None
        
        with np.load(self.train_data_path) as train_data:
            for sig in self.signal_names:
                key = sig.replace(' ', '_')
                X_train[sig] = torch.tensor(train_data[key], dtype=torch.float32).unsqueeze(1)
            y_train = torch.tensor(train_data["y"], dtype=torch.long)

        with np.load(self.test_data_path) as test_data:
            for sig in self.signal_names:
                key = sig.replace(' ', '_')
                X_test[sig] = torch.tensor(test_data[key], dtype=torch.float32).unsqueeze(1)
            y_test = torch.tensor(test_data["y"], dtype=torch.long)
            
        return X_train, y_train, X_test, y_test

    def create_data_loaders(self, X_train, y_train, X_test, y_test, batch_size=32):
        train_dataset = MultimodalDataset(X_train, y_train)
        test_dataset = MultimodalDataset(X_test, y_test)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=torch.cuda.is_available()
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=torch.cuda.is_available()
        )
        return train_loader, test_loader

    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        model = CNN_BinaryClassifier(**checkpoint["model_args"])
        model.load_state_dict(checkpoint["state_dict"])
        return model

    def load_models(self):
        models_dict = {}
        stage_names = [stage.lower() for stage in Sleepstage.ALL_STAGES]
        
        for sig in self.signal_names:
            key = sig.replace(' ', '_')
            models_dict[sig] = [
                self.load_model(f"{self.model_save_dir}/telemetry_{key}_{stage}_classifier.pt")
                for stage in stage_names
            ]
        return models_dict

    def train_model(self, model, train_loader, test_loader, epochs=5, lr=5e-5, wd=1e-4):
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        best_test_f1 = 0.0

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            all_train_preds = []
            all_train_targets = []

            for batch in train_loader:
                x_dict, labels = batch
                x_dict = {key: val.to(self.device) for key, val in x_dict.items()}
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(x_dict).to(self.device)
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
                    x_dict = {key: val.to(self.device) for key, val in x_dict.items()}
                    labels = labels.to(self.device)

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

            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}, "
                f"Precision={train_precision:.4f}, Recall={train_recall:.4f}, F1={train_f1:.4f}")
            print(f"  Test:  Loss={test_loss:.4f}, Acc={test_acc:.4f}, "
                f"Precision={test_precision:.4f}, Recall={test_recall:.4f}, F1={test_f1:.4f}")

            print("\nTest Classification Report:")
            print(classification_report(
                all_test_targets,
                all_test_preds,
                target_names=self.class_names,
                digits=4,
                zero_division=0
                ))

        return model.state_dict()

    def plot_sample_predictions(self, model, data_dict, sample_idx):
        signals = {
            "EEG Fpz-Cz": data_dict["EEG Fpz-Cz"][sample_idx],
            "EEG Pz-Oz": data_dict["EEG Pz-Oz"][sample_idx],
            "EOG horizontal": data_dict["EOG horizontal"][sample_idx],
            "EMG submental": data_dict["EMG submental"][sample_idx],
        }

        if true_label is None:
            true_label = data_dict["y"][sample_idx]

        if pred_label is None:
            model.eval()
            with torch.inference_mode():
                input_dict = {}
                for ch, signal in signals.items():
                    input_dict[ch] = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

                output = model(input_dict)
                probs = torch.softmax(output, dim=1)
                pred_label = torch.argmax(probs, dim=1).item()

        fig, axs = plt.subplots(4, 1, figsize=(14, 10))
        fig.suptitle(f"Sample {sample_idx} - True: {self.class_names[true_label]}, Pred: {self.class_names[pred_label]}",
                    fontsize=16, fontweight='bold')

        for i, (ch, signal) in enumerate(signals.items()):
            axs[i].plot(signal, color='royalblue')
            axs[i].set_title(f"{ch} ({len(signal)} samples)", fontsize=12)
            axs[i].set_ylabel("Amplitude", fontsize=10)
            axs[i].grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

        return true_label, pred_label

    def analyze_predictions(self, model, data_dict):
        all_true = []
        all_preds = []

        model.eval()
        with torch.inference_mode():
            for i in range(len(data_dict["y"])):
                input_dict = {}
                for ch in ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal", "EMG submental"]: # TODO: why is this not a constant list teitur
                    signal = data_dict[ch][i]
                    input_dict[ch] = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

                true_label = data_dict["y"][i]

                output = model(input_dict)
                pred_label = torch.argmax(output, dim=1).item()

                all_true.append(true_label)
                all_preds.append(pred_label)

        all_true = np.array(all_true)
        all_preds = np.array(all_preds)

        cm = confusion_matrix(all_true, all_preds, labels=range(len(self.class_names)))

        plt.figure(figsize=(10, 8))

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=self.class_names
        )
        disp.plot(cmap="Blues", values_format="d")  # 'd' for integer formatting
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")

        plt.show()

        for class_idx, class_name in enumerate(self.class_names):
            print(f"\n===== {class_name} Analysis =====")

            # True Positives (TP): Correctly predicted as this class
            tp_indices = np.where((all_true == class_idx) & (all_preds == class_idx))[0]
            print(f"True Positives: {len(tp_indices)} samples")

            # False Positives (FP): Predicted as this class but actually not
            fp_indices = np.where((all_true != class_idx) & (all_preds == class_idx))[0]
            print(f"False Positives: {len(fp_indices)} samples")

            # False Negatives (FN): Actually this class but predicted as something else
            fn_indices = np.where((all_true == class_idx) & (all_preds != class_idx))[0]
            print(f"False Negatives: {len(fn_indices)} samples")

            # True Negatives (TN): Not this class and not predicted as that class
            tn_indices = np.where((all_true != class_idx) & (all_preds == all_true))[0]
            print(f"True Negatives: {len(tn_indices)} samples")

            if len(tp_indices) > 0:
                print("\nExample True Positive:")
                self.plot_sample_predictions(model, data_dict, tp_indices[np.random.choice(len(tp_indices))])
            else:
                print("Your model sux, NO TRUE POSITIVES")

            if len(fp_indices) > 0:
                print("\nExample False Positive:")
                self.plot_sample_predictions(model, data_dict, fp_indices[np.random.choice(len(fp_indices))])
            else:
                print("Wow, good model... maybe too good, NO FALSE POSITIVES")

            if len(fn_indices) > 0:
                print("\nExample False Negative:")
                self.plot_sample_predictions(model, data_dict, fn_indices[np.random.choice(len(fn_indices))])
            else:
                print("hm... NO FALSE NEGATIVES?")


            if len(tn_indices) > 0:
                print("\nExample True Negative:")
                self.plot_sample_predictions(model, data_dict, tn_indices[np.random.choice(len(tn_indices))])

            else:
                print("no true negatives? Bro get out")

    def main(self):
        # Load and prepare data
        X_train, y_train, X_test, y_test = self.load_data()
        train_loader, test_loader = self.create_data_loaders(X_train, y_train, X_test, y_test)
        
        # Load pre-trained models
        models_dict = self.load_models()
        
        # Build and train ensemble
        model = EnsembleModel(models_dict, self.signal_names)
        trained_state = self.train_model(model, train_loader, test_loader)
        model.load_state_dict(trained_state)
        
        # Evaluate
        test_dict = X_test.copy()
        test_dict['y'] = y_test
        self.analyze_predictions(model, test_dict)

# Example usage
if __name__ == "__main__":
    maker = EnsembleModelMaker(
        model_save_dir="/path/to/models",
        train_data_path="/path/to/train.npz",
        test_data_path="/path/to/test.npz"
    )
    maker.main()
