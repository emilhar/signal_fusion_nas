import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score
from Globals import device

class _Branch(nn.Module):
    def __init__(self, num_kernels, kernel_sizes, paddings, strides, pool_sizes, pool_strides, dropout_rates):
        super().__init__()
        layers = []
        in_channels = 1

        if len(num_kernels) != len(kernel_sizes):
            raise ValueError()
        if len(num_kernels) != len(paddings):
            raise ValueError()
        if len(num_kernels) != len(strides):
            raise ValueError()

        for i, (out_channels, k, p, s) in enumerate(zip(num_kernels, kernel_sizes, paddings, strides)):
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=p, stride=s, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU()
            ])
            in_channels = out_channels

            # add pooling and dropout after each block except the last
            if i < len(pool_sizes):
                layers.append(nn.MaxPool1d(kernel_size=pool_sizes[i], stride=pool_strides[i]))
                if dropout_rates[i] > 0: # -> 0 dropout rate? Don't add the layer ya dingus
                    layers.append(nn.Dropout(dropout_rates[i]))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class CNN_BinaryClassifier(nn.Module):

    def __init__(self, n_samples, branch_configs, batch_size):
        super().__init__()
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.branches = nn.ModuleDict()
        self.branch_output_sizes = {}

        for name, config in branch_configs.items():
            self.branches[name] = _Branch(**config)

        with torch.inference_mode():
            dummy = torch.zeros((self.batch_size, 1, n_samples))
            for name, branch in self.branches.items():
                branch.eval()
                out = branch(dummy)
                self.branch_output_sizes[name] = out.numel() // out.shape[0]
                branch.train()

        total_features = sum(self.branch_output_sizes.values())

        self.fc = nn.Sequential(
            nn.Linear(total_features, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.classifier = nn.Linear(32, 1)


    def forward(self, x):
        outputs = [branch(x).flatten(1) for branch in self.branches.values()]
        combined = torch.cat(outputs, dim=1)
        x = self.fc(combined)
        return self.classifier(x)


    @staticmethod
    def train_model(model, train_loader, test_loader, pos_weight, epochs=50):
        
        def _get_kernel_sizes(branch):
            kernel_sizes = []
            for layer in branch.net:
                if isinstance(layer, nn.Conv1d):
                    # layer.kernel_size is a tuple
                    kernel_sizes.append(layer.kernel_size[0])
            return kernel_sizes

        training_time_start = datetime.datetime.now()

        if device.type == "cpu":
            raise ("WARNING: Using CPU as device. This may take a while...")

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
        optimizer = optim.AdamW(model.parameters())
        #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=p, factor=f)
        best_f1, best_auc = 0.0, 0.0

        train_losses_data, test_losses_data = [], []
        best_state_dict = model.state_dict()

        best_true, best_scores = None, None

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device).float()
                optimizer.zero_grad()
                outputs = model(X_batch).squeeze(-1)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * X_batch.size(0)
                
            model.eval()
            test_loss = 0.0
            all_preds, all_targets, all_probs = [], [], []

            with torch.inference_mode():
                for X_batch, y_batch in test_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device).float()
                    outputs = model(X_batch).squeeze(-1)
                    loss = criterion(outputs, y_batch)
                    test_loss += loss.item() * X_batch.size(0)

                    probs = torch.sigmoid(outputs)
                    preds = probs > 0.5

                    all_probs.extend(probs.cpu().numpy().flatten())
                    all_preds.extend(preds.cpu().numpy().flatten())
                    all_targets.extend(y_batch.cpu().numpy().flatten())

            train_loss /= len(train_loader.dataset)
            test_loss /= len(test_loader.dataset)
            train_losses_data.append(train_loss)
            test_losses_data.append(test_loss)

            all_targets_np = np.array(all_targets)
            all_preds_np = np.array(all_preds)
            all_probs_np = np.array(all_probs)

            precision, recall, f1, _ = precision_recall_fscore_support(
                all_targets_np, all_preds_np, average='binary', zero_division=0
            )
            auc_score = roc_auc_score(all_targets_np, all_probs_np)
            accuracy = accuracy_score(all_targets_np, all_preds_np)
            #scheduler.step(auc_score)
            current_lr = optimizer.param_groups[0]['lr']

            kernel_sizes = []
            for branch in model.branches.values():
                kernel_sizes.append(_get_kernel_sizes(branch))
                    
            if f1 > best_f1:
                best_f1 = f1
                
            if auc_score > best_auc:
                best_auc = auc_score
                best_true = all_targets_np
                best_scores = all_probs_np
                best_state_dict = model.state_dict()
                

            elapsed = (datetime.datetime.now() - training_time_start).total_seconds()

        kernel_sizes = []
        for branch in model.branches.values():
            kernel_sizes.append(_get_kernel_sizes(branch))
        
        output = {
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "lr": current_lr,
            "branches": kernel_sizes,
            "best_f1": best_f1,
            # "best_auc": best_auc,
            # "best_true": best_true,
            # "best_scores": best_scores,
            "time": elapsed,
            "state_dict": best_state_dict,
            # "train_loss_history": train_losses_data,
            # "test_loss_history": test_losses_data,
        }
        
        return output
