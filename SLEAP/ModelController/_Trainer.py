"""
This file contains only the 'train_model' function
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

import datetime # For max training time
from Globals import ModelSettings

def train_model(model, device, train_loader, test_loader, pos_weight, lr=2.5e-5, wd=1e-4, p=5, f=0.5, epochs=50, output_period=1, verbose=False, champion = False):

    training_time_start = datetime.datetime.now()

    if device.type == "cpu":
        raise ("WARNING: Using CPU as device. This may take a while...")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=p, factor=f)
    best_f1 = 0.0
    best_epoch = -1

    train_losses_data, test_losses_data = [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).float()
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
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
                outputs = model(X_batch).squeeze()
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

        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets_np, all_preds_np, average='binary', zero_division=0
        )

        accuracy = accuracy_score(all_targets_np, all_preds_np)
        scheduler.step(f1)
        current_lr = optimizer.param_groups[0]['lr']

        if verbose:
            if epoch % output_period == 0 or epoch == epochs-1:
                print(f"Epoch {epoch+1:2}/{epochs} -> "
                    f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:2.4f} | "
                    f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | "
                    f"Accuracy: {accuracy:.3f} ---> Learning rate: \x1b[31m{current_lr}\x1b[0m")

        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch+1

        if not champion:
            elapsed = (datetime.datetime.now() - training_time_start).total_seconds()
            if elapsed > ModelSettings.MAX_TIME_SPENT_TRAINING:
                if verbose:
                    print(f"Stopping training: elapsed time {elapsed:.1f}s > max {ModelSettings.MAX_TIME_SPENT_TRAINING}s")
                break


    output = {"Epoch": epoch,
            "Train Loss": train_loss,
            "Test Loss": test_loss,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Accuracy": accuracy,
            "Learning rate": current_lr}

    return output