import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, classification_report
from Globals import device

def train_model(model, train_loader, test_loader, epochs=5, lr=1e-4, wd=1e-4, class_names=None):

    print(f"Running for {epochs} epochs...")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

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

    return best_model_state