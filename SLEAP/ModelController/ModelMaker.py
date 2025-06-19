"""
This file defines the CNN binary classifier model
"""

import torch
import torch.nn as nn

class _Branch(nn.Module):
    def __init__(self, num_kernels, kernel_sizes, paddings, strides, pool_sizes, pool_strides, dropout_rates):
        super().__init__()
        layers = []
        in_channels = 1

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
    """
    Two-branched convolutional neural network for binary sleep classification.
    Input is arbitrary, whether it be EEG, EOG or EMG, even something else entirely.

    An example of a model predicting N3 with 3000 samples as input:
    X -> 3000 samples of whatever data
    y -> 0 or 1
      0 -> NOT N3
      1 -> N3

    branch_configs = {
      "left": {
          "num_kernels": [32, 64, 64],
          "kernel_sizes": [22, 8, 8],
          "paddings": [22//2, 3, 3],
          "strides": [6, 1, 1],
          "pool_sizes": [8, 4],
          "pool_strides": [8, 4],
          "dropout_rates": [0.1, 0.0]  # dropout only after first pool
        },
      "right": {
          "num_kernels": [32, 64, 64],
          "kernel_sizes": [400, 6, 6],
          "paddings": [175, 2, 2],
          "strides": [50, 1, 1],
          "pool_sizes": [4, 2],
          "pool_strides": [4, 2],
          "dropout_rates": [0.1, 0.0]  # dropout only after first pool
        }
    }

    model_args = {
        "name": "MyN3Classifier",
        "n_samples": 3000,
        "branch_configs": branch_configs
    }

    model = SleepstageClassifier(**model_args)
    """
    WAKE = 0
    LIGHT_SLEEP = 1
    DEEP_SLEEP = 2
    REM = 3

    def __init__(self, name, n_samples, branch_configs):
        super().__init__()
        self.name = name
        self.branches = nn.ModuleDict()
        self.branch_output_sizes = {}

        for name, config in branch_configs.items():
            self.branches[name] = _Branch(**config)

        # output sizes using dummy input
        with torch.inference_mode():
            dummy = torch.zeros(1, 1, n_samples)
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
            nn.ReLU(),
        )

        self.classifier = nn.Linear(32, 1)

    def forward(self, x):
        outputs = [branch(x).flatten(1) for branch in self.branches.values()]
        combined = torch.cat(outputs, dim=1)
        x = self.fc(combined)
        return self.classifier(x)