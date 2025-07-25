import torch
import torch.nn as nn

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
