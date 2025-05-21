import torch
import torch
import torch.nn as nn

class MLPClassifier(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=8, output_dim=3):
        super(MLPClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)
