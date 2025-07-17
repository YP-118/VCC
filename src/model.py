# src/model.py

import torch
import torch.nn as nn

class BaselineMLP(nn.Module):
    """
    Maps 5120-d ESM2 perturbation embeddings
    to 18080-d full gene expression profiles.

    Architecture:
        Input: 5120-d ESM2 embedding
        → Linear(5120 → 1024) + ReLU + Dropout
        → Linear(1024 → 2048) + ReLU + Dropout
        → Linear(2048 → 1024) + ReLU + Dropout
        → Linear(1024 → 18080) → Output
    """

    def __init__(self, input_dim=5120, output_dim=18080, dropout_rate=0.1):
        super(BaselineMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, output_dim)
        )

    def forward(self, x):
        return self.net(x)
