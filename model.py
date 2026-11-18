import torch
from torch import nn

class ChurnANN(nn.Module):
    """Feed-forward ANN for binary customer-churn classification."""
    def __init__(self, input_dim: int, hidden1: int = 64, hidden2: int = 32, dropout: float = 0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1)  # logits; apply sigmoid only for probabilities
        )

    def forward(self, x):
        return self.network(x)
