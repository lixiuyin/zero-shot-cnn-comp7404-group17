"""
Text encoder ft(·): R^p -> R^k. Paper: "two-hidden layer fully-connected neural
network whose architecture is p-300-k" (p=text dim, k=50). We implement p-300-k
as one hidden layer (300); "two-hidden layer" is often read as two weight layers.
For conv (Sec 3.3), the 300-d hidden is used to predict conv filters.
"""
import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    def __init__(self,
                 input_dim: int = 9763,
                 hidden_dim: int = 300,
                 output_dim: int = 50):
        """
        Args:
            input_dim (p): dimensionality of the text feature vectors (paper: p).
            hidden_dim: hidden layer size; paper architecture p-300-k uses 300.
            output_dim (k): size of the predicted weight vector w_c for the FC layer (paper: k=50).
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        # Paper: p-300-k (input p, hidden 300, output k=50)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # Initialization strategy (best practices):
        # - fc1 (hidden layer): Use Kaiming Uniform (PyTorch default) - optimal for ReLU
        # - fc2 (weight prediction): Small initialization to prevent large predicted weights
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc2.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [*, p] -> [*, k]"""
        h = torch.relu(self.fc1(x))
        return self.fc2(h)

    def forward_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """x: [*, p] -> [*, hidden_dim]. Used to predict conv filters (Sec 3.3)."""
        return torch.relu(self.fc1(x))
