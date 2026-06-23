"""MLP predictor: flat overlap feature → binary logit."""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Two-or-more layer ReLU MLP.
    in_dim  : feature_dim() from SyntheticBenchmark
    hidden_dim : sweep over {64, 256, 1024}
    n_layers   : number of hidden layers (default 2)
    dropout    : applied after each hidden activation
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(prev, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = hidden_dim
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_dim) → logits: (B,)"""
        return self.net(x).squeeze(-1)
