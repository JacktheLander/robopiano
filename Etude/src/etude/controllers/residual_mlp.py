from __future__ import annotations

import torch
from torch import nn


class ResidualMLP(nn.Module):
    """Small MLP used for behavior cloning or PD residual prediction."""

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.05,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        act_cls = nn.GELU if activation == "gelu" else nn.ReLU
        layers: list[nn.Module] = []
        dim = input_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(dim, hidden_dim), act_cls()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = hidden_dim
        layers.append(nn.Linear(dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)
