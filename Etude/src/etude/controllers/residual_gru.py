from __future__ import annotations

import torch
from torch import nn


class ResidualGRU(nn.Module):
    """Temporal residual controller: MLP encoder -> GRU -> action head."""

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        encoder_dim: int = 128,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, encoder_dim), nn.GELU())
        self.gru = nn.GRU(
            input_size=encoder_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, action_dim)

    def forward(
        self, features: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(features)
        output, hidden_out = self.gru(encoded, hidden)
        return self.head(output), hidden_out
