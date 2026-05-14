from __future__ import annotations

import torch
import torch.nn as nn


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, hidden_mult: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * int(hidden_mult)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(dim * int(hidden_mult), dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class PoseAutoencoder(nn.Module):
    """
    Learns a compressed latent representation of valid press poses.

    Input:
        joint_state: (B, input_dim)

    Output:
        recon_joint_state: (B, input_dim)
        z: (B, latent_dim)
    """

    def __init__(self, input_dim: int = 46, latent_dim: int = 12, dropout: float = 0.05) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.GELU(),
            ResidualMLPBlock(256, dropout=dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            ResidualMLPBlock(128, dropout=dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, self.latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            ResidualMLPBlock(128, dropout=dropout),
            nn.Linear(128, 256),
            nn.GELU(),
            ResidualMLPBlock(256, dropout=dropout),
            nn.Linear(256, self.input_dim),
        )

    def encode(self, hand_state: torch.Tensor) -> torch.Tensor:
        return self.encoder(hand_state.float())

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z.float())

    def forward(self, hand_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(hand_state)
        recon = self.decode(z)
        return recon, z


def build_pose_autoencoder(config: dict) -> PoseAutoencoder:
    cfg = dict(config.get("autoencoder", {}))
    cfg.pop("training", None)
    cfg.pop("loss", None)
    return PoseAutoencoder(**cfg)
