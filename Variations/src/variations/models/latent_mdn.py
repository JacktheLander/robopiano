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


class LatentMDN(nn.Module):
    """
    Predicts a Gaussian mixture distribution over latent press-pose codes.

    Input:
        target_keys: (B, 88)

    Output:
        mixture_logits: (B, K)
        means:          (B, K, latent_dim)
        log_stds:       (B, K, latent_dim)
    """

    def __init__(
        self,
        input_dim: int = 88,
        latent_dim: int = 12,
        num_components: int = 5,
        hidden_dim: int = 512,
        dropout: float = 0.05,
        min_log_std: float = -6.0,
        max_log_std: float = 2.0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.num_components = int(num_components)
        self.min_log_std = float(min_log_std)
        self.max_log_std = float(max_log_std)
        self.trunk = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.GELU(),
            ResidualMLPBlock(256, dropout=dropout),
            nn.Linear(256, int(hidden_dim)),
            nn.GELU(),
            ResidualMLPBlock(int(hidden_dim), dropout=dropout),
            ResidualMLPBlock(int(hidden_dim), dropout=dropout),
            nn.Linear(int(hidden_dim), 256),
            nn.GELU(),
        )
        self.logits_head = nn.Linear(256, self.num_components)
        self.means_head = nn.Linear(256, self.num_components * self.latent_dim)
        self.log_stds_head = nn.Linear(256, self.num_components * self.latent_dim)

    def forward(self, target_keys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.trunk(target_keys.float())
        mixture_logits = self.logits_head(h)
        means = self.means_head(h).view(-1, self.num_components, self.latent_dim)
        log_stds = self.log_stds_head(h).view(-1, self.num_components, self.latent_dim)
        log_stds = torch.clamp(log_stds, self.min_log_std, self.max_log_std)
        return mixture_logits, means, log_stds


def build_latent_mdn(config: dict) -> LatentMDN:
    cfg = dict(config.get("mdn", {}))
    cfg.pop("training", None)
    cfg.pop("loss", None)
    return LatentMDN(**cfg)

