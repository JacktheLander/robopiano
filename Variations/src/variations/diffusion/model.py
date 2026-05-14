from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        exponent = torch.arange(half, device=timestep.device, dtype=torch.float32) / max(half - 1, 1)
        freqs = torch.exp(-math.log(10_000.0) * exponent)
        args = timestep.float().unsqueeze(-1) * freqs.unsqueeze(0)
        embedding = torch.cat([args.sin(), args.cos()], dim=-1)
        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class FiLMBlock(nn.Module):
    def __init__(self, hidden_dim: int, cond_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.film = nn.Linear(cond_dim, hidden_dim * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.film(cond).chunk(2, dim=-1)
        hidden = self.norm(x)
        hidden = hidden * (1.0 + gamma) + beta
        return x + self.net(hidden)


class VariationsDenoiser(nn.Module):
    def __init__(
        self,
        *,
        target_dim: int = 88,
        hand_dim: int = 46,
        hidden_dim: int = 256,
        num_blocks: int = 6,
        time_dim: int = 128,
        cond_dim: int = 256,
    ) -> None:
        super().__init__()
        self.target_dim = target_dim
        self.hand_dim = hand_dim
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.cond_encoder = nn.Sequential(
            nn.Linear(target_dim, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.input_proj = nn.Linear(hand_dim, hidden_dim)
        self.blocks = nn.ModuleList([FiLMBlock(hidden_dim, cond_dim) for _ in range(num_blocks)])
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hand_dim),
        )

    def forward(self, x_noisy: torch.Tensor, timestep: torch.Tensor, target_keys: torch.Tensor) -> torch.Tensor:
        cond = self.cond_encoder(target_keys.float()) + self.time_embed(timestep)
        hidden = self.input_proj(x_noisy)
        for block in self.blocks:
            hidden = block(hidden, cond)
        return self.output(hidden)
