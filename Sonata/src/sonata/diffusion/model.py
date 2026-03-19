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


class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int, cond_dim: int, dilation: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(4, channels)
        self.norm2 = nn.GroupNorm(4, channels)
        self.cond_proj = nn.Linear(cond_dim, channels * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.cond_proj(cond).chunk(2, dim=-1)
        scale = scale.unsqueeze(-1)
        shift = shift.unsqueeze(-1)
        hidden = self.conv1(F.silu(self.norm1(x)))
        hidden = hidden * (1.0 + scale) + shift
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return x + hidden


class ConditionalTemporalDenoiser(nn.Module):
    def __init__(
        self,
        *,
        action_dim: int,
        prior_dim: int,
        global_cond_dim: int,
        model_dim: int,
        num_blocks: int,
    ) -> None:
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(model_dim)
        self.global_proj = nn.Sequential(
            nn.Linear(global_cond_dim + model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.input_proj = nn.Conv1d(action_dim + prior_dim, model_dim, kernel_size=1)
        self.blocks = nn.ModuleList(
            [ResidualBlock1D(model_dim, model_dim, dilation=2 ** (idx % 4)) for idx in range(num_blocks)]
        )
        self.output = nn.Sequential(
            nn.GroupNorm(4, model_dim),
            nn.SiLU(),
            nn.Conv1d(model_dim, action_dim, kernel_size=1),
        )

    def forward(self, noisy_action: torch.Tensor, prior: torch.Tensor, timestep: torch.Tensor, global_condition: torch.Tensor) -> torch.Tensor:
        cond = torch.cat([self.time_embed(timestep), global_condition], dim=-1)
        cond = self.global_proj(cond)
        hidden = torch.cat([noisy_action, prior], dim=-1).transpose(1, 2)
        hidden = self.input_proj(hidden)
        for block in self.blocks:
            hidden = block(hidden, cond)
        return self.output(hidden).transpose(1, 2)
