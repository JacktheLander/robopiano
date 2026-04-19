from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half_dim = dim // 2
    exponent = -math.log(10000.0) / max(half_dim - 1, 1)
    frequencies = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * exponent)
    angles = timesteps.float()[:, None] * frequencies[None, :]
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        *,
        score_dim: int,
        state_dim: int,
        action_dim: int,
        obs_horizon: int,
        pred_horizon: int,
        model_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.score_dim = int(score_dim)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.obs_horizon = int(obs_horizon)
        self.pred_horizon = int(pred_horizon)
        self.model_dim = int(model_dim)
        self.score_proj = nn.Linear(self.score_dim, self.model_dim)
        self.state_proj = nn.Linear(self.state_dim, self.model_dim)
        self.obs_fuse = nn.Linear(self.model_dim * 2, self.model_dim)
        self.action_proj = nn.Linear(self.action_dim, self.model_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim * 4),
            nn.SiLU(),
            nn.Linear(self.model_dim * 4, self.model_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=int(num_heads),
            dim_feedforward=self.model_dim * 4,
            dropout=float(dropout),
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers))
        self.position_embedding = nn.Parameter(torch.zeros(1, 1 + self.obs_horizon + self.pred_horizon, self.model_dim))
        self.output_proj = nn.Linear(self.model_dim, self.action_dim)

    def forward(
        self,
        noisy_actions: torch.Tensor,
        score_window: torch.Tensor,
        state_window: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        score_tokens = self.score_proj(score_window)
        state_tokens = self.state_proj(state_window)
        obs_tokens = self.obs_fuse(torch.cat([score_tokens, state_tokens], dim=-1))
        action_tokens = self.action_proj(noisy_actions)
        time_token = self.time_proj(sinusoidal_embedding(timesteps, self.model_dim)).unsqueeze(1)
        tokens = torch.cat([time_token, obs_tokens, action_tokens], dim=1)
        tokens = tokens + self.position_embedding[:, : tokens.shape[1]]
        encoded = self.transformer(tokens)
        action_encoded = encoded[:, -self.pred_horizon :, :]
        return self.output_proj(action_encoded)


@dataclass
class DiffusionConfig:
    score_dim: int
    state_dim: int
    action_dim: int
    obs_horizon: int
    pred_horizon: int
    model_dim: int
    num_layers: int
    num_heads: int
    dropout: float
    diffusion_steps: int
    beta_start: float
    beta_end: float

    def to_payload(self) -> dict[str, Any]:
        return {
            "score_dim": int(self.score_dim),
            "state_dim": int(self.state_dim),
            "action_dim": int(self.action_dim),
            "obs_horizon": int(self.obs_horizon),
            "pred_horizon": int(self.pred_horizon),
            "model_dim": int(self.model_dim),
            "num_layers": int(self.num_layers),
            "num_heads": int(self.num_heads),
            "dropout": float(self.dropout),
            "diffusion_steps": int(self.diffusion_steps),
            "beta_start": float(self.beta_start),
            "beta_end": float(self.beta_end),
        }


class GaussianDiffusionScheduler:
    def __init__(self, steps: int, beta_start: float, beta_end: float) -> None:
        self.steps = int(steps)
        self.betas = torch.linspace(float(beta_start), float(beta_end), self.steps, dtype=torch.float32)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alpha_cumprod_prev = torch.cat([torch.ones(1), self.alpha_cumprod[:-1]], dim=0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)
        self.posterior_variance = self.betas * (1.0 - self.alpha_cumprod_prev) / (1.0 - self.alpha_cumprod)

    def to(self, device: torch.device) -> "GaussianDiffusionScheduler":
        for name in (
            "betas",
            "alphas",
            "alpha_cumprod",
            "alpha_cumprod_prev",
            "sqrt_alpha_cumprod",
            "sqrt_one_minus_alpha_cumprod",
            "posterior_variance",
        ):
            setattr(self, name, getattr(self, name).to(device))
        return self

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.steps, (batch_size,), device=device, dtype=torch.long)

    def q_sample(self, clean_actions: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(clean_actions)
        return self._extract(self.sqrt_alpha_cumprod, timesteps, clean_actions.shape) * clean_actions + self._extract(
            self.sqrt_one_minus_alpha_cumprod, timesteps, clean_actions.shape
        ) * noise

    def predict_x0(self, noisy_actions: torch.Tensor, timesteps: torch.Tensor, predicted_noise: torch.Tensor) -> torch.Tensor:
        return (
            noisy_actions
            - self._extract(self.sqrt_one_minus_alpha_cumprod, timesteps, noisy_actions.shape) * predicted_noise
        ) / self._extract(self.sqrt_alpha_cumprod, timesteps, noisy_actions.shape)

    @torch.no_grad()
    def sample(
        self,
        model: DiffusionTransformer,
        *,
        score_window: torch.Tensor,
        state_window: torch.Tensor,
        action_shape: tuple[int, int, int],
        device: torch.device,
        initial_sample: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sample = initial_sample.to(device) if initial_sample is not None else torch.randn(action_shape, device=device)
        for step in reversed(range(self.steps)):
            timesteps = torch.full((action_shape[0],), step, device=device, dtype=torch.long)
            predicted_noise = model(sample, score_window, state_window, timesteps)
            beta_t = self._extract(self.betas, timesteps, sample.shape)
            alpha_t = self._extract(self.alphas, timesteps, sample.shape)
            alpha_bar_t = self._extract(self.alpha_cumprod, timesteps, sample.shape)
            alpha_bar_prev = self._extract(self.alpha_cumprod_prev, timesteps, sample.shape)
            x0 = self.predict_x0(sample, timesteps, predicted_noise).clamp(-1.0, 1.0)
            mean = (
                torch.sqrt(alpha_bar_prev) * beta_t / (1.0 - alpha_bar_t) * x0
                + torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t) * sample
            )
            if step > 0:
                noise = torch.randn_like(sample)
                variance = torch.sqrt(self._extract(self.posterior_variance, timesteps, sample.shape).clamp(min=1e-8))
                sample = mean + variance * noise
            else:
                sample = mean
        return sample

    @staticmethod
    def _extract(buffer: torch.Tensor, timesteps: torch.Tensor, target_shape: torch.Size | tuple[int, ...]) -> torch.Tensor:
        values = buffer.gather(0, timesteps)
        while values.ndim < len(target_shape):
            values = values.unsqueeze(-1)
        return values
