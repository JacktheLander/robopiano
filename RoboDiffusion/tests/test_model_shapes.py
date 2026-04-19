from __future__ import annotations

import torch

from robodiffusion.model.diffusion import DiffusionTransformer, GaussianDiffusionScheduler


def test_diffusion_transformer_and_sampler_shapes() -> None:
    model = DiffusionTransformer(
        score_dim=14,
        state_dim=32,
        action_dim=5,
        obs_horizon=8,
        pred_horizon=16,
        model_dim=64,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
    )
    scheduler = GaussianDiffusionScheduler(steps=8, beta_start=1e-4, beta_end=2e-2).to(torch.device("cpu"))
    score = torch.randn(2, 8, 14)
    state = torch.randn(2, 8, 32)
    clean = torch.randn(2, 16, 5)
    timesteps = scheduler.sample_timesteps(2, torch.device("cpu"))
    noise = torch.randn_like(clean)
    noisy = scheduler.q_sample(clean, timesteps, noise=noise)
    pred = model(noisy, score, state, timesteps)
    assert pred.shape == clean.shape
    sampled = scheduler.sample(
        model,
        score_window=score[:1],
        state_window=state[:1],
        action_shape=(1, 16, 5),
        device=torch.device("cpu"),
    )
    assert sampled.shape == (1, 16, 5)
