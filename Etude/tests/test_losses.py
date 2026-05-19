from __future__ import annotations

import torch

from etude.training.losses import behavior_cloning_loss, residual_loss, smoothness_loss


def test_losses_are_finite() -> None:
    pred = torch.zeros(2, 3)
    expert = torch.ones(2, 3)
    assert torch.isfinite(behavior_cloning_loss(pred, expert))
    assert torch.isfinite(residual_loss(pred, pred, expert))
    assert smoothness_loss(pred.unsqueeze(0)).item() == 0.0
