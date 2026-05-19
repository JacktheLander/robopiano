from __future__ import annotations

import torch

from etude.training.losses import residual_loss, smoothness_loss


def residual_objective(
    u_pd: torch.Tensor,
    delta_u_pred: torch.Tensor,
    u_expert: torch.Tensor,
    residual_weight: float = 0.05,
    smoothness_weight: float = 0.02,
) -> torch.Tensor:
    loss = residual_loss(u_pd, delta_u_pred, u_expert, residual_weight=residual_weight)
    if smoothness_weight:
        loss = loss + smoothness_weight * smoothness_loss(u_pd + delta_u_pred)
    return loss
