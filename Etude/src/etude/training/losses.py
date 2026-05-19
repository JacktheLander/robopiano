from __future__ import annotations

import torch
import torch.nn.functional as F


def behavior_cloning_loss(u_pred: torch.Tensor, u_expert: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    return F.smooth_l1_loss(u_pred, u_expert, beta=beta)


def residual_loss(
    u_pd: torch.Tensor,
    delta_u_pred: torch.Tensor,
    u_expert: torch.Tensor,
    residual_weight: float = 0.05,
    beta: float = 1.0,
) -> torch.Tensor:
    u = u_pd + delta_u_pred
    action = F.smooth_l1_loss(u, u_expert, beta=beta)
    residual = torch.mean(delta_u_pred.square())
    return action + residual_weight * residual


def smoothness_loss(actions: torch.Tensor) -> torch.Tensor:
    if actions.shape[-2] < 2:
        return actions.new_tensor(0.0)
    return torch.mean((actions[..., 1:, :] - actions[..., :-1, :]).square())


def joint_tracking_loss(
    q_pred: torch.Tensor,
    q_ref: torch.Tensor,
    qdot_pred: torch.Tensor | None = None,
    qdot_ref: torch.Tensor | None = None,
    velocity_weight: float = 0.1,
) -> torch.Tensor:
    loss = torch.mean((q_pred - q_ref).square())
    if qdot_pred is not None and qdot_ref is not None:
        loss = loss + velocity_weight * torch.mean((qdot_pred - qdot_ref).square())
    return loss


def key_press_loss(logits: torch.Tensor, target_keys: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, target_keys.float())
