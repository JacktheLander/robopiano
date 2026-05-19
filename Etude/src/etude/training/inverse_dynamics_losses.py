from __future__ import annotations

import torch
import torch.nn.functional as F


def inverse_dynamics_bc_loss(
    action_pred: torch.Tensor,
    action_expert: torch.Tensor,
    *,
    beta: float = 1.0,
) -> torch.Tensor:
    return F.smooth_l1_loss(action_pred, action_expert, beta=beta)


def inverse_dynamics_residual_bc_loss(
    pd_action: torch.Tensor,
    residual_pred: torch.Tensor,
    action_expert: torch.Tensor,
    *,
    residual_weight: float = 0.05,
    beta: float = 1.0,
) -> torch.Tensor:
    action_loss = F.smooth_l1_loss(pd_action + residual_pred, action_expert, beta=beta)
    residual_penalty = torch.mean(residual_pred.square())
    return action_loss + residual_weight * residual_penalty


def next_state_consistency_loss(
    q_next_pred: torch.Tensor | None,
    q_next_target: torch.Tensor | None,
    *,
    enabled: bool = False,
) -> torch.Tensor:
    if not enabled or q_next_pred is None or q_next_target is None:
        device = None
        if q_next_pred is not None:
            device = q_next_pred.device
        elif q_next_target is not None:
            device = q_next_target.device
        return torch.zeros((), dtype=torch.float32, device=device)
    return torch.mean((q_next_pred - q_next_target).square())


def inverse_dynamics_smoothness_loss(actions: torch.Tensor) -> torch.Tensor:
    if actions.shape[-2] < 2:
        return actions.new_tensor(0.0)
    return torch.mean((actions[..., 1:, :] - actions[..., :-1, :]).square())
