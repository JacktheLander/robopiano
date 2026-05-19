from __future__ import annotations

from typing import Any

import torch


def weighted_fingertip_mse(
    fingertip_pred: torch.Tensor,
    fingertip_target: torch.Tensor,
    fingertip_weights: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    pred = _as_float(fingertip_pred, "fingertip_pred")
    target = _as_float(fingertip_target, "fingertip_target")
    _require_same_shape(pred, target, "fingertip_pred", "fingertip_target")
    error = (pred - target).square()
    reduce_dims = tuple(range(2, error.ndim)) if error.ndim > 2 else ()
    if reduce_dims:
        error = error.mean(dim=reduce_dims)
    weights = _broadcast_like(fingertip_weights, error, fill_value=1.0)
    active_mask = _broadcast_like(mask, error, fill_value=1.0)
    weighted = error * weights * active_mask
    denom = torch.clamp((weights * active_mask).sum(), min=eps)
    return weighted.sum() / denom


def contact_window_fingertip_loss(
    fingertip_pred: torch.Tensor,
    fingertip_target: torch.Tensor,
    contact_mask: torch.Tensor,
    fingertip_weights: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    return weighted_fingertip_mse(
        fingertip_pred=fingertip_pred,
        fingertip_target=fingertip_target,
        fingertip_weights=fingertip_weights,
        mask=contact_mask,
        eps=eps,
    )


def inactive_finger_clearance_loss(
    fingertip_positions: torch.Tensor,
    inactive_mask: torch.Tensor,
    clearance_margin: float = 0.01,
    vertical_axis: int = -1,
    eps: float = 1e-8,
) -> torch.Tensor:
    positions = _as_float(fingertip_positions, "fingertip_positions")
    if positions.ndim < 2:
        raise ValueError("fingertip_positions must have at least 2 dimensions")
    clearance = positions.select(dim=vertical_axis, index=positions.shape[vertical_axis] - 1)
    mask = _broadcast_like(inactive_mask, clearance, fill_value=0.0)
    penalty = torch.relu(clearance_margin - clearance) * mask
    denom = torch.clamp(mask.sum(), min=eps)
    return penalty.sum() / denom


def phase_weighted_joint_tracking_loss(
    q_pred: torch.Tensor,
    q_ref: torch.Tensor,
    phase_weights: torch.Tensor,
    qdot_pred: torch.Tensor | None = None,
    qdot_ref: torch.Tensor | None = None,
    velocity_weight: float = 0.1,
    eps: float = 1e-8,
) -> torch.Tensor:
    pred = _as_float(q_pred, "q_pred")
    ref = _as_float(q_ref, "q_ref")
    _require_same_shape(pred, ref, "q_pred", "q_ref")
    joint_error = (pred - ref).square().mean(dim=-1)
    weights = _broadcast_like(phase_weights, joint_error, fill_value=1.0)
    weighted = joint_error * weights
    denom = torch.clamp(weights.sum(), min=eps)
    loss = weighted.sum() / denom
    if qdot_pred is not None or qdot_ref is not None:
        if qdot_pred is None or qdot_ref is None:
            raise ValueError("qdot_pred and qdot_ref must be provided together")
        qdot_pred = _as_float(qdot_pred, "qdot_pred")
        qdot_ref = _as_float(qdot_ref, "qdot_ref")
        _require_same_shape(qdot_pred, qdot_ref, "qdot_pred", "qdot_ref")
        vel_error = (qdot_pred - qdot_ref).square().mean(dim=-1)
        loss = loss + velocity_weight * (vel_error * weights).sum() / denom
    return loss


def release_phase_penalty_hook(
    penalty_values: torch.Tensor,
    phase_ids: torch.Tensor | None = None,
    phase_one_hot: torch.Tensor | None = None,
    release_phase_id: int = 3,
    reduction: str = "mean",
    eps: float = 1e-8,
) -> torch.Tensor:
    values = _as_float(penalty_values, "penalty_values")
    release_mask = _release_mask(phase_ids=phase_ids, phase_one_hot=phase_one_hot, release_phase_id=release_phase_id)
    mask = _broadcast_like(release_mask, values, fill_value=0.0)
    selected = values * mask
    if reduction == "sum":
        return selected.sum()
    if reduction != "mean":
        raise ValueError(f"Unsupported reduction: {reduction}")
    denom = torch.clamp(mask.sum(), min=eps)
    return selected.sum() / denom


def _release_mask(
    *,
    phase_ids: torch.Tensor | None,
    phase_one_hot: torch.Tensor | None,
    release_phase_id: int,
) -> torch.Tensor:
    if phase_ids is not None:
        return (_as_float(phase_ids, "phase_ids") == float(release_phase_id)).float()
    if phase_one_hot is not None:
        one_hot = _as_float(phase_one_hot, "phase_one_hot")
        if one_hot.ndim == 0:
            raise ValueError("phase_one_hot must have at least one dimension")
        return one_hot[..., release_phase_id].float()
    raise ValueError("Either phase_ids or phase_one_hot is required")


def _as_float(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    return value.float()


def _require_same_shape(first: torch.Tensor, second: torch.Tensor, first_name: str, second_name: str) -> None:
    if first.shape != second.shape:
        raise ValueError(f"{first_name} shape {tuple(first.shape)} must match {second_name} shape {tuple(second.shape)}")


def _broadcast_like(value: torch.Tensor | None, reference: torch.Tensor, fill_value: float) -> torch.Tensor:
    if value is None:
        return torch.full_like(reference, float(fill_value))
    tensor = _as_float(value, "broadcast_value")
    if tensor.shape == reference.shape:
        return tensor
    while tensor.ndim < reference.ndim:
        tensor = tensor.unsqueeze(-1)
    return torch.broadcast_to(tensor, reference.shape).float()
