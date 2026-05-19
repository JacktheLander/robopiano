from __future__ import annotations

import torch
import torch.nn.functional as F


def target_key_activation_loss(
    logits: torch.Tensor,
    target_keys: torch.Tensor,
    *,
    focal_gamma: float = 0.0,
    positive_weight: float | None = None,
) -> torch.Tensor:
    targets = target_keys.float()
    pos_weight = None
    if positive_weight is not None:
        pos_weight = torch.as_tensor(positive_weight, dtype=logits.dtype, device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pos_weight)
    if focal_gamma <= 0:
        return bce.mean()
    probs = torch.sigmoid(logits)
    pt = torch.where(targets > 0.5, probs, 1.0 - probs)
    return (((1.0 - pt) ** focal_gamma) * bce).mean()


def wrong_key_penalty(
    key_state: torch.Tensor,
    target_keys: torch.Tensor,
    *,
    active_threshold: float = 0.5,
) -> torch.Tensor:
    wrong_mask = _wrong_key_mask(key_state, target_keys, active_threshold=active_threshold)
    return wrong_mask.float().mean()


def missed_key_penalty(
    key_state: torch.Tensor,
    target_keys: torch.Tensor,
    *,
    active_threshold: float = 0.5,
) -> torch.Tensor:
    missed_mask = _missed_key_mask(key_state, target_keys, active_threshold=active_threshold)
    return missed_mask.float().mean()


def key_transition_loss(logits: torch.Tensor, target_keys: torch.Tensor) -> torch.Tensor:
    if logits.shape[-2] < 2 or target_keys.shape[-2] < 2:
        return logits.new_tensor(0.0)
    pred_delta = torch.sigmoid(logits[..., 1:, :]) - torch.sigmoid(logits[..., :-1, :])
    target_delta = target_keys[..., 1:, :].float() - target_keys[..., :-1, :].float()
    return F.smooth_l1_loss(pred_delta, target_delta)


def press_intent_aux_loss(press_intent: torch.Tensor, target_keys: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(press_intent, target_keys.float())


def _wrong_key_mask(
    key_state: torch.Tensor,
    target_keys: torch.Tensor,
    *,
    active_threshold: float,
) -> torch.Tensor:
    pressed = key_state.float() >= active_threshold
    target = target_keys.float() >= active_threshold
    return torch.logical_and(pressed, torch.logical_not(target))


def _missed_key_mask(
    key_state: torch.Tensor,
    target_keys: torch.Tensor,
    *,
    active_threshold: float,
) -> torch.Tensor:
    pressed = key_state.float() >= active_threshold
    target = target_keys.float() >= active_threshold
    return torch.logical_and(target, torch.logical_not(pressed))
