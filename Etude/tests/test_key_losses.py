from __future__ import annotations

import torch

from etude.training.key_losses import (
    key_transition_loss,
    missed_key_penalty,
    press_intent_aux_loss,
    target_key_activation_loss,
    wrong_key_penalty,
)


def test_key_losses_are_finite() -> None:
    logits = torch.tensor([[0.0, 2.0, -1.0], [1.5, -0.5, 0.25]])
    targets = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    press = torch.tensor([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])

    assert torch.isfinite(target_key_activation_loss(logits, targets))
    assert torch.isfinite(target_key_activation_loss(logits, targets, focal_gamma=2.0))
    assert torch.isfinite(wrong_key_penalty(press, targets))
    assert torch.isfinite(missed_key_penalty(press, targets))
    assert torch.isfinite(press_intent_aux_loss(logits, targets))


def test_key_transition_loss_is_finite() -> None:
    logits = torch.tensor([[[0.0, 1.0], [1.0, -1.0], [0.5, 0.5]]])
    targets = torch.tensor([[[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]])
    assert torch.isfinite(key_transition_loss(logits, targets))
