from __future__ import annotations

import torch

from etude.training.fingertip_phase_losses import (
    contact_window_fingertip_loss,
    inactive_finger_clearance_loss,
    phase_weighted_joint_tracking_loss,
    release_phase_penalty_hook,
    weighted_fingertip_mse,
)


def test_weighted_fingertip_losses_respect_weights_and_contact_mask() -> None:
    pred = torch.tensor([[[0.0, 0.0], [2.0, 2.0]]])
    target = torch.tensor([[[1.0, 1.0], [0.0, 0.0]]])
    weights = torch.tensor([[1.0, 3.0]])
    contact_mask = torch.tensor([[1.0, 0.0]])

    weighted = weighted_fingertip_mse(pred, target, fingertip_weights=weights)
    contact = contact_window_fingertip_loss(pred, target, contact_mask=contact_mask, fingertip_weights=weights)

    assert torch.isclose(weighted, torch.tensor(3.25))
    assert torch.isclose(contact, torch.tensor(1.0))


def test_inactive_finger_clearance_loss_only_penalizes_below_margin() -> None:
    positions = torch.tensor(
        [
            [[0.0, 0.0, 0.03], [0.0, 0.0, 0.005]],
            [[0.0, 0.0, 0.02], [0.0, 0.0, 0.001]],
        ]
    )
    inactive_mask = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    loss = inactive_finger_clearance_loss(positions, inactive_mask, clearance_margin=0.01)
    assert torch.isclose(loss, torch.tensor(0.007))


def test_phase_weighted_joint_tracking_and_release_hook_are_finite() -> None:
    q_pred = torch.tensor([[[0.0, 0.0], [2.0, 2.0]]])
    q_ref = torch.tensor([[[1.0, 1.0], [0.0, 0.0]]])
    qdot_pred = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
    qdot_ref = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]])
    phase_weights = torch.tensor([[1.0, 0.5]])
    phase_ids = torch.tensor([[2, 3]])
    penalty_values = torch.tensor([[0.2, 0.6]])

    tracking = phase_weighted_joint_tracking_loss(
        q_pred,
        q_ref,
        phase_weights,
        qdot_pred=qdot_pred,
        qdot_ref=qdot_ref,
        velocity_weight=0.5,
    )
    release = release_phase_penalty_hook(penalty_values, phase_ids=phase_ids, release_phase_id=3)

    assert torch.isclose(tracking, torch.tensor(2.1666667))
    assert torch.isclose(release, torch.tensor(0.6))
