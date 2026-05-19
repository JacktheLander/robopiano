from __future__ import annotations

import numpy as np
import torch

from etude.features.inverse_dynamics_blocks import (
    InverseDynamicsFeatureSpec,
    build_inverse_dynamics_features,
    infer_inverse_dynamics_feature_dim,
)
from etude.training.inverse_dynamics_losses import (
    inverse_dynamics_bc_loss,
    inverse_dynamics_residual_bc_loss,
    inverse_dynamics_smoothness_loss,
    next_state_consistency_loss,
)


def test_inverse_dynamics_feature_dimensions_are_stable() -> None:
    spec = InverseDynamicsFeatureSpec(
        desired_state_horizon=1,
        future_steps=(1, 2, 5),
        condition_on_keys=True,
        condition_on_fingertips=True,
        condition_on_previous_action=True,
    )
    q = np.arange(46, dtype=np.float32)
    qdot = np.arange(46, dtype=np.float32) * 0.1
    q_ref = np.tile(q, (8, 1)).astype(np.float32)
    fingertips = np.ones((8, 30), dtype=np.float32)
    target_keys = np.zeros((8, 88), dtype=np.float32)
    previous_action = np.zeros(12, dtype=np.float32)

    features = build_inverse_dynamics_features(
        q=q,
        qdot=qdot,
        q_ref=q_ref,
        t=0,
        fingertips=fingertips[0],
        fingertip_ref=fingertips,
        target_keys=target_keys,
        previous_action=previous_action,
        spec=spec,
    )

    expected_dim = infer_inverse_dynamics_feature_dim(
        q_dim=46,
        key_dim=88,
        fingertip_dim=30,
        action_dim=12,
        spec=spec,
    )
    assert features.dtype == np.float32
    assert features.shape == (expected_dim,)


def test_inverse_dynamics_losses_are_finite() -> None:
    pred = torch.zeros(2, 4)
    expert = torch.ones(2, 4)
    pd_action = torch.full((2, 4), 0.25)

    assert torch.isfinite(inverse_dynamics_bc_loss(pred, expert))
    assert torch.isfinite(inverse_dynamics_residual_bc_loss(pd_action, pred, expert))
    assert torch.isfinite(next_state_consistency_loss(None, None, enabled=False))
    assert inverse_dynamics_smoothness_loss(pred.unsqueeze(0)).item() == 0.0
