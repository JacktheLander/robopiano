from __future__ import annotations

import numpy as np
import torch

from etude.controllers.inverse_dynamics import InverseDynamicsController
from etude.features.inverse_dynamics_blocks import InverseDynamicsFeatureSpec, infer_inverse_dynamics_feature_dim
from etude.robopianist.state_mapping import StateMapping


class ConstantActionModel(torch.nn.Module):
    def __init__(self, action_dim: int, value: float) -> None:
        super().__init__()
        self.register_buffer("action", torch.full((action_dim,), value, dtype=torch.float32))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch = features.shape[0]
        return self.action.view(1, -1).expand(batch, -1)


def _mapping() -> StateMapping:
    return StateMapping(
        qpos_indices_46=list(range(46)),
        qvel_indices_46=list(range(46)),
        action_indices=list(range(4)),
        fingertip_indices=None,
        key_state_indices=None,
        action_low=-np.ones(4, dtype=np.float32),
        action_high=np.ones(4, dtype=np.float32),
    )


def _metadata() -> dict[str, np.ndarray]:
    return {
        "target_keys": np.zeros((6, 88), dtype=np.float32),
        "fingertip_ref": np.zeros((6, 30), dtype=np.float32),
    }


def _observation() -> dict[str, np.ndarray]:
    return {
        "q": np.zeros(46, dtype=np.float32),
        "qdot": np.zeros(46, dtype=np.float32),
        "fingertips": np.zeros(30, dtype=np.float32),
        "target_keys": np.zeros(88, dtype=np.float32),
    }


def test_inverse_dynamics_controller_full_action_mode_returns_action_shape() -> None:
    mapping = _mapping()
    spec = InverseDynamicsFeatureSpec()
    controller = InverseDynamicsController(
        mapping,
        ConstantActionModel(action_dim=4, value=0.25),
        output_mode="full_action",
        feature_spec=spec,
    )
    controller.reset(np.ones((6, 46), dtype=np.float32), np.zeros((6, 46), dtype=np.float32), metadata=_metadata())
    action = controller.act(_observation(), 0)

    assert action.shape == (4,)
    assert np.allclose(action, 0.25)
    assert controller.diagnostics()["control/output_mode"] == "full_action"


def test_inverse_dynamics_controller_residual_mode_adds_pd_base_without_crashing() -> None:
    mapping = _mapping()
    spec = InverseDynamicsFeatureSpec()
    controller = InverseDynamicsController(
        mapping,
        ConstantActionModel(action_dim=4, value=0.5),
        output_mode="pd_residual",
        feature_spec=spec,
    )
    controller.reset(np.ones((6, 46), dtype=np.float32), np.zeros((6, 46), dtype=np.float32), metadata=_metadata())
    action = controller.act(_observation(), 0)

    assert action.shape == (4,)
    assert np.allclose(action, 1.0)
    assert controller.last_output is not None
    assert controller.last_output.residual is not None
    assert controller.last_output.residual.shape == (4,)
    assert controller.diagnostics()["control/output_mode"] == "pd_residual"


def test_inverse_dynamics_controller_feature_dim_matches_spec() -> None:
    spec = InverseDynamicsFeatureSpec()
    expected_dim = infer_inverse_dynamics_feature_dim(
        q_dim=46,
        key_dim=88,
        fingertip_dim=30,
        action_dim=4,
        spec=spec,
    )
    controller = InverseDynamicsController(
        _mapping(),
        ConstantActionModel(action_dim=4, value=0.0),
        output_mode="full_action",
        feature_spec=spec,
    )
    controller.reset(np.ones((6, 46), dtype=np.float32), np.zeros((6, 46), dtype=np.float32), metadata=_metadata())
    controller.act(_observation(), 0)

    assert controller.diagnostics()["control/feature_dim"] == expected_dim
