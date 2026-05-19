from __future__ import annotations

import numpy as np
import torch

from etude.controllers.hybrid_safe import SafeHybridPDResidualController
from etude.controllers.residual_safety import (
    PhaseGatingConfig,
    ResidualSafetyConfig,
    ResidualSafetyProcessor,
    clip_residual_norm,
    clip_residual_per_dim,
)
from etude.robopianist.state_mapping import StateMapping


class ConstantResidualModel(torch.nn.Module):
    def __init__(self, action_dim: int, value: float) -> None:
        super().__init__()
        self.register_buffer("residual", torch.full((action_dim,), value, dtype=torch.float32))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch = features.shape[0]
        steps = features.shape[1]
        return self.residual.view(1, 1, -1).expand(batch, steps, -1)


def test_norm_clipping_works() -> None:
    residual = np.array([3.0, 4.0], dtype=np.float32)
    clipped = clip_residual_norm(residual, 2.5)
    assert np.isclose(np.linalg.norm(clipped), 2.5)
    assert clipped.shape == residual.shape


def test_per_dim_clipping_works() -> None:
    residual = np.array([0.4, -0.3, 0.05], dtype=np.float32)
    clipped = clip_residual_per_dim(residual, 0.1)
    assert np.allclose(clipped, np.array([0.1, -0.1, 0.05], dtype=np.float32))


def test_phase_gate_changes_residual_scale() -> None:
    processor = ResidualSafetyProcessor(
        ResidualSafetyConfig(
            scale=1.0,
            phase_gating=PhaseGatingConfig(enabled=True, approach=0.25, contact=1.0),
        )
    )
    approach_residual, approach_diag = processor.process(np.array([1.0, 1.0], dtype=np.float32), phase="approach")
    processor.reset()
    contact_residual, contact_diag = processor.process(np.array([1.0, 1.0], dtype=np.float32), phase="contact")
    assert np.allclose(approach_residual, np.array([0.25, 0.25], dtype=np.float32))
    assert np.allclose(contact_residual, np.array([1.0, 1.0], dtype=np.float32))
    assert approach_diag["control/phase_gate"] == 0.25
    assert contact_diag["control/phase_gate"] == 1.0


def test_safe_hybrid_controller_returns_expected_shape_and_diagnostics() -> None:
    mapping = StateMapping(
        qpos_indices_46=list(range(46)),
        qvel_indices_46=list(range(46)),
        action_indices=list(range(4)),
        fingertip_indices=None,
        key_state_indices=None,
        action_low=-np.ones(4, dtype=np.float32),
        action_high=np.ones(4, dtype=np.float32),
    )
    controller = SafeHybridPDResidualController(
        mapping=mapping,
        residual_model=ConstantResidualModel(action_dim=4, value=0.3),
        safety=ResidualSafetyConfig(
            scale=1.0,
            clip_norm=0.4,
            clip_per_dim=0.2,
            phase_gating=PhaseGatingConfig(enabled=True, contact=0.5),
        ),
        device="cpu",
    )
    controller.reset(np.zeros((2, 46), dtype=np.float32), np.zeros((2, 46), dtype=np.float32))
    action = controller.act(
        {
            "q": np.zeros(46, dtype=np.float32),
            "qdot": np.zeros(46, dtype=np.float32),
            "phase": "contact",
        },
        0,
    )
    diagnostics = controller.diagnostics()
    assert action.shape == (4,)
    assert np.all(np.abs(action) <= 1.0)
    for key in (
        "control/raw_residual_norm",
        "control/clipped_residual_norm",
        "control/residual_clip_fraction",
        "control/final_action_clip_fraction",
        "control/phase_gate",
    ):
        assert key in diagnostics
