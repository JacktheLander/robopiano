from __future__ import annotations

import numpy as np

from etude.controllers.pd_scheduled import ScheduledPDController
from etude.robopianist.state_mapping import StateMapping


def _mapping(action_dim: int = 6) -> StateMapping:
    return StateMapping(
        qpos_indices_46=list(range(46)),
        qvel_indices_46=list(range(46)),
        action_indices=list(range(action_dim)),
        fingertip_indices=None,
        key_state_indices=None,
        action_low=-np.ones(action_dim, dtype=np.float32),
        action_high=np.ones(action_dim, dtype=np.float32),
    )


def _obs(fill_q: float = 0.0, fill_qdot: float = 0.0) -> dict[str, np.ndarray]:
    return {
        "q": np.full(46, fill_q, dtype=np.float32),
        "qdot": np.full(46, fill_qdot, dtype=np.float32),
    }


def test_scheduled_pd_reset_clears_state() -> None:
    controller = ScheduledPDController(_mapping(), kp=8.0, kd=0.4)
    q_ref = np.ones((3, 46), dtype=np.float32)
    controller.reset(q_ref, np.zeros_like(q_ref))
    controller.act(_obs(), 0)

    controller.reset(np.zeros((2, 46), dtype=np.float32), np.zeros((2, 46), dtype=np.float32))

    assert controller.previous_action is None
    assert controller.stats.steps == 0
    assert controller.phase_schedule is None


def test_scheduled_pd_returns_action_shape() -> None:
    controller = ScheduledPDController(_mapping(action_dim=4), kp=20.0, kd=0.0, lookahead_steps=0)
    q_ref = np.ones((2, 46), dtype=np.float32)
    controller.reset(q_ref, np.zeros_like(q_ref))

    action = controller.act(_obs(), 0)

    assert action.shape == (4,)
    assert np.allclose(action, 1.0)


def test_scheduled_pd_supports_scalar_and_grouped_gains() -> None:
    q_ref = np.zeros((2, 46), dtype=np.float32)
    q_ref[:, 0] = 1.0
    q_ref[:, 20] = 1.0

    scalar_controller = ScheduledPDController(_mapping(), kp=2.0, kd=0.0, lookahead_steps=0)
    scalar_controller.reset(q_ref, np.zeros_like(q_ref))
    scalar_action = scalar_controller.act(_obs(), 0)

    grouped_controller = ScheduledPDController(
        _mapping(),
        mode="grouped",
        kp=0.0,
        kd=0.0,
        kp_groups={"left_arm": 3.0, "left_hand": 5.0},
        lookahead_steps=0,
    )
    grouped_controller.reset(q_ref, np.zeros_like(q_ref))
    grouped_action = grouped_controller.act(_obs(), 0)

    assert np.isclose(scalar_action[0], 1.0)
    assert np.isclose(grouped_action[0], 1.0)
    assert np.isclose(grouped_controller.base_kp[0], 3.0)
    assert np.isclose(grouped_controller.base_kp[20], 5.0)


def test_scheduled_pd_smoothing_does_not_crash() -> None:
    controller = ScheduledPDController(
        _mapping(),
        kp=0.5,
        kd=0.0,
        lookahead_steps=0,
        action_smoothing={"enabled": True, "alpha": 0.5},
    )
    q_ref = np.zeros((3, 46), dtype=np.float32)
    q_ref[0, 0] = 1.0
    q_ref[1, 0] = -1.0
    q_ref[2, 0] = 1.0
    controller.reset(q_ref, np.zeros_like(q_ref))

    first = controller.act(_obs(), 0)
    second = controller.act(_obs(), 1)

    assert first.shape == second.shape == (6,)
    assert not np.isnan(second).any()


def test_scheduled_pd_diagnostics_return_numbers() -> None:
    controller = ScheduledPDController(
        _mapping(),
        mode="phase_scheduled",
        kp=2.0,
        kd=0.1,
        lookahead_steps=1,
        phase_kp_scales={"approach": 1.5},
        phase_kd_scales={"approach": 1.2},
        action_smoothing={"enabled": True, "alpha": 0.25},
    )
    q_ref = np.ones((3, 46), dtype=np.float32)
    metadata = {"phases": ["attack", "attack", "release"]}
    controller.reset(q_ref, np.zeros_like(q_ref), metadata=metadata)
    controller.act(_obs(), 0)

    diagnostics = controller.diagnostics()

    assert diagnostics["control/action_clip_rate"] >= 0.0
    assert diagnostics["control/unclipped_action_l2"] >= 0.0
    assert diagnostics["control/lookahead_steps"] == 1.0
    assert diagnostics["control/smoothing_alpha"] == 0.25
    assert diagnostics["control/phase_count/approach"] == 1.0
