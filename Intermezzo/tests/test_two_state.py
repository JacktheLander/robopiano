from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from intermezzo.kinematics import FakeHandKinematics  # noqa: E402
from intermezzo.planner import PlannerConfig, build_two_state_trajectory  # noqa: E402


def _key_geometry() -> np.ndarray:
    out = np.zeros((88, 2), dtype=np.float32)
    out[:, 1] = np.arange(88, dtype=np.float32) / 100.0
    return out


def _keyset(*keys: int) -> np.ndarray:
    out = np.zeros((88,), dtype=np.float32)
    for key in keys:
        out[int(key)] = 1.0
    return out


def test_two_state_planning_produces_num_steps_by_46() -> None:
    endpoints = np.zeros((2, 46), dtype=np.float32)

    plan = build_two_state_trajectory(_keyset(60), _keyset(61), endpoint_hand_joints=endpoints, num_steps=9)

    assert plan.planned_hand_joints.shape == (9, 46)
    assert plan.planned_hand_velocities.shape == (9, 46)
    assert plan.planned_hand_joints_dense.shape == (90, 46)
    assert plan.planned_hand_velocities_dense.shape == (90, 46)
    assert plan.metadata["intermezzo_planned_timestep_count"] == 9
    assert plan.metadata["intermezzo_target_state_count"] == 2
    assert plan.metadata["intermezzo_interpolated_timestep_count"] == 7
    assert plan.metadata["intermezzo_interpolated_timestep_fraction"] == 7 / 9
    assert plan.metadata["intermezzo_upsampling_factor"] == 4.5
    assert plan.metadata["intermezzo_internal_planned_timestep_count"] == 90
    assert plan.metadata["intermezzo_internal_upsampling_factor"] == 45.0
    assert plan.metadata["planned_hand_joints_dense_shape"] == [90, 46]
    assert plan.metadata["dense_control_timestep"] == 0.005
    np.testing.assert_array_equal(plan.waypoint_frames, np.asarray([0, 8], dtype=np.int64))


def test_magnetic_mode_reduces_active_error_without_moving_inactive_fingertips() -> None:
    key_xy = _key_geometry()
    endpoints = np.zeros((2, 46), dtype=np.float32)
    endpoints[1, 0:2] = [0.0, 0.20]  # right fingertip 0, short of key 60 at y=0.60.
    base = build_two_state_trajectory(
        _keyset(50),
        _keyset(60),
        endpoint_hand_joints=endpoints,
        num_steps=12,
        key_geometry=key_xy,
        kinematics=FakeHandKinematics(),
    )
    magnetic = build_two_state_trajectory(
        _keyset(50),
        _keyset(60),
        endpoint_hand_joints=endpoints,
        num_steps=12,
        config=PlannerConfig(
            enable_key_magnetism=True,
            preserve_waypoint_endpoints=False,
            magnet_radius=1.0,
            magnet_sigma=1.0,
            magnet_gain=1.0,
            magnet_max_xy_step=0.20,
            magnet_start_fraction=0.0,
            ik_max_delta_q=0.20,
        ),
        key_geometry=key_xy,
        kinematics=FakeHandKinematics(),
    )

    target = key_xy[60]
    base_error = np.linalg.norm(base.planned_hand_joints[-1, 0:2] - target)
    magnetic_error = np.linalg.norm(magnetic.planned_hand_joints[-1, 0:2] - target)
    assert magnetic_error < base_error
    # Fake kinematics maps inactive fingertip 1 to q[2:4], which should not be attracted.
    np.testing.assert_allclose(magnetic.planned_hand_joints[:, 2:4], base.planned_hand_joints[:, 2:4])
