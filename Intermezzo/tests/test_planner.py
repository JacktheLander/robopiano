from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from intermezzo.constants import LEFT_FOREARM_TY_INDEX, RIGHT_FOREARM_TY_INDEX  # noqa: E402
from intermezzo.kinematics import FakeHandKinematics  # noqa: E402
from intermezzo.io import create_unique_run_dir  # noqa: E402
from intermezzo.keys import active_hands_for_transition, extract_waypoint_frames, keyset_hand_sides  # noqa: E402
from intermezzo.planner import (  # noqa: E402
    PlannerConfig,
    _apply_press_windows,
    _apply_selected_finger_z_windows,
    _interpolate_segment_dense,
    build_intermezzo_trajectory,
    plan_between_waypoints,
)
import pytest


def _row(*keys: int) -> np.ndarray:
    out = np.zeros((88,), dtype=np.float32)
    for key in keys:
        out[int(key)] = 1.0
    return out


def test_extract_waypoint_frames_nonempty_keyset_changes() -> None:
    target = np.stack(
        [
            _row(),
            _row(10),
            _row(10),
            _row(10, 50),
            _row(50),
            _row(),
        ],
        axis=0,
    )

    frames = extract_waypoint_frames(target)

    np.testing.assert_array_equal(frames, np.asarray([1, 3, 4], dtype=np.int64))


def test_keyset_hand_sides_and_transition_union() -> None:
    assert keyset_hand_sides(_row(12)) == {"left": True, "right": False}
    assert keyset_hand_sides(_row(60)) == {"left": False, "right": True}
    assert keyset_hand_sides(_row(12, 60)) == {"left": True, "right": True}

    hands = active_hands_for_transition(_row(12), _row(60))

    assert hands == {"left": True, "right": True}


def test_plan_preserves_waypoint_endpoints_and_velocity_shape() -> None:
    frames = np.asarray([1, 5], dtype=np.int64)
    target_keys = np.stack([_row(50), _row(51)], axis=0)
    waypoints = np.zeros((2, 46), dtype=np.float32)
    waypoints[0, 0] = 0.25
    waypoints[0, RIGHT_FOREARM_TY_INDEX] = 0.01
    waypoints[1, 0] = 0.75
    waypoints[1, RIGHT_FOREARM_TY_INDEX] = 0.02

    planned, velocities, segment_ids, sanitized = plan_between_waypoints(
        total_steps=7,
        waypoint_frames=frames,
        waypoint_target_keys=target_keys,
        waypoint_hand_joints=waypoints,
        config=PlannerConfig(control_timestep=0.05),
    )

    assert planned.shape == (7, 46)
    assert velocities.shape == planned.shape
    assert segment_ids.shape == (7,)
    np.testing.assert_array_equal(segment_ids[:2], np.asarray([0, 0], dtype=np.int32))
    np.testing.assert_allclose(planned[1], sanitized[0])
    np.testing.assert_allclose(planned[5], sanitized[1])


def test_right_hand_clearance_lifts_only_active_side_and_clamps() -> None:
    frames = np.asarray([0, 6], dtype=np.int64)
    target_keys = np.stack([_row(50), _row(10)], axis=0)
    waypoints = np.zeros((2, 46), dtype=np.float32)
    waypoints[:, RIGHT_FOREARM_TY_INDEX] = [0.055, 0.06]
    waypoints[:, LEFT_FOREARM_TY_INDEX] = [0.01, 0.02]

    planned, _velocities, _segment_ids, sanitized = plan_between_waypoints(
        total_steps=7,
        waypoint_frames=frames,
        waypoint_target_keys=target_keys,
        waypoint_hand_joints=waypoints,
        config=PlannerConfig(clearance_height=0.02),
    )

    assert float(planned[:, RIGHT_FOREARM_TY_INDEX].max()) <= 0.06
    assert float(planned[1, RIGHT_FOREARM_TY_INDEX]) > float(sanitized[0, RIGHT_FOREARM_TY_INDEX])
    assert float(planned[:, LEFT_FOREARM_TY_INDEX].max()) <= 0.02
    np.testing.assert_allclose(planned[0], sanitized[0])
    np.testing.assert_allclose(planned[6], sanitized[1])


def test_press_windows_use_waypoint_active_hands() -> None:
    frames = np.asarray([0, 6], dtype=np.int64)
    target_keys = np.stack([_row(10), _row(60)], axis=0)
    waypoints = np.zeros((2, 46), dtype=np.float32)
    waypoints[:, RIGHT_FOREARM_TY_INDEX] = [0.01, 0.01]
    waypoints[:, LEFT_FOREARM_TY_INDEX] = [0.01, 0.01]

    planned, _velocities, _segment_ids, sanitized = plan_between_waypoints(
        total_steps=7,
        waypoint_frames=frames,
        waypoint_target_keys=target_keys,
        waypoint_hand_joints=waypoints,
        config=PlannerConfig(clearance_height=0.02),
    )

    assert float(planned[1, LEFT_FOREARM_TY_INDEX]) > float(sanitized[0, LEFT_FOREARM_TY_INDEX])
    np.testing.assert_allclose(planned[6], sanitized[1])


def test_selected_finger_z_windows_do_not_move_unselected_finger_or_forearm() -> None:
    cfg = PlannerConfig(
        interpolation_substeps=8,
        press_approach_s=0.05,
        press_hold_s=0.0,
        press_release_s=0.05,
        ik_max_delta_q=0.05,
        ik_iterations_per_frame=1,
    )
    dense = np.zeros((17, 46), dtype=np.float32)
    waypoints = np.zeros((1, 46), dtype=np.float32)
    # In FakeHandKinematics, right index fingertip 1 uses hand-state coordinates 2 and 3.
    waypoints[0, 2] = 0.0
    waypoints[0, 3] = -0.02
    key_xy = np.zeros((88, 2), dtype=np.float32)
    key_xy[60] = [0.0, -0.02]

    corrected = _apply_selected_finger_z_windows(
        dense,
        waypoint_frames_dense=np.asarray([8], dtype=np.int64),
        waypoint_target_keys=np.stack([_row(60)], axis=0),
        waypoint_hand_joints=waypoints,
        config=cfg,
        key_geometry=key_xy,
        kinematics=FakeHandKinematics(),
    )

    assert float(corrected[7, 3]) < 0.0
    np.testing.assert_allclose(corrected[7, [0, 1, RIGHT_FOREARM_TY_INDEX, LEFT_FOREARM_TY_INDEX]], 0.0)
    np.testing.assert_allclose(corrected[8], waypoints[0])


def test_build_intermezzo_trajectory_with_mock_predictor() -> None:
    target = np.stack([_row(), _row(10), _row(10), _row(60), _row(60)], axis=0)

    def predictor(keys: np.ndarray) -> np.ndarray:
        out = np.zeros((keys.shape[0], 46), dtype=np.float32)
        out[:, 0] = keys.sum(axis=1)
        out[:, RIGHT_FOREARM_TY_INDEX] = 0.01
        out[:, LEFT_FOREARM_TY_INDEX] = 0.01
        return out

    plan = build_intermezzo_trajectory(target, predictor=predictor, config=PlannerConfig())

    np.testing.assert_array_equal(plan.waypoint_frames, np.asarray([1, 3], dtype=np.int64))
    assert plan.waypoint_hand_joints.shape == (2, 46)
    assert plan.planned_hand_joints.shape == (5, 46)
    assert plan.planned_hand_velocities.shape == (5, 46)
    assert plan.planned_hand_joints_dense.shape == (50, 46)
    assert plan.planned_hand_velocities_dense.shape == (50, 46)
    assert plan.metadata["intermezzo_planned_timestep_count"] == 5
    assert plan.metadata["intermezzo_target_state_count"] == 2
    assert plan.metadata["intermezzo_interpolated_timestep_count"] == 3
    assert plan.metadata["intermezzo_upsampling_factor"] == 2.5
    assert plan.metadata["intermezzo_internal_planned_timestep_count"] == 50
    assert plan.metadata["intermezzo_internal_upsampling_factor"] == 25.0
    assert plan.metadata["planned_hand_joints_dense_shape"] == [50, 46]
    assert plan.metadata["dense_control_timestep"] == 0.005
    np.testing.assert_allclose(plan.planned_hand_joints[1], plan.waypoint_hand_joints[0])
    np.testing.assert_allclose(plan.planned_hand_joints[3], plan.waypoint_hand_joints[1])


def test_dense_interpolation_uses_internal_substeps() -> None:
    dense = np.zeros((9, 46), dtype=np.float32)
    segment_ids = np.full((9,), -1, dtype=np.int32)
    q0 = np.zeros((46,), dtype=np.float32)
    q1 = np.ones((46,), dtype=np.float32)

    _interpolate_segment_dense(dense, segment_ids, segment_index=0, start=0, end=8, q0=q0, q1=q1)

    assert np.unique(dense[:, 0]).size == 9
    np.testing.assert_array_equal(segment_ids, np.zeros((9,), dtype=np.int32))


def test_press_window_is_centered_on_waypoint_frame() -> None:
    cfg = PlannerConfig(interpolation_substeps=8, press_approach_s=0.05, press_hold_s=0.0, press_release_s=0.05)
    dense = np.zeros((17, 46), dtype=np.float32)
    dense[:, RIGHT_FOREARM_TY_INDEX] = 0.05
    waypoints = np.zeros((1, 46), dtype=np.float32)
    waypoints[0, RIGHT_FOREARM_TY_INDEX] = 0.01

    _apply_press_windows(
        dense,
        waypoint_frames_dense=np.asarray([8], dtype=np.int64),
        waypoint_target_keys=np.stack([_row(60)], axis=0),
        waypoint_hand_joints=waypoints,
        config=cfg,
    )

    np.testing.assert_allclose(dense[8, RIGHT_FOREARM_TY_INDEX], waypoints[0, RIGHT_FOREARM_TY_INDEX])
    preceding = dense[4:9, RIGHT_FOREARM_TY_INDEX]
    assert np.all(np.diff(preceding) <= 1e-6)


def test_press_windows_preserve_all_waypoint_channels() -> None:
    cfg = PlannerConfig(interpolation_substeps=8, press_approach_s=0.05, press_hold_s=0.0, press_release_s=0.05)
    dense = np.zeros((33, 46), dtype=np.float32)
    waypoints = np.zeros((2, 46), dtype=np.float32)
    waypoints[0, LEFT_FOREARM_TY_INDEX] = 0.01
    waypoints[1, RIGHT_FOREARM_TY_INDEX] = 0.02
    waypoints[1, LEFT_FOREARM_TY_INDEX] = 0.0

    _apply_press_windows(
        dense,
        waypoint_frames_dense=np.asarray([8, 24], dtype=np.int64),
        waypoint_target_keys=np.stack([_row(10), _row(60)], axis=0),
        waypoint_hand_joints=waypoints,
        config=cfg,
    )

    np.testing.assert_allclose(dense[8], waypoints[0])
    np.testing.assert_allclose(dense[24], waypoints[1])


def test_press_windows_keep_same_hand_pressed_between_sustained_waypoints() -> None:
    cfg = PlannerConfig(
        interpolation_substeps=10,
        press_approach_s=0.04,
        press_hold_s=0.01,
        press_release_s=0.04,
        clearance_height=0.02,
    )
    dense = np.zeros((70, 46), dtype=np.float32)
    dense[:, RIGHT_FOREARM_TY_INDEX] = 0.03
    waypoints = np.zeros((2, 46), dtype=np.float32)
    waypoints[:, RIGHT_FOREARM_TY_INDEX] = 0.01

    _apply_press_windows(
        dense,
        waypoint_frames_dense=np.asarray([10, 50], dtype=np.int64),
        waypoint_target_keys=np.stack([_row(60), _row(60, 61)], axis=0),
        waypoint_hand_joints=waypoints,
        config=cfg,
    )

    np.testing.assert_allclose(dense[10:51, RIGHT_FOREARM_TY_INDEX], 0.01)
    assert float(dense[55, RIGHT_FOREARM_TY_INDEX]) > 0.01


def test_press_depth_pushes_active_waypoint_forearm_down() -> None:
    cfg = PlannerConfig(interpolation_substeps=8, press_depth=0.015)
    dense = np.zeros((9, 46), dtype=np.float32)
    waypoints = np.zeros((1, 46), dtype=np.float32)
    waypoints[0, RIGHT_FOREARM_TY_INDEX] = 0.03
    waypoints[0, LEFT_FOREARM_TY_INDEX] = 0.03

    _apply_press_windows(
        dense,
        waypoint_frames_dense=np.asarray([4], dtype=np.int64),
        waypoint_target_keys=np.stack([_row(60)], axis=0),
        waypoint_hand_joints=waypoints,
        config=cfg,
    )

    assert dense[4, RIGHT_FOREARM_TY_INDEX] == pytest.approx(0.015)
    assert dense[4, LEFT_FOREARM_TY_INDEX] == pytest.approx(0.03)


def test_build_intermezzo_trajectory_passes_batch_size_when_supported() -> None:
    target = np.stack([_row(10), _row(60)], axis=0)
    seen: list[int] = []

    def predictor(keys: np.ndarray, *, batch_size: int) -> np.ndarray:
        seen.append(int(batch_size))
        return np.zeros((keys.shape[0], 46), dtype=np.float32)

    build_intermezzo_trajectory(target, predictor=predictor, batch_size=17)

    assert seen == [17]


def test_plan_rejects_waypoints_outside_empty_timeline() -> None:
    with pytest.raises(ValueError, match="waypoint_frames"):
        plan_between_waypoints(
            total_steps=0,
            waypoint_frames=np.asarray([0], dtype=np.int64),
            waypoint_target_keys=np.stack([_row(10)], axis=0),
            waypoint_hand_joints=np.zeros((1, 46), dtype=np.float32),
        )


def test_create_unique_run_dir_does_not_overwrite(tmp_path: Path) -> None:
    first = create_unique_run_dir(tmp_path, run_name="fixed")
    second = create_unique_run_dir(tmp_path, run_name="fixed")

    assert first.name == "fixed"
    assert second.name == "fixed_001"
    assert first.exists()
    assert second.exists()
