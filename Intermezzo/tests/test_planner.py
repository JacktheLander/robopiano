from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from intermezzo.constants import LEFT_FOREARM_TY_INDEX, RIGHT_FOREARM_TY_INDEX  # noqa: E402
from intermezzo.io import create_unique_run_dir  # noqa: E402
from intermezzo.keys import active_hands_for_transition, extract_waypoint_frames, keyset_hand_sides  # noqa: E402
from intermezzo.planner import PlannerConfig, build_intermezzo_trajectory, plan_between_waypoints  # noqa: E402
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
    target_keys = np.stack([_row(50), _row(51)], axis=0)
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
    assert float(planned[2, RIGHT_FOREARM_TY_INDEX]) > float(max(sanitized[0, RIGHT_FOREARM_TY_INDEX], sanitized[1, RIGHT_FOREARM_TY_INDEX]) - 1e-6)
    assert float(planned[:, LEFT_FOREARM_TY_INDEX].max()) <= 0.02
    np.testing.assert_allclose(planned[0], sanitized[0])
    np.testing.assert_allclose(planned[6], sanitized[1])


def test_cross_hand_transition_lifts_both_hands() -> None:
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

    assert float(planned[2, RIGHT_FOREARM_TY_INDEX]) > float(sanitized[0, RIGHT_FOREARM_TY_INDEX])
    assert float(planned[2, LEFT_FOREARM_TY_INDEX]) > float(sanitized[0, LEFT_FOREARM_TY_INDEX])


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
    np.testing.assert_allclose(plan.planned_hand_joints[1], plan.waypoint_hand_joints[0])
    np.testing.assert_allclose(plan.planned_hand_joints[3], plan.waypoint_hand_joints[1])


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
