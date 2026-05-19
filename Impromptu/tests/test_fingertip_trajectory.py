from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from impromptu.config import ImpromptuConfig  # noqa: E402
from impromptu.fingertip_trajectory import build_fingertip_trajectory  # noqa: E402


def _inputs(assignments: np.ndarray, targets: np.ndarray, frames: np.ndarray | None = None):
    waypoint_tips = np.zeros((assignments.shape[0], 10, 3), dtype=np.float32)
    return {
        "total_steps": 8,
        "waypoint_frames": np.asarray([1, 5] if frames is None else frames, dtype=np.int64),
        "assignments": assignments.astype(np.int32),
        "fingertip_targets": targets.astype(np.float32),
        "waypoint_fingertips": waypoint_tips,
        "config": ImpromptuConfig(interpolation_substeps=4, approach_s=0.05, hold_s=0.0, release_s=0.05),
    }


def test_unassigned_fingers_are_nan_with_zero_weight() -> None:
    assignments = np.full((1, 10), -1, dtype=np.int32)
    targets = np.full((1, 10, 3), np.nan, dtype=np.float32)
    traj = build_fingertip_trajectory(**_inputs(assignments, targets, frames=np.asarray([1])))

    assert np.all(traj.weights[:, 3] == 0.0)
    assert np.all(np.isnan(traj.targets[:, 3]))


def test_press_frame_equals_press_target_and_precontact_is_higher() -> None:
    assignments = np.full((1, 10), -1, dtype=np.int32)
    assignments[0, 2] = 40
    targets = np.full((1, 10, 3), np.nan, dtype=np.float32)
    targets[0, 2] = np.asarray([0.1, 0.2, -0.01], dtype=np.float32)
    traj = build_fingertip_trajectory(**_inputs(assignments, targets, frames=np.asarray([2])))
    press_frame = 2 * 4

    np.testing.assert_allclose(traj.targets[press_frame, 2], targets[0, 2])
    assert traj.weights[press_frame, 2] == 1.0
    assert traj.targets[press_frame - 1, 2, 2] > traj.targets[press_frame, 2, 2]


def test_sustained_same_finger_key_does_not_release_between_adjacent_waypoints() -> None:
    assignments = np.full((2, 10), -1, dtype=np.int32)
    assignments[:, 1] = 12
    targets = np.full((2, 10, 3), np.nan, dtype=np.float32)
    targets[:, 1] = np.asarray([0.1, 0.0, -0.01], dtype=np.float32)
    traj = build_fingertip_trajectory(**_inputs(assignments, targets))
    first = 1 * 4
    second = 5 * 4

    assert np.all(traj.weights[first: second + 1, 1] == 1.0)
    expected = np.broadcast_to(targets[0, 1], traj.targets[first: second + 1, 1].shape)
    np.testing.assert_allclose(traj.targets[first: second + 1, 1], expected)


def test_changed_assignment_lifts_before_moving_to_next_key() -> None:
    assignments = np.full((2, 10), -1, dtype=np.int32)
    assignments[0, 1] = 12
    assignments[1, 1] = 20
    targets = np.full((2, 10, 3), np.nan, dtype=np.float32)
    targets[0, 1] = np.asarray([0.0, 0.0, -0.01], dtype=np.float32)
    targets[1, 1] = np.asarray([0.3, 0.0, -0.01], dtype=np.float32)
    traj = build_fingertip_trajectory(**_inputs(assignments, targets))
    first = 1 * 4
    second = 5 * 4

    assert np.nanmax(traj.targets[first + 1 : second, 1, 2]) > targets[0, 1, 2]
    assert traj.targets[second - 1, 1, 2] > traj.targets[second, 1, 2]
