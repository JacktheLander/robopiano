from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bagatelle.config import BagatelleConfig  # noqa: E402
from bagatelle.kinematics import IKResult  # noqa: E402
from bagatelle.planner import IK_METRIC_COLUMNS, plan_target_keys  # noqa: E402


def _row(*keys: int) -> np.ndarray:
    out = np.zeros((88,), dtype=np.float32)
    for key in keys:
        out[int(key)] = 1.0
    return out


class FakeKinematics:
    def __init__(self) -> None:
        self.neutral_qpos = np.zeros((46,), dtype=np.float32)
        self.environment_name = "fake-env"
        self.midi_proto_path = "fake.proto"
        self.load_info = {}

    def close(self) -> None:
        raise AssertionError("plan_target_keys must not close caller-owned kinematics")

    def fingertip_positions_for_qpos(self, qpos: np.ndarray) -> np.ndarray:
        tips = np.zeros((10, 3), dtype=np.float32)
        tips[:, 0] = np.arange(10, dtype=np.float32)
        tips[:, 1] = float(np.asarray(qpos, dtype=np.float32)[0])
        return tips

    def key_contact_targets(self, keys: np.ndarray) -> np.ndarray:
        return np.asarray([[float(key % 10), 0.0, 0.0] for key in keys], dtype=np.float32)

    def key_press_targets(self, keys: np.ndarray) -> np.ndarray:
        out = self.key_contact_targets(keys)
        out[:, 2] -= 0.008
        return out

    def solve_press_pose(self, assignments, previous_qpos, neutral_qpos=None, config=None) -> IKResult:
        pose = np.asarray(previous_qpos, dtype=np.float32).copy()
        pose[0] += 0.25 + 0.01 * float(assignments.assigned_keys.sum() if assignments.count else 0)
        fingertips = self.fingertip_positions_for_qpos(pose)
        if assignments.count:
            fingertips[assignments.assigned_finger_indices] = assignments.target_positions
        distances = (
            np.linalg.norm(fingertips[assignments.assigned_finger_indices] - assignments.target_positions, axis=1)
            if assignments.count
            else np.zeros((0,), dtype=np.float32)
        )
        return IKResult(
            pose=pose,
            fingertip_positions=fingertips,
            assigned_distances=distances.astype(np.float32),
            residual_norm=float(np.linalg.norm(distances)),
            max_residual=float(np.max(distances)) if distances.size else 0.0,
            success=True,
            optimizer_success=True,
            optimizer_status=1,
            optimizer_message="fake",
            optimizer_cost=0.0,
            nfev=3,
            active_keys=assignments.active_keys,
            assigned_keys=assignments.assigned_keys,
            assigned_finger_indices=assignments.assigned_finger_indices,
            unassigned_keys=assignments.unassigned_keys,
        )


def test_plan_target_keys_with_fake_kinematics_outputs_expected_shapes() -> None:
    target = np.stack([_row(), _row(10), _row(10), _row(11, 55), _row()], axis=0)

    plan = plan_target_keys(target, config=BagatelleConfig(), kinematics=FakeKinematics())

    np.testing.assert_array_equal(plan.waypoint_frames, np.asarray([1, 3], dtype=np.int64))
    assert plan.waypoint_target_keys.shape == (2, 88)
    assert plan.waypoint_hand_joints.shape == (2, 46)
    assert plan.planned_hand_joints.shape == (5, 46)
    assert plan.planned_hand_velocities.shape == (5, 46)
    assert plan.assignments.shape == (2, 10)
    assert plan.assignment_costs.shape == (2, 10)
    assert plan.fingertip_targets.shape == (2, 10, 3)
    assert plan.waypoint_fingertips.shape == (2, 10, 3)
    assert plan.ik_metrics.shape == (2, len(IK_METRIC_COLUMNS))
    assert np.isfinite(plan.planned_hand_joints).all()
    np.testing.assert_allclose(plan.planned_hand_joints[1], plan.waypoint_hand_joints[0])
    np.testing.assert_allclose(plan.planned_hand_joints[3], plan.waypoint_hand_joints[1])


def test_plan_target_keys_no_waypoints_returns_neutral_trajectory() -> None:
    target = np.zeros((4, 88), dtype=np.float32)

    plan = plan_target_keys(target, config=BagatelleConfig(), kinematics=FakeKinematics())

    assert plan.waypoint_frames.shape == (0,)
    assert plan.planned_hand_joints.shape == (4, 46)
    assert np.all(plan.segment_ids == -1)
    np.testing.assert_allclose(plan.planned_hand_joints, np.zeros((4, 46), dtype=np.float32))


def test_npz_payload_contains_required_arrays() -> None:
    plan = plan_target_keys(np.stack([_row(1), _row(2)], axis=0), kinematics=FakeKinematics())

    payload = plan.npz_payload()

    for name in (
        "target_keys",
        "waypoint_frames",
        "planned_hand_joints",
        "planned_hand_velocities",
        "assignments",
        "fingertip_targets",
        "ik_metrics",
    ):
        assert name in payload
