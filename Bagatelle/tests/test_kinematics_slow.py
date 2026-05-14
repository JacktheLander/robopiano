from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bagatelle.assignment import assign_fingers_previous_pose  # noqa: E402
from bagatelle.config import BagatelleConfig  # noqa: E402
from bagatelle.kinematics import BagatelleKinematics  # noqa: E402
from bagatelle.planner import plan_target_keys  # noqa: E402


pytestmark = pytest.mark.skipif(
    os.environ.get("BAGATELLE_RUN_SLOW") != "1",
    reason="Set BAGATELLE_RUN_SLOW=1 to run RoboPianist integration tests.",
)


def _tiny_targets() -> np.ndarray:
    target = np.zeros((2, 88), dtype=np.float32)
    target[0, 40] = 1.0
    target[1, 41] = 1.0
    return target


def test_kinematics_loads_and_solves_tiny_assignment() -> None:
    config = BagatelleConfig(ik_max_nfev=3)
    with BagatelleKinematics(config, target_keys=_tiny_targets()) as kin:
        previous_qpos = kin.neutral_qpos
        previous_tips = kin.fingertip_positions_for_qpos(previous_qpos)
        active = np.asarray([40], dtype=np.int32)
        assignment = assign_fingers_previous_pose(active, previous_tips, kin.key_contact_targets(active), config)
        result = kin.solve_press_pose(assignment, previous_qpos, config=config)

        lower, upper = kin.joint_bounds
        assert result.pose.shape == (46,)
        assert result.fingertip_positions.shape == (10, 3)
        assert np.isfinite(result.pose).all()
        assert np.all(result.pose >= lower - 1e-6)
        assert np.all(result.pose <= upper + 1e-6)


def test_plan_target_keys_tiny_sequence_with_real_simulator() -> None:
    plan = plan_target_keys(_tiny_targets(), config=BagatelleConfig(ik_max_nfev=3))

    assert plan.planned_hand_joints.shape == (2, 46)
    assert np.isfinite(plan.planned_hand_joints).all()
    assert plan.ik_metrics.shape[0] == 2
