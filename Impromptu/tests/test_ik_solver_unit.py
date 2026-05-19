from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
REPO = ROOT.parents[0]
for path in (SRC, REPO / "Bagatelle" / "src"):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from impromptu.config import ImpromptuConfig  # noqa: E402
from impromptu.ik_solver import IK_ANCHOR_METRIC_COLUMNS, ik_metric_row, solve_fingertip_frame  # noqa: E402


class FakeKinematics:
    def __init__(self) -> None:
        self.joint_lower = np.full((46,), -1.0, dtype=np.float32)
        self.joint_upper = np.full((46,), 1.0, dtype=np.float32)

    def clip_qpos(self, qpos: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(qpos, dtype=np.float32), self.joint_lower, self.joint_upper)

    def fingertip_positions_for_qpos(self, qpos: np.ndarray) -> np.ndarray:
        tips = np.zeros((10, 3), dtype=np.float32)
        tips[:, 0] = np.arange(10, dtype=np.float32) * 0.01
        tips[0, 2] = float(np.asarray(qpos, dtype=np.float32)[0])
        return tips


def test_no_active_targets_returns_previous_qpos_with_nfev_zero() -> None:
    kin = FakeKinematics()
    previous = np.full((46,), 0.1, dtype=np.float32)
    result = solve_fingertip_frame(
        kin=kin,
        fingertip_targets=np.full((10, 3), np.nan, dtype=np.float32),
        fingertip_weights=np.zeros((10,), dtype=np.float32),
        previous_qpos=previous,
        neutral_qpos=np.zeros((46,), dtype=np.float32),
        config=ImpromptuConfig(),
    )

    np.testing.assert_allclose(result.pose, previous)
    assert result.nfev == 0


def test_active_target_path_returns_qpos_shape_and_metric_shape() -> None:
    kin = FakeKinematics()
    targets = np.full((10, 3), np.nan, dtype=np.float32)
    weights = np.zeros((10,), dtype=np.float32)
    targets[0] = np.asarray([0.0, 0.0, 0.2], dtype=np.float32)
    weights[0] = 1.0
    result = solve_fingertip_frame(
        kin=kin,
        fingertip_targets=targets,
        fingertip_weights=weights,
        previous_qpos=np.zeros((46,), dtype=np.float32),
        neutral_qpos=np.zeros((46,), dtype=np.float32),
        config=ImpromptuConfig(ik_max_nfev=5),
    )

    assert result.pose.shape == (46,)
    assert ik_metric_row(result).shape == (len(IK_ANCHOR_METRIC_COLUMNS),)
