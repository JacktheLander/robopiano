from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bagatelle.evaluation import fingertip_summary_from_trajectory  # noqa: E402


def test_fingertip_summary_ignores_unassigned_nan_targets() -> None:
    targets = np.full((1, 10, 3), np.nan, dtype=np.float32)
    measured = np.zeros((1, 10, 3), dtype=np.float32)
    targets[0, 2] = [0.0, 0.0, 0.0]
    measured[0, 2] = [0.01, 0.0, 0.0]

    summary = fingertip_summary_from_trajectory(
        {"fingertip_targets": targets, "waypoint_fingertips": measured},
        success_threshold=0.02,
    )

    assert summary["fingertip_assignments"] == 1
    assert summary["fingertip_distance_mean"] == np.float32(0.01)
    assert summary["fingertip_success_rate"] == 1.0
