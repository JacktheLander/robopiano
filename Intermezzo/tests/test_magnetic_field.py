from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from intermezzo.magnetic_field import combined_magnetic_weight, frame_window_envelope, landing_envelope, radial_weight  # noqa: E402
from intermezzo.planner import PlannerConfig  # noqa: E402


def test_radial_weight_zero_outside_radius() -> None:
    assert radial_weight(0.11, radius=0.10, sigma=0.03) == 0.0
    assert radial_weight(0.10, radius=0.10, sigma=0.03) > 0.0


def test_radial_weight_decays_exponentially_with_distance() -> None:
    near = radial_weight(0.01, radius=0.10, sigma=0.03)
    far = radial_weight(0.05, radius=0.10, sigma=0.03)
    assert 0.0 < far < near < 1.0


def test_combined_weight_is_late_trajectory_biased() -> None:
    cfg = PlannerConfig(magnet_start_fraction=0.5, magnet_power=2.0, magnet_gain=1.0)
    early = combined_magnetic_weight(0.01, 0.25, cfg)
    late = combined_magnetic_weight(0.01, 0.95, cfg)
    assert early == 0.0
    assert late > early
    assert landing_envelope(1.0, 0.5, 2.0) == 1.0


def test_frame_window_envelope_peaks_at_waypoint() -> None:
    assert frame_window_envelope(-4, approach=4, hold=1, release=4, power=2.0) == 0.0
    assert frame_window_envelope(0, approach=4, hold=1, release=4, power=2.0) == 1.0
    assert frame_window_envelope(1, approach=4, hold=1, release=4, power=2.0) == 1.0
    assert frame_window_envelope(5, approach=4, hold=1, release=4, power=2.0) == 0.0
