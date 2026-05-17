from __future__ import annotations

import warnings
from pathlib import Path
import sys

import numpy as np

from intermezzo.constants import NUM_PIANO_KEYS


def approximate_key_geometry() -> np.ndarray:
    """Return deterministic approximate 88-key x/y centers for unit tests."""
    y = np.linspace(-0.61, 0.61, NUM_PIANO_KEYS, dtype=np.float32)
    x = np.zeros((NUM_PIANO_KEYS,), dtype=np.float32)
    return np.stack([x, y], axis=1).astype(np.float32)


def load_key_geometry(*, allow_approximate: bool = False) -> np.ndarray:
    """Load 88 key x/y positions from RoboPianist/MuJoCo.

    Approximate geometry is only returned when explicitly requested, and a visible warning
    is emitted so rendering/evaluation code does not silently use test geometry.
    """
    try:
        return load_robopianist_key_geometry()
    except Exception as exc:
        if not allow_approximate:
            raise RuntimeError(
                "Could not load RoboPianist key geometry. Install/import RoboPianist and MuJoCo, "
                "or pass explicit test geometry. Approximate geometry is intentionally not used silently."
            ) from exc
        warnings.warn(
            "Using approximate Intermezzo key geometry because RoboPianist/MuJoCo key positions "
            f"could not be queried: {exc}. Do not use this fallback for real rendering.",
            RuntimeWarning,
            stacklevel=2,
        )
        return approximate_key_geometry()


def load_robopianist_key_geometry() -> np.ndarray:
    """Query key center x/y positions from the RoboPianist piano MJCF model."""
    _ensure_repo_on_path()
    from dm_control import mjcf
    from robopianist.models.piano import piano

    model = piano.Piano(change_color_on_activation=False, add_actuators=False)
    physics = mjcf.Physics.from_mjcf_model(model.mjcf_model)
    if hasattr(physics, "forward"):
        physics.forward()

    positions = None
    sites = getattr(model, "sites", None)
    if sites is not None:
        try:
            positions = np.asarray(physics.bind(sites).xpos, dtype=np.float32)
        except Exception:
            positions = None
    if positions is None or positions.shape[0] < NUM_PIANO_KEYS:
        positions = np.asarray(physics.bind(model.keys).xpos, dtype=np.float32)
    if positions.shape[0] < NUM_PIANO_KEYS or positions.shape[1] < 2:
        raise RuntimeError(f"RoboPianist returned invalid key positions with shape {positions.shape}")
    return np.ascontiguousarray(positions[:NUM_PIANO_KEYS, :2], dtype=np.float32)


def _ensure_repo_on_path() -> None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "robopianist"
        if candidate.exists():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return
