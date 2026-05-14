from __future__ import annotations

from functools import lru_cache
import os
from typing import Any

import numpy as np


NUM_PIANO_KEYS = 88
NUM_FINGERTIPS = 10
FINGERTIP_COORDS = 3
FINGERTIP_STATE_DIM = NUM_FINGERTIPS * FINGERTIP_COORDS


def coord_mask_from_tip_mask(active_tip_mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(active_tip_mask, dtype=np.float32)
    if mask.ndim == 1:
        if mask.shape[0] != NUM_FINGERTIPS:
            raise ValueError(f"active_tip_mask must have width {NUM_FINGERTIPS}, got {mask.shape}")
        return np.repeat(mask, FINGERTIP_COORDS).astype(np.float32, copy=False)
    if mask.ndim != 2 or mask.shape[1] != NUM_FINGERTIPS:
        raise ValueError(f"active_tip_mask must have shape (N, {NUM_FINGERTIPS}), got {mask.shape}")
    return np.repeat(mask, FINGERTIP_COORDS, axis=1).astype(np.float32, copy=False)


def infer_active_tip_mask(
    target_keys: np.ndarray,
    fingertip_state: np.ndarray,
    *,
    key_positions: np.ndarray | None = None,
    threshold: float = 0.5,
) -> np.ndarray:
    """Infer active fingertips by nearest observed fingertip to each active key."""
    keys = np.asarray(target_keys, dtype=np.float32)
    tips = np.asarray(fingertip_state, dtype=np.float32)
    if keys.ndim != 2 or keys.shape[1] < NUM_PIANO_KEYS:
        raise ValueError(f"target_keys must have shape (N, 88+), got {keys.shape}")
    if tips.ndim != 2 or tips.shape[1] != FINGERTIP_STATE_DIM:
        raise ValueError(f"fingertip_state must have shape (N, {FINGERTIP_STATE_DIM}), got {tips.shape}")
    if keys.shape[0] != tips.shape[0]:
        raise ValueError(f"target_keys rows {keys.shape[0]} do not match fingertip rows {tips.shape[0]}")
    positions = canonical_piano_key_positions() if key_positions is None else np.asarray(key_positions, dtype=np.float32)
    if positions.shape != (NUM_PIANO_KEYS, FINGERTIP_COORDS):
        raise ValueError(f"key_positions must have shape ({NUM_PIANO_KEYS}, {FINGERTIP_COORDS}), got {positions.shape}")

    active = keys[:, :NUM_PIANO_KEYS] > float(threshold)
    tip_xyz = tips.reshape(tips.shape[0], NUM_FINGERTIPS, FINGERTIP_COORDS)
    out = np.zeros((tips.shape[0], NUM_FINGERTIPS), dtype=np.float32)
    for row_idx in range(tips.shape[0]):
        active_keys = np.flatnonzero(active[row_idx])
        if active_keys.size == 0:
            continue
        row_tips = tip_xyz[row_idx]
        for key_idx in active_keys:
            dist = np.linalg.norm(row_tips - positions[int(key_idx)], axis=1)
            out[row_idx, int(np.argmin(dist))] = 1.0
    return out


@lru_cache(maxsize=1)
def canonical_piano_key_positions() -> np.ndarray:
    """Return RoboPianist piano key site positions in key-index order."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        from dm_control import mjcf
        from robopianist.models.piano import piano
    except Exception as exc:  # pragma: no cover - depends on full RoboPianist env
        raise RuntimeError(
            "RoboPianist and dm_control are required to infer FingerPred active-tip masks. "
            "Pass explicit key_positions in tests or run inside the WAVE sonata environment."
        ) from exc

    piano_model: Any = piano.Piano()
    physics = mjcf.Physics.from_mjcf_model(piano_model.mjcf_model)
    key_sites = getattr(piano_model, "sites", None) or getattr(piano_model, "_sites")
    positions = np.asarray(physics.bind(key_sites).xpos, dtype=np.float32)
    if positions.shape != (NUM_PIANO_KEYS, FINGERTIP_COORDS):
        raise RuntimeError(f"Expected key positions shape ({NUM_PIANO_KEYS}, {FINGERTIP_COORDS}), got {positions.shape}")
    return positions
