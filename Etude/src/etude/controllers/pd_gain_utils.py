from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

REFERENCE_DIM = 46

_PHASE_ALIASES: dict[str, str] = {
    "approach": "approach",
    "attack": "approach",
    "precontact": "pre_contact",
    "pre_contact": "pre_contact",
    "contact": "contact",
    "press": "contact",
    "hold": "hold",
    "sustain": "hold",
    "release": "release",
    "recovery": "recovery",
    "recover": "recovery",
    "idle": "unknown",
    "unknown": "unknown",
}


def expand_gain(gain: float | Sequence[float] | np.ndarray, *, name: str = "gain") -> np.ndarray:
    """Expand a scalar gain or validate a 46D gain vector."""
    gain_array = np.asarray(gain, dtype=np.float32)
    if gain_array.ndim == 0:
        return np.full(REFERENCE_DIM, float(gain_array), dtype=np.float32)
    return validate_gain_array(gain_array, name=name)


def validate_gain_array(gain: Sequence[float] | np.ndarray, *, name: str = "gain") -> np.ndarray:
    """Validate a gain vector with Etude's 46D reference layout."""
    gain_array = np.asarray(gain, dtype=np.float32).reshape(-1)
    if gain_array.shape != (REFERENCE_DIM,):
        raise ValueError(f"{name} must have shape [46], got {gain_array.shape}")
    return gain_array.astype(np.float32, copy=False)


def build_grouped_gains(
    group_gains: Mapping[str, float | Sequence[float] | np.ndarray],
    joint_groups: Mapping[str, Sequence[int]],
    *,
    default_gain: float | Sequence[float] | np.ndarray = 0.0,
    name: str = "gain",
) -> np.ndarray:
    """Construct a 46D gain vector from named joint groups."""
    gains = expand_gain(default_gain, name=f"{name}.default")
    for group_name, group_value in group_gains.items():
        if group_name == "default":
            gains = expand_gain(group_value, name=f"{name}.default")
            continue
        if group_name not in joint_groups:
            raise KeyError(f"Unknown joint group '{group_name}' for {name}")
        indices = np.asarray(joint_groups[group_name], dtype=np.int64).reshape(-1)
        if indices.size == 0:
            raise ValueError(f"Joint group '{group_name}' for {name} is empty")
        if np.any(indices < 0) or np.any(indices >= REFERENCE_DIM):
            raise ValueError(f"Joint group '{group_name}' has out-of-range indices")

        group_gain = np.asarray(group_value, dtype=np.float32)
        if group_gain.ndim == 0:
            gains[indices] = float(group_gain)
            continue
        group_gain = group_gain.reshape(-1)
        if group_gain.shape == (REFERENCE_DIM,):
            gains[indices] = group_gain[indices]
            continue
        if group_gain.shape == (indices.size,):
            gains[indices] = group_gain
            continue
        raise ValueError(
            f"{name}.{group_name} must be scalar, shape [46], or shape [{indices.size}], "
            f"got {group_gain.shape}"
        )
    return gains.astype(np.float32, copy=False)


def apply_phase_gain_scale(
    gains: Sequence[float] | np.ndarray,
    phase: Any,
    phase_scales: Mapping[str, float | Sequence[float] | np.ndarray] | None,
    *,
    name: str = "gain",
) -> np.ndarray:
    """Apply an optional phase-dependent scale to a 46D gain vector."""
    gain_array = validate_gain_array(gains, name=name)
    if phase is None or not phase_scales:
        return gain_array
    normalized_scales = {
        canonicalize_phase_key(key): value for key, value in phase_scales.items()
    }
    phase_key = canonicalize_phase_key(phase)
    if phase_key not in normalized_scales:
        return gain_array
    scale = np.asarray(normalized_scales[phase_key], dtype=np.float32)
    if scale.ndim == 0:
        return gain_array * float(scale)
    scale = scale.reshape(-1)
    if scale.shape != (REFERENCE_DIM,):
        raise ValueError(f"{name} phase scale for '{phase_key}' must have shape [46], got {scale.shape}")
    return gain_array * scale.astype(np.float32, copy=False)


def canonicalize_phase_key(phase: Any) -> str:
    normalized = str(phase).strip().lower().replace("-", "_").replace(" ", "_")
    return _PHASE_ALIASES.get(normalized, normalized)


def smooth_action(
    current_action: Sequence[float] | np.ndarray,
    previous_action: Sequence[float] | np.ndarray | None,
    alpha: float,
) -> np.ndarray:
    """Blend actions with an exponential moving average."""
    current = np.asarray(current_action, dtype=np.float32).reshape(-1)
    if previous_action is None:
        return current
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    previous = np.asarray(previous_action, dtype=np.float32).reshape(-1)
    if previous.shape != current.shape:
        raise ValueError(f"previous_action shape {previous.shape} does not match {current.shape}")
    return ((1.0 - float(alpha)) * previous + float(alpha) * current).astype(np.float32)
