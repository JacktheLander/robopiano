from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class KeyFeatureSpec:
    key_dim: int = 88
    lookahead_steps: tuple[int, ...] = (1, 5, 10)
    density_horizon: int = 8
    include_target_now: bool = True
    include_target_lookahead: bool = True
    include_current_state: bool = True
    include_key_error: bool = True
    include_note_density: bool = True
    include_time_to_next: bool = True
    include_transitions: bool = False
    zero_fill_missing: bool = True
    active_threshold: float = 0.5


def build_key_features(
    *,
    t: int,
    target_keys: np.ndarray | None = None,
    key_state: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
    spec: KeyFeatureSpec | None = None,
    **_: Any,
) -> np.ndarray:
    """Build key-aware features for controller inputs using only NumPy."""
    spec = _coerce_spec(spec)
    parts: list[np.ndarray] = []
    target = _prepare_key_sequence(target_keys, spec, fill_missing=True)
    state = _prepare_key_sequence(
        key_state,
        spec,
        fill_missing=spec.zero_fill_missing,
        allow_single_frame=True,
    )

    target_now = _frame_at(target, t, spec)
    state_now = _frame_at(state, t, spec)
    missed_mask, wrong_mask = compute_key_error_masks(target_now, state_now, spec.active_threshold)

    if spec.include_target_now:
        parts.append(target_now)
    if spec.include_target_lookahead:
        parts.extend(_build_lookahead_features(target, t, spec))
    if spec.include_current_state:
        parts.append(state_now)
    if spec.include_key_error:
        parts.extend([missed_mask, wrong_mask])
    if spec.include_note_density:
        parts.append(np.asarray([compute_note_density(target, t, spec)], dtype=np.float32))
    if spec.include_time_to_next:
        parts.append(np.asarray([_resolve_time_to_next_active_key(metadata, t)], dtype=np.float32))
    if spec.include_transitions:
        parts.append(compute_key_transition_features(target, t, spec))
    return np.concatenate(parts).astype(np.float32) if parts else np.zeros(0, dtype=np.float32)


def compute_key_error_masks(
    target_now: np.ndarray,
    key_state_now: np.ndarray,
    active_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    target_mask = np.asarray(target_now, dtype=np.float32).reshape(-1) >= active_threshold
    state_mask = np.asarray(key_state_now, dtype=np.float32).reshape(-1) >= active_threshold
    missed = np.logical_and(target_mask, np.logical_not(state_mask)).astype(np.float32)
    wrong = np.logical_and(state_mask, np.logical_not(target_mask)).astype(np.float32)
    return missed, wrong


def compute_note_density(target_keys: np.ndarray, t: int, spec: KeyFeatureSpec | None = None) -> float:
    spec = _coerce_spec(spec)
    target = _prepare_key_sequence(target_keys, spec, fill_missing=True)
    if target.shape[0] == 0:
        return 0.0
    start = _clip_index(t, target.shape[0])
    stop = min(target.shape[0], start + max(spec.density_horizon, 1))
    window = target[start:stop]
    if window.size == 0:
        return 0.0
    return float(np.mean(window >= spec.active_threshold))


def compute_key_transition_features(
    target_keys: np.ndarray | None,
    t: int,
    spec: KeyFeatureSpec | None = None,
) -> np.ndarray:
    spec = _coerce_spec(spec)
    target = _prepare_key_sequence(target_keys, spec, fill_missing=True)
    current = _frame_at(target, t, spec)
    previous = _frame_at(target, t - 1, spec)
    return (current - previous).astype(np.float32)


def _build_lookahead_features(
    target: np.ndarray,
    t: int,
    spec: KeyFeatureSpec,
) -> list[np.ndarray]:
    return [_frame_at(target, t + step, spec) for step in spec.lookahead_steps]


def _prepare_key_sequence(
    values: np.ndarray | None,
    spec: KeyFeatureSpec,
    *,
    fill_missing: bool,
    allow_single_frame: bool = False,
) -> np.ndarray:
    if values is None:
        if not fill_missing:
            raise ValueError("Missing key data and zero_fill_missing=False")
        return np.zeros((1, spec.key_dim), dtype=np.float32)
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        if not allow_single_frame and array.shape[0] != spec.key_dim:
            raise ValueError(f"Expected key vector of length {spec.key_dim}, got {array.shape}")
        if array.shape[0] != spec.key_dim:
            raise ValueError(f"Expected key vector of length {spec.key_dim}, got {array.shape}")
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] != spec.key_dim:
        raise ValueError(f"Expected key sequence shape [T, {spec.key_dim}], got {array.shape}")
    return array.astype(np.float32)


def _frame_at(values: np.ndarray, t: int, spec: KeyFeatureSpec) -> np.ndarray:
    if values.shape[0] == 0:
        return np.zeros(spec.key_dim, dtype=np.float32)
    return values[_clip_index(t, values.shape[0])].astype(np.float32)


def _resolve_time_to_next_active_key(metadata: dict[str, Any] | None, t: int) -> float:
    if not metadata:
        return 0.0
    if "time_to_next_active_key" in metadata:
        return _scalar_or_indexed(metadata["time_to_next_active_key"], t)
    if "steps_to_next_active_key" in metadata:
        dt = float(metadata.get("dt", 1.0))
        return dt * _scalar_or_indexed(metadata["steps_to_next_active_key"], t)
    return 0.0


def _scalar_or_indexed(value: Any, t: int) -> float:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        return float(array)
    flat = array.reshape(-1)
    if flat.size == 0:
        return 0.0
    return float(flat[_clip_index(t, flat.size)])


def _clip_index(t: int, horizon: int) -> int:
    return int(np.clip(t, 0, max(horizon - 1, 0)))


def _coerce_spec(spec: KeyFeatureSpec | dict[str, Any] | None) -> KeyFeatureSpec:
    if spec is None:
        return KeyFeatureSpec()
    if isinstance(spec, KeyFeatureSpec):
        return spec
    if isinstance(spec, dict):
        return KeyFeatureSpec(**spec)
    raise TypeError(f"Unsupported key feature spec type: {type(spec)!r}")
