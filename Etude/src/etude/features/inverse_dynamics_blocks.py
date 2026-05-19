from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class InverseDynamicsFeatureSpec:
    desired_state_horizon: int = 1
    future_steps: tuple[int, ...] = (1, 2, 5, 10)
    include_future_deltas: bool = True
    condition_on_keys: bool = True
    condition_on_fingertips: bool = True
    condition_on_previous_action: bool = True
    zero_fill_missing: bool = True


def build_inverse_dynamics_features(
    *,
    q: np.ndarray,
    qdot: np.ndarray,
    q_ref: np.ndarray,
    t: int,
    desired_state_horizon: int | None = None,
    future_steps: tuple[int, ...] | list[int] | None = None,
    fingertips: np.ndarray | None = None,
    fingertip_ref: np.ndarray | None = None,
    target_keys: np.ndarray | None = None,
    previous_action: np.ndarray | None = None,
    spec: InverseDynamicsFeatureSpec | dict[str, Any] | None = None,
    **_: Any,
) -> np.ndarray:
    spec = _coerce_spec(spec, desired_state_horizon=desired_state_horizon, future_steps=future_steps)
    q_now = _vector(q, name="q")
    qdot_now = _vector(qdot, name="qdot")
    q_ref_seq = _sequence(q_ref, width=q_now.size, name="q_ref")

    parts = [q_now, qdot_now]

    desired_next = _frame_at(q_ref_seq, t + spec.desired_state_horizon)
    parts.append(desired_next)
    if spec.include_future_deltas:
        parts.append(desired_next - q_now)

    for step in spec.future_steps:
        frame = _frame_at(q_ref_seq, t + step)
        parts.append(frame)
        if spec.include_future_deltas:
            parts.append(frame - q_now)

    if spec.condition_on_fingertips:
        current_fingertips = _optional_feature(
            fingertips,
            reference=fingertip_ref,
            t=t,
            name="fingertips",
            zero_fill_missing=spec.zero_fill_missing,
        )
        desired_fingertips = _optional_feature(
            fingertip_ref,
            reference=fingertips,
            t=t + spec.desired_state_horizon,
            name="fingertip_ref",
            zero_fill_missing=spec.zero_fill_missing,
        )
        parts.extend((current_fingertips, desired_fingertips))
        if current_fingertips.size and desired_fingertips.size:
            parts.append(desired_fingertips - current_fingertips)

    if spec.condition_on_keys:
        key_features = _optional_feature(
            target_keys,
            reference=None,
            t=t + spec.desired_state_horizon,
            name="target_keys",
            zero_fill_missing=spec.zero_fill_missing,
        )
        parts.append(key_features)

    if spec.condition_on_previous_action:
        action_features = _optional_vector_feature(
            previous_action,
            reference=None,
            name="previous_action",
            zero_fill_missing=spec.zero_fill_missing,
        )
        parts.append(action_features)

    return np.concatenate(parts).astype(np.float32)


def infer_inverse_dynamics_feature_dim(
    *,
    q_dim: int,
    key_dim: int = 0,
    fingertip_dim: int = 0,
    action_dim: int = 0,
    spec: InverseDynamicsFeatureSpec | dict[str, Any] | None = None,
) -> int:
    spec = _coerce_spec(spec)
    total = q_dim * 2
    total += q_dim * 2
    total += len(spec.future_steps) * q_dim * (2 if spec.include_future_deltas else 1)
    if spec.condition_on_fingertips:
        total += fingertip_dim * 3
    if spec.condition_on_keys:
        total += key_dim
    if spec.condition_on_previous_action:
        total += action_dim
    return total


def _coerce_spec(
    spec: InverseDynamicsFeatureSpec | dict[str, Any] | None,
    *,
    desired_state_horizon: int | None = None,
    future_steps: tuple[int, ...] | list[int] | None = None,
) -> InverseDynamicsFeatureSpec:
    overrides: dict[str, Any] = {}
    if desired_state_horizon is not None:
        overrides["desired_state_horizon"] = int(desired_state_horizon)
    if future_steps is not None:
        overrides["future_steps"] = tuple(int(step) for step in future_steps)
    if spec is None:
        return InverseDynamicsFeatureSpec(**overrides)
    if isinstance(spec, InverseDynamicsFeatureSpec):
        if not overrides:
            return spec
        data = spec.__dict__.copy()
        data.update(overrides)
        return InverseDynamicsFeatureSpec(**data)
    if isinstance(spec, dict):
        data = dict(spec)
        data.update(overrides)
        if "future_steps" in data:
            data["future_steps"] = tuple(int(step) for step in data["future_steps"])
        return InverseDynamicsFeatureSpec(**data)
    raise TypeError(f"Unsupported inverse dynamics feature spec type: {type(spec)!r}")


def _vector(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    return array


def _sequence(value: np.ndarray, *, width: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{name} must have shape [T, {width}], got {array.shape}")
    return array


def _frame_at(sequence: np.ndarray, t: int) -> np.ndarray:
    return sequence[int(np.clip(t, 0, max(sequence.shape[0] - 1, 0)))].astype(np.float32)


def _optional_feature(
    value: np.ndarray | None,
    *,
    reference: np.ndarray | None,
    t: int,
    name: str,
    zero_fill_missing: bool,
) -> np.ndarray:
    if value is None:
        if reference is None:
            if zero_fill_missing:
                return np.zeros(0, dtype=np.float32)
            raise ValueError(f"{name} is required when zero_fill_missing=False")
        ref_array = np.asarray(reference, dtype=np.float32)
        width = ref_array.shape[-1]
        return np.zeros(width, dtype=np.float32) if zero_fill_missing else _missing_feature(name)
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        return array.reshape(-1).astype(np.float32)
    if array.ndim == 2:
        return _frame_at(array, t)
    raise ValueError(f"{name} must be a vector or [T, D] array, got {array.shape}")


def _optional_vector_feature(
    value: np.ndarray | None,
    *,
    reference: np.ndarray | None,
    name: str,
    zero_fill_missing: bool,
) -> np.ndarray:
    if value is None:
        if reference is None:
            if zero_fill_missing:
                return np.zeros(0, dtype=np.float32)
            raise ValueError(f"{name} is required when zero_fill_missing=False")
        return np.zeros(np.asarray(reference, dtype=np.float32).reshape(-1).size, dtype=np.float32)
    return _vector(value, name=name)


def _missing_feature(name: str) -> np.ndarray:
    raise ValueError(f"{name} is required")
