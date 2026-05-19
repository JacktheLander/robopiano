from __future__ import annotations

from typing import Any

import numpy as np


def _as_float_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def _safe_mean(value: Any) -> float:
    array = _as_float_array(value)
    if array.size == 0:
        return 0.0
    return float(np.mean(array))


def _resolve_state_array(
    state: dict[str, Any],
    key: str,
    *,
    default: Any = None,
    required: bool = False,
) -> np.ndarray:
    if key in state:
        return _as_float_array(state[key])
    if required:
        raise KeyError(f"Missing required state key: {key}")
    if default is None:
        return np.asarray(0.0, dtype=np.float32)
    return _as_float_array(default)


def target_key_activation_reward(
    state: dict[str, Any],
    *_: dict[str, Any],
    **__: Any,
) -> float:
    predicted = _resolve_state_array(state, "predicted_keys", required=True)
    target = _resolve_state_array(state, "target_keys", required=True)
    return float(np.sum(np.clip(predicted, 0.0, 1.0) * np.clip(target, 0.0, 1.0)))


def missed_key_penalty(
    state: dict[str, Any],
    *_: dict[str, Any],
    **__: Any,
) -> float:
    predicted = _resolve_state_array(state, "predicted_keys", required=True)
    target = _resolve_state_array(state, "target_keys", required=True)
    misses = np.clip(target - predicted, 0.0, None)
    return float(np.sum(misses))


def wrong_key_penalty(
    state: dict[str, Any],
    *_: dict[str, Any],
    **__: Any,
) -> float:
    predicted = _resolve_state_array(state, "predicted_keys", required=True)
    target = _resolve_state_array(state, "target_keys", required=True)
    wrong = np.clip(predicted - target, 0.0, None)
    return float(np.sum(wrong))


def timing_error_penalty(
    state: dict[str, Any],
    *_: dict[str, Any],
    **__: Any,
) -> float:
    error = _resolve_state_array(state, "timing_error", default=0.0)
    return _safe_mean(np.abs(error))


def release_error_penalty(
    state: dict[str, Any],
    *_: dict[str, Any],
    **__: Any,
) -> float:
    error = _resolve_state_array(state, "release_error", default=0.0)
    return _safe_mean(np.abs(error))


def fingertip_tracking_penalty(
    state: dict[str, Any],
    *_: dict[str, Any],
    **__: Any,
) -> float:
    current = _resolve_state_array(state, "fingertip_positions", required=True)
    target = _resolve_state_array(state, "target_fingertip_positions", required=True)
    return _safe_mean(np.linalg.norm(current - target, axis=-1))


def joint_tracking_penalty(
    state: dict[str, Any],
    *_: dict[str, Any],
    **__: Any,
) -> float:
    current = _resolve_state_array(state, "joint_positions", required=True)
    target = _resolve_state_array(state, "target_joint_positions", required=True)
    return _safe_mean(np.square(current - target))


def action_l2_penalty(
    state: dict[str, Any],
    *_: dict[str, Any],
    **__: Any,
) -> float:
    action = _resolve_state_array(state, "action", required=True)
    return _safe_mean(np.square(action))


def action_smoothness_penalty(
    state: dict[str, Any],
    *_: dict[str, Any],
    **__: Any,
) -> float:
    action = _resolve_state_array(state, "action", required=True)
    previous_action = _resolve_state_array(state, "previous_action", default=np.zeros_like(action))
    return _safe_mean(np.square(action - previous_action))


def residual_l2_penalty(
    state: dict[str, Any],
    *_: dict[str, Any],
    **__: Any,
) -> float:
    residual = _resolve_state_array(state, "residual_action", default=state.get("action"), required=False)
    return _safe_mean(np.square(residual))


REWARD_TERMS = {
    "target_key_activation": target_key_activation_reward,
    "missed_key": missed_key_penalty,
    "wrong_key": wrong_key_penalty,
    "timing_error": timing_error_penalty,
    "release_error": release_error_penalty,
    "fingertip_tracking": fingertip_tracking_penalty,
    "joint_tracking": joint_tracking_penalty,
    "action_l2": action_l2_penalty,
    "action_smoothness": action_smoothness_penalty,
    "residual_l2": residual_l2_penalty,
}
