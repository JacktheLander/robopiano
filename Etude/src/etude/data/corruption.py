from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from etude.core.plan_bundle import PlanBundle
from etude.data.corruption_config import PlanCorruptionConfig


def corrupt_q_ref(
    q_ref: np.ndarray,
    config: PlanCorruptionConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    values = _matrix(q_ref, name="q_ref")
    if not config.enabled:
        return values.copy()
    scale = float(config.curriculum_scale)
    if scale <= 0.0:
        return values.copy()
    corrupted = values.copy()
    noise_std = config.scaled_float(config.q_reference_noise_std)
    if noise_std > 0.0:
        corrupted += rng.normal(0.0, noise_std, size=corrupted.shape).astype(np.float32)
    drift_std = config.scaled_float(config.q_smooth_drift_std)
    if drift_std > 0.0 and corrupted.shape[0] > 1:
        start = rng.normal(0.0, drift_std, size=(1, corrupted.shape[1])).astype(np.float32)
        end = rng.normal(0.0, drift_std, size=(1, corrupted.shape[1])).astype(np.float32)
        alpha = np.linspace(0.0, 1.0, corrupted.shape[0], dtype=np.float32)[:, None]
        corrupted += start + (end - start) * alpha
    return corrupted.astype(np.float32)


def corrupt_fingertip_ref(
    fingertip_ref: np.ndarray,
    config: PlanCorruptionConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    values = _matrix(fingertip_ref, name="fingertip_ref")
    if values.shape[1] % 3 != 0:
        raise ValueError(f"fingertip_ref width must be divisible by 3, got {values.shape[1]}")
    if not config.enabled:
        return values.copy()
    scale = float(config.curriculum_scale)
    if scale <= 0.0:
        return values.copy()
    reshaped = values.reshape(values.shape[0], values.shape[1] // 3, 3).copy()
    xy_std = config.scaled_float(config.fingertip_xy_noise_std)
    if xy_std > 0.0:
        reshaped[..., :2] += rng.normal(0.0, xy_std, size=reshaped[..., :2].shape).astype(np.float32)
    z_std = config.scaled_float(config.fingertip_z_noise_std)
    if z_std > 0.0:
        reshaped[..., 2] += rng.normal(0.0, z_std, size=reshaped[..., 2].shape).astype(np.float32)
    hover_bias = config.scaled_float(config.hover_height_bias)
    press_bias = config.scaled_float(config.press_depth_bias)
    if hover_bias > 0.0 or press_bias > 0.0:
        median_z = np.median(reshaped[..., 2], axis=0, keepdims=True)
        hover_mask = reshaped[..., 2] >= median_z
        press_mask = np.logical_not(hover_mask)
        if hover_bias > 0.0:
            reshaped[..., 2] += hover_mask.astype(np.float32) * hover_bias
        if press_bias > 0.0:
            reshaped[..., 2] -= press_mask.astype(np.float32) * press_bias
    return reshaped.reshape(values.shape).astype(np.float32)


def corrupt_timing(
    values: np.ndarray,
    timing_jitter_frames: int,
    rng: np.random.Generator,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 0:
        raise ValueError("timing corruption requires at least one temporal dimension")
    jitter = int(timing_jitter_frames)
    if jitter <= 0 or array.shape[0] <= 1:
        return array.copy()
    offsets = rng.integers(-jitter, jitter + 1, size=array.shape[0], endpoint=False)
    indices = np.clip(np.arange(array.shape[0]) + offsets, 0, array.shape[0] - 1)
    return array[indices].astype(np.float32)


def drop_waypoints(
    values: np.ndarray,
    missing_waypoint_probability: float,
    rng: np.random.Generator,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 0:
        raise ValueError("waypoint dropout requires at least one temporal dimension")
    probability = float(missing_waypoint_probability)
    if probability <= 0.0 or array.shape[0] <= 1:
        return array.copy()
    if probability >= 1.0:
        return np.repeat(array[:1], array.shape[0], axis=0).astype(np.float32)
    keep_mask = rng.random(array.shape[0]) >= probability
    keep_mask[0] = True
    source_indices = np.maximum.accumulate(np.where(keep_mask, np.arange(array.shape[0]), 0))
    return array[source_indices].astype(np.float32)


def drop_lookahead(
    lookahead: np.ndarray,
    lookahead_dropout_probability: float,
    rng: np.random.Generator,
) -> np.ndarray:
    array = np.asarray(lookahead, dtype=np.float32)
    if array.ndim < 2:
        raise ValueError(f"lookahead dropout expects at least 2 dimensions, got {array.shape}")
    probability = float(lookahead_dropout_probability)
    if probability <= 0.0 or array.shape[1] <= 1:
        return array.copy()
    dropped = array.copy()
    mask = rng.random(array.shape[:2]) < probability
    mask[:, 0] = False
    expanded_mask = mask.reshape(mask.shape + (1,) * (array.ndim - 2))
    return np.where(expanded_mask, 0.0, dropped).astype(np.float32)


def apply_plan_corruption(
    plan_bundle: PlanBundle,
    config: PlanCorruptionConfig,
    rng: np.random.Generator,
) -> PlanBundle:
    if not isinstance(plan_bundle, PlanBundle):
        raise TypeError(f"plan_bundle must be a PlanBundle, got {type(plan_bundle)!r}")
    plan_bundle.validate_step_aligned()
    q_ref = _matrix(plan_bundle.q_ref, name="plan_bundle.q_ref").copy()
    fingertip_ref = _optional_timed_array(plan_bundle.fingertip_ref)
    phase = _optional_timed_array(plan_bundle.phase)
    metadata = _copy_metadata(plan_bundle.metadata)

    if config.enabled and config.curriculum_scale > 0.0:
        q_ref = corrupt_q_ref(q_ref, config, rng)
        q_ref = corrupt_timing(q_ref, config.scaled_frames(config.timing_jitter_frames), rng)
        q_ref = drop_waypoints(q_ref, config.scaled_probability(config.missing_waypoint_probability), rng)
        if fingertip_ref is not None:
            fingertip_ref = corrupt_fingertip_ref(fingertip_ref, config, rng)
            fingertip_ref = corrupt_timing(
                fingertip_ref,
                config.scaled_frames(config.timing_jitter_frames),
                rng,
            )
            fingertip_ref = drop_waypoints(
                fingertip_ref,
                config.scaled_probability(config.missing_waypoint_probability),
                rng,
            )
        if phase is not None:
            phase = corrupt_timing(phase, config.scaled_frames(config.timing_jitter_frames), rng)
            phase = drop_waypoints(phase, config.scaled_probability(config.missing_waypoint_probability), rng)
        metadata = _drop_lookahead_metadata(
            metadata,
            config.scaled_probability(config.lookahead_dropout_probability),
            rng,
        )

    qdot_ref = None
    if plan_bundle.qdot_ref is not None:
        qdot_ref = _finite_difference(q_ref, plan_bundle.dt)

    corrupted = PlanBundle(
        q_ref=q_ref,
        qdot_ref=qdot_ref,
        dt=plan_bundle.dt,
        target_keys=plan_bundle.target_keys.copy() if plan_bundle.target_keys is not None else None,
        fingertip_ref=fingertip_ref.copy() if fingertip_ref is not None else None,
        fingertip_weights=plan_bundle.fingertip_weights.copy() if plan_bundle.fingertip_weights is not None else None,
        assignments=deepcopy(plan_bundle.assignments),
        phase=phase.copy() if phase is not None else None,
        metadata=metadata,
    )
    corrupted.validate_step_aligned()
    return corrupted


def _matrix(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape [T, D], got {array.shape}")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must have at least one row")
    return array


def _optional_timed_array(value: np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        raise ValueError("timed arrays must have at least one dimension")
    return array.copy()


def _copy_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, np.ndarray):
            copied[key] = value.copy()
        else:
            copied[key] = deepcopy(value)
    return copied


def _drop_lookahead_metadata(
    metadata: dict[str, Any],
    dropout_probability: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    if dropout_probability <= 0.0:
        return metadata
    updated = dict(metadata)
    for key, value in metadata.items():
        if "lookahead" not in key.lower():
            continue
        if not isinstance(value, np.ndarray) or value.ndim < 2:
            continue
        updated[key] = drop_lookahead(value, dropout_probability, rng)
    return updated


def _finite_difference(q_ref: np.ndarray, dt: float) -> np.ndarray:
    if q_ref.shape[0] <= 1:
        return np.zeros_like(q_ref, dtype=np.float32)
    return np.gradient(q_ref, float(dt), axis=0).astype(np.float32)
