from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np


DEFAULT_PHASE_NAMES: tuple[str, ...] = (
    "approach",
    "pre_contact",
    "contact",
    "hold",
    "release",
    "recovery",
    "unknown",
)

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
    "rest": "unknown",
    "unknown": "unknown",
}


@dataclass(frozen=True, slots=True)
class FingertipFeatureSpec:
    include_current: bool = True
    include_desired: bool = True
    include_error: bool = True
    include_weights: bool = True
    include_active_mask: bool = True
    include_inactive_mask: bool = True
    flatten: bool = True
    allow_missing: bool = False
    missing_fill_value: float = 0.0


@dataclass(frozen=True, slots=True)
class PhaseFeatureSpec:
    source: str = "metadata"
    encode_as: str = "both"
    include_mask: bool = True
    allow_missing: bool = False
    missing_fill_value: float = 0.0
    phase_names: tuple[str, ...] = DEFAULT_PHASE_NAMES


def build_fingertip_features(
    *,
    current_fingertips: np.ndarray | None = None,
    desired_fingertips: np.ndarray | None = None,
    fingertip_error: np.ndarray | None = None,
    fingertip_weights: np.ndarray | None = None,
    active_finger_mask: np.ndarray | None = None,
    inactive_finger_mask: np.ndarray | None = None,
    spec: FingertipFeatureSpec | None = None,
) -> np.ndarray:
    spec = spec or FingertipFeatureSpec()
    shape = _resolve_fingertip_shape(
        current_fingertips,
        desired_fingertips,
        fingertip_error,
        fingertip_weights,
        active_finger_mask,
        inactive_finger_mask,
    )
    if shape is None:
        if not spec.allow_missing:
            raise ValueError("At least one fingertip input is required unless allow_missing=True")
        return _finalize_feature([np.full((1, 1), spec.missing_fill_value, dtype=np.float32)], spec.flatten)

    num_fingers, dims = shape
    pieces: list[np.ndarray] = []
    current = _coerce_fingertip_matrix(
        current_fingertips,
        shape=shape,
        allow_missing=spec.allow_missing,
        fill_value=spec.missing_fill_value,
        name="current_fingertips",
    )
    desired = _coerce_fingertip_matrix(
        desired_fingertips,
        shape=shape,
        allow_missing=spec.allow_missing,
        fill_value=spec.missing_fill_value,
        name="desired_fingertips",
    )
    if spec.include_current:
        pieces.append(current)
    if spec.include_desired:
        pieces.append(desired)

    if spec.include_error:
        if fingertip_error is not None:
            error = _coerce_fingertip_matrix(
                fingertip_error,
                shape=shape,
                allow_missing=False,
                fill_value=0.0,
                name="fingertip_error",
            )
        elif current_fingertips is not None and desired_fingertips is not None:
            error = desired - current
        elif spec.allow_missing:
            error = np.full((num_fingers, dims), spec.missing_fill_value, dtype=np.float32)
        else:
            raise ValueError("fingertip_error requires explicit data or both current and desired fingertips")
        pieces.append(error)

    if spec.include_weights:
        pieces.append(
            _coerce_finger_vector(
                fingertip_weights,
                length=num_fingers,
                allow_missing=spec.allow_missing,
                fill_value=1.0 if fingertip_weights is None else spec.missing_fill_value,
                name="fingertip_weights",
            )[:, None]
        )
    if spec.include_active_mask:
        pieces.append(
            _coerce_finger_vector(
                active_finger_mask,
                length=num_fingers,
                allow_missing=spec.allow_missing,
                fill_value=0.0,
                name="active_finger_mask",
            )[:, None]
        )
    if spec.include_inactive_mask:
        pieces.append(
            _coerce_finger_vector(
                inactive_finger_mask,
                length=num_fingers,
                allow_missing=spec.allow_missing,
                fill_value=0.0,
                name="inactive_finger_mask",
            )[:, None]
        )
    return _finalize_feature(pieces, spec.flatten)


def build_phase_features(
    *,
    t: int,
    metadata: dict[str, Any] | None = None,
    plan_bundle: Any | None = None,
    target_keys: np.ndarray | None = None,
    spec: PhaseFeatureSpec | None = None,
) -> np.ndarray:
    spec = spec or PhaseFeatureSpec()
    phase_info = resolve_phase_state(
        t=t,
        metadata=metadata,
        plan_bundle=plan_bundle,
        target_keys=target_keys,
        phase_names=spec.phase_names,
        allow_missing=spec.allow_missing,
        fill_value=spec.missing_fill_value,
    )
    pieces: list[np.ndarray] = []
    if spec.encode_as in {"one_hot", "both"}:
        pieces.append(phase_info["one_hot"])
    if spec.encode_as in {"scalar", "both"}:
        pieces.append(np.asarray([phase_info["scalar"]], dtype=np.float32))
    if spec.include_mask:
        pieces.append(np.asarray([phase_info["mask"]], dtype=np.float32))
    return np.concatenate(pieces, dtype=np.float32)


def build_fingertip_phase_features(
    *,
    t: int,
    metadata: dict[str, Any] | None = None,
    plan_bundle: Any | None = None,
    target_keys: np.ndarray | None = None,
    fingertip_spec: FingertipFeatureSpec | None = None,
    phase_spec: PhaseFeatureSpec | None = None,
    **kwargs: Any,
) -> np.ndarray:
    fingertip_features = build_fingertip_features(spec=fingertip_spec, **kwargs)
    phase_features = build_phase_features(
        t=t,
        metadata=metadata,
        plan_bundle=plan_bundle,
        target_keys=target_keys,
        spec=phase_spec,
    )
    return np.concatenate([fingertip_features.reshape(-1), phase_features.reshape(-1)]).astype(np.float32)


def resolve_phase_state(
    *,
    t: int,
    metadata: dict[str, Any] | None = None,
    plan_bundle: Any | None = None,
    target_keys: np.ndarray | None = None,
    phase_names: Sequence[str] = DEFAULT_PHASE_NAMES,
    allow_missing: bool = False,
    fill_value: float = 0.0,
) -> dict[str, np.ndarray | float | int | str]:
    phase_names = tuple(phase_names)
    num_phases = len(phase_names)
    phase_sequence = _find_phase_sequence(metadata, plan_bundle, num_phases)
    scalar_sequence = _find_phase_scalar_sequence(metadata, plan_bundle)
    mask_sequence = _find_phase_mask_sequence(metadata, plan_bundle)
    index = _resolve_time_index(t, phase_sequence, scalar_sequence, mask_sequence, target_keys)

    if phase_sequence is not None:
        phase_id = int(np.clip(phase_sequence[index], 0, num_phases - 1))
    else:
        inferred = infer_phase_from_target_keys(target_keys, t=index, phase_names=phase_names)
        if inferred is None:
            if not allow_missing:
                raise ValueError("Phase data is missing and could not be inferred from target_keys")
            return {
                "phase_id": -1,
                "phase_name": "missing",
                "one_hot": np.full(num_phases, fill_value, dtype=np.float32),
                "scalar": float(fill_value),
                "mask": float(fill_value),
            }
        phase_id = inferred

    if scalar_sequence is not None:
        scalar = float(np.asarray(scalar_sequence, dtype=np.float32)[index])
    else:
        scalar = _default_phase_scalar(phase_id, num_phases)

    if mask_sequence is not None:
        mask = float(np.asarray(mask_sequence, dtype=np.float32)[index])
    else:
        mask = 0.0 if phase_names[phase_id] == "unknown" else 1.0

    one_hot = np.zeros(num_phases, dtype=np.float32)
    one_hot[phase_id] = 1.0
    return {
        "phase_id": phase_id,
        "phase_name": phase_names[phase_id],
        "one_hot": one_hot,
        "scalar": scalar,
        "mask": mask,
    }


def infer_phase_from_target_keys(
    target_keys: np.ndarray | None,
    *,
    t: int,
    phase_names: Sequence[str] = DEFAULT_PHASE_NAMES,
) -> int | None:
    if target_keys is None:
        return None
    target = _coerce_time_matrix(target_keys, name="target_keys")
    index = _bounded_index(t, target.shape[0])
    current_active = bool(np.any(target[index] > 0))
    prev_active = bool(np.any(target[max(0, index - 1)] > 0))
    future_active = bool(np.any(target[min(target.shape[0] - 1, index + 1)] > 0))
    phase_to_id = {_canonicalize_phase_name(name): i for i, name in enumerate(phase_names)}
    if current_active and prev_active:
        return phase_to_id.get("hold", phase_to_id.get("contact", 0))
    if current_active:
        return phase_to_id.get("contact", min(len(phase_names) - 1, 2))
    if prev_active and not current_active:
        return phase_to_id.get("release", min(len(phase_names) - 1, 4))
    if future_active and not current_active:
        return phase_to_id.get("approach", min(len(phase_names) - 1, 0))
    return phase_to_id.get("unknown", len(phase_names) - 1)


def _resolve_fingertip_shape(*values: np.ndarray | None) -> tuple[int, int] | None:
    for value in values:
        if value is None:
            continue
        array = np.asarray(value, dtype=np.float32)
        if array.ndim == 1:
            return int(array.shape[0]), 1
        if array.ndim == 2:
            return int(array.shape[0]), int(array.shape[1])
        raise ValueError(f"Expected fingertip data with ndim 1 or 2, got {array.shape}")
    return None


def _coerce_fingertip_matrix(
    value: np.ndarray | None,
    *,
    shape: tuple[int, int],
    allow_missing: bool,
    fill_value: float,
    name: str,
) -> np.ndarray:
    if value is None:
        if not allow_missing:
            raise ValueError(f"{name} is required")
        return np.full(shape, fill_value, dtype=np.float32)
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = array[:, None]
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    return array.astype(np.float32)


def _coerce_finger_vector(
    value: np.ndarray | None,
    *,
    length: int,
    allow_missing: bool,
    fill_value: float,
    name: str,
) -> np.ndarray:
    if value is None:
        if not allow_missing:
            raise ValueError(f"{name} is required")
        return np.full(length, fill_value, dtype=np.float32)
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {array.shape}")
    return array.astype(np.float32)


def _finalize_feature(parts: Iterable[np.ndarray], flatten: bool) -> np.ndarray:
    arrays = [np.asarray(part, dtype=np.float32) for part in parts]
    if flatten:
        return np.concatenate([part.reshape(-1) for part in arrays]).astype(np.float32)
    return np.concatenate(arrays, axis=-1).astype(np.float32)


def _find_phase_sequence(
    metadata: dict[str, Any] | None,
    plan_bundle: Any | None,
    num_phases: int,
) -> np.ndarray | None:
    candidates = _find_value(metadata, plan_bundle, ("phase_ids", "phases", "phase"))
    if candidates is None:
        return None
    array = np.asarray(candidates)
    if array.ndim == 2 and array.shape[-1] == num_phases:
        return np.argmax(array, axis=-1).astype(np.int64)
    if array.ndim == 1 and array.dtype.kind in {"U", "S", "O"}:
        names = tuple(DEFAULT_PHASE_NAMES[:num_phases])
        return np.asarray(
            [_phase_name_to_id(item, names, fallback="unknown") for item in array.tolist()],
            dtype=np.int64,
        )
    if array.ndim != 1:
        raise ValueError(f"phase sequence must be [T] or [T, P], got {array.shape}")
    return array.astype(np.int64)


def _find_phase_scalar_sequence(metadata: dict[str, Any] | None, plan_bundle: Any | None) -> np.ndarray | None:
    value = _find_value(metadata, plan_bundle, ("phase_scalar", "phase_progress", "note_progress"))
    if value is None:
        return None
    return np.asarray(value, dtype=np.float32).reshape(-1)


def _find_phase_mask_sequence(metadata: dict[str, Any] | None, plan_bundle: Any | None) -> np.ndarray | None:
    value = _find_value(metadata, plan_bundle, ("phase_mask", "phase_active_mask"))
    if value is None:
        return None
    return np.asarray(value, dtype=np.float32).reshape(-1)


def _find_value(metadata: dict[str, Any] | None, plan_bundle: Any | None, keys: Sequence[str]) -> Any | None:
    for container in (metadata, _container_from_bundle(plan_bundle)):
        if container is None:
            continue
        for key in keys:
            if key in container:
                return container[key]
    return None


def _container_from_bundle(plan_bundle: Any | None) -> dict[str, Any] | None:
    if plan_bundle is None:
        return None
    if isinstance(plan_bundle, dict):
        return plan_bundle
    values = {}
    for name in ("metadata", "phase_ids", "phase_scalar", "phase_mask", "target_keys"):
        if hasattr(plan_bundle, name):
            values[name] = getattr(plan_bundle, name)
    return values or None


def _resolve_time_index(
    t: int,
    phase_sequence: np.ndarray | None,
    scalar_sequence: np.ndarray | None,
    mask_sequence: np.ndarray | None,
    target_keys: np.ndarray | None,
) -> int:
    lengths = [
        sequence.shape[0]
        for sequence in (phase_sequence, scalar_sequence, mask_sequence)
        if sequence is not None
    ]
    if target_keys is not None:
        lengths.append(_coerce_time_matrix(target_keys, name="target_keys").shape[0])
    if not lengths:
        return max(0, int(t))
    return _bounded_index(t, min(lengths))


def _coerce_time_matrix(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape [T, D], got {array.shape}")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must have at least one timestep")
    return array


def _default_phase_scalar(phase_id: int, num_phases: int) -> float:
    if num_phases <= 1:
        return 0.0
    return float(phase_id / float(num_phases - 1))


def _bounded_index(index: int, length: int) -> int:
    return int(np.clip(index, 0, max(length - 1, 0)))


def _canonicalize_phase_name(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return _PHASE_ALIASES.get(normalized, normalized)


def _phase_name_to_id(value: Any, phase_names: Sequence[str], *, fallback: str) -> int:
    canonical_names = [_canonicalize_phase_name(name) for name in phase_names]
    canonical_value = _canonicalize_phase_name(value)
    if canonical_value in canonical_names:
        return canonical_names.index(canonical_value)
    fallback_value = _canonicalize_phase_name(fallback)
    if fallback_value in canonical_names:
        return canonical_names.index(fallback_value)
    return len(canonical_names) - 1
