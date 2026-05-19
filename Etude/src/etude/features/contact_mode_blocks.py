from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


DEFAULT_CONTACT_MODE_NAMES: tuple[str, ...] = ("inactive", "approach", "press", "release")


@dataclass(frozen=True, slots=True)
class ContactModeFeatureSpec:
    include_one_hot: bool = True
    include_active_mask: bool = True
    include_inactive_mask: bool = True
    include_distance_to_key: bool = True
    include_press_release_intent: bool = True
    flatten: bool = True
    allow_missing: bool = False
    missing_fill_value: float = 0.0
    contact_mode_names: tuple[str, ...] = DEFAULT_CONTACT_MODE_NAMES


def build_contact_mode_one_hot(
    contact_mode_ids: np.ndarray,
    *,
    mode_names: Sequence[str] = DEFAULT_CONTACT_MODE_NAMES,
) -> np.ndarray:
    mode_ids = _coerce_contact_mode_ids(contact_mode_ids, name="contact_mode_ids")
    num_modes = len(tuple(mode_names))
    one_hot = np.zeros((mode_ids.shape[0], num_modes), dtype=np.float32)
    one_hot[np.arange(mode_ids.shape[0]), np.clip(mode_ids, 0, num_modes - 1)] = 1.0
    return one_hot


def build_active_inactive_finger_masks(
    contact_mode_ids: np.ndarray,
    *,
    active_mode_ids: Iterable[int] = (1, 2, 3),
) -> tuple[np.ndarray, np.ndarray]:
    mode_ids = _coerce_contact_mode_ids(contact_mode_ids, name="contact_mode_ids")
    active_lookup = np.isin(mode_ids, np.asarray(tuple(active_mode_ids), dtype=np.int64))
    active_mask = active_lookup.astype(np.float32)
    inactive_mask = 1.0 - active_mask
    return active_mask, inactive_mask


def build_distance_to_assigned_key(
    distance_to_assigned_key: np.ndarray | None,
    *,
    num_fingers: int | None = None,
    allow_missing: bool = False,
    fill_value: float = 0.0,
) -> np.ndarray:
    if distance_to_assigned_key is None:
        if not allow_missing or num_fingers is None:
            raise ValueError("distance_to_assigned_key requires explicit data or num_fingers with allow_missing=True")
        return np.full(num_fingers, fill_value, dtype=np.float32)
    distance = np.asarray(distance_to_assigned_key, dtype=np.float32).reshape(-1)
    if distance.size == 0:
        raise ValueError("distance_to_assigned_key must contain at least one value")
    return distance.astype(np.float32)


def build_press_release_intent_features(
    *,
    press_intent: np.ndarray | None = None,
    release_intent: np.ndarray | None = None,
    num_fingers: int | None = None,
    allow_missing: bool = False,
    fill_value: float = 0.0,
) -> np.ndarray:
    resolved_num_fingers = _resolve_num_fingers(press_intent, release_intent, num_fingers)
    press = _coerce_optional_vector(
        press_intent,
        length=resolved_num_fingers,
        allow_missing=allow_missing,
        fill_value=fill_value,
        name="press_intent",
    )
    release = _coerce_optional_vector(
        release_intent,
        length=resolved_num_fingers,
        allow_missing=allow_missing,
        fill_value=fill_value,
        name="release_intent",
    )
    return np.concatenate([press, release]).astype(np.float32)


def build_contact_mode_features(
    *,
    contact_mode_ids: np.ndarray | None = None,
    distance_to_assigned_key: np.ndarray | None = None,
    press_intent: np.ndarray | None = None,
    release_intent: np.ndarray | None = None,
    spec: ContactModeFeatureSpec | None = None,
) -> np.ndarray:
    spec = spec or ContactModeFeatureSpec()
    num_fingers = _resolve_num_fingers(contact_mode_ids, distance_to_assigned_key, press_intent, release_intent)
    if num_fingers is None:
        if not spec.allow_missing:
            raise ValueError("At least one contact feature input is required unless allow_missing=True")
        return np.full((1,), spec.missing_fill_value, dtype=np.float32)

    mode_ids = _coerce_optional_vector(
        contact_mode_ids,
        length=num_fingers,
        allow_missing=spec.allow_missing,
        fill_value=0.0,
        name="contact_mode_ids",
    ).astype(np.int64)

    parts: list[np.ndarray] = []
    if spec.include_one_hot:
        parts.append(build_contact_mode_one_hot(mode_ids, mode_names=spec.contact_mode_names))
    active_mask, inactive_mask = build_active_inactive_finger_masks(mode_ids)
    if spec.include_active_mask:
        parts.append(active_mask[:, None])
    if spec.include_inactive_mask:
        parts.append(inactive_mask[:, None])
    if spec.include_distance_to_key:
        parts.append(
            build_distance_to_assigned_key(
                distance_to_assigned_key,
                num_fingers=num_fingers,
                allow_missing=spec.allow_missing,
                fill_value=spec.missing_fill_value,
            )[:, None]
        )
    if spec.include_press_release_intent:
        intent = build_press_release_intent_features(
            press_intent=press_intent,
            release_intent=release_intent,
            num_fingers=num_fingers,
            allow_missing=spec.allow_missing,
            fill_value=spec.missing_fill_value,
        ).reshape(2, num_fingers).T
        parts.append(intent)
    return _finalize_features(parts, flatten=spec.flatten)


def _resolve_num_fingers(*values: np.ndarray | int | None) -> int | None:
    for value in values:
        if value is None:
            continue
        if isinstance(value, int):
            return int(value)
        array = np.asarray(value)
        if array.ndim == 0:
            return 1
        return int(array.reshape(-1).shape[0])
    return None


def _coerce_contact_mode_ids(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.int64).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    return array


def _coerce_optional_vector(
    value: np.ndarray | None,
    *,
    length: int | None,
    allow_missing: bool,
    fill_value: float,
    name: str,
) -> np.ndarray:
    if value is None:
        if not allow_missing or length is None:
            raise ValueError(f"{name} is required")
        return np.full(length, fill_value, dtype=np.float32)
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if length is not None and array.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {array.shape}")
    return array.astype(np.float32)


def _finalize_features(parts: Sequence[np.ndarray], *, flatten: bool) -> np.ndarray:
    arrays = [np.asarray(part, dtype=np.float32) for part in parts]
    if flatten:
        return np.concatenate([part.reshape(-1) for part in arrays]).astype(np.float32)
    return np.concatenate(arrays, axis=-1).astype(np.float32)
