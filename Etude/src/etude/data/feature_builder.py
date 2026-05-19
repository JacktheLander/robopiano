from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    lookahead_steps: tuple[int, ...] = (1, 5, 10)
    include_target_keys: bool = True
    include_fingertips: bool = True


def build_tracking_features(
    *,
    q: np.ndarray,
    qdot: np.ndarray,
    q_ref: np.ndarray,
    qdot_ref: np.ndarray,
    t: int,
    previous_action: np.ndarray,
    target_keys: np.ndarray | None = None,
    fingertips: np.ndarray | None = None,
    spec: FeatureSpec | None = None,
) -> np.ndarray:
    spec = spec or FeatureSpec()
    q = _vector(q, name="q")
    qdot = _vector(qdot, name="qdot")
    previous_action = _vector(previous_action, name="previous_action")
    q_ref = _matrix(q_ref, name="q_ref")
    qdot_ref = _matrix(qdot_ref, name="qdot_ref")
    if q.shape[0] != q_ref.shape[1]:
        raise ValueError("q dimension must match q_ref feature dimension")
    if qdot.shape != q.shape:
        raise ValueError("qdot must match q")
    if qdot_ref.shape != q_ref.shape:
        raise ValueError("qdot_ref must match q_ref")
    index = _bounded_index(t, q_ref.shape[0])
    pieces: list[np.ndarray] = [q, qdot, q_ref[index], qdot_ref[index]]
    pieces.extend(q_ref[_bounded_index(index + step, q_ref.shape[0])] for step in spec.lookahead_steps)
    pieces.append(previous_action)
    if spec.include_target_keys and target_keys is not None:
        target_matrix = _matrix(target_keys, name="target_keys")
        pieces.append(target_matrix[_bounded_index(index, target_matrix.shape[0])])
    if spec.include_fingertips and fingertips is not None:
        fingertip_matrix = _matrix(fingertips, name="fingertips")
        pieces.append(fingertip_matrix[_bounded_index(index, fingertip_matrix.shape[0])])
    return _concat_float32(pieces)


def _vector(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    return array


def _matrix(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape [T, D], got {array.shape}")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must have at least one row")
    return array


def _bounded_index(index: int, length: int) -> int:
    return int(np.clip(index, 0, length - 1))


def _concat_float32(parts: Iterable[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(part, dtype=np.float32).reshape(-1) for part in parts]).astype(np.float32)
