from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from intermezzo.constants import KEY_SPLIT_LEFT_RIGHT, NUM_PIANO_KEYS
from intermezzo.kinematics import HandKinematics


RIGHT_FINGERTIPS = tuple(range(5))
LEFT_FINGERTIPS = tuple(range(5, 10))


@dataclass(frozen=True)
class FingertipAssignment:
    fingertip_index: int
    key_index: int
    hand: str
    key_xy: np.ndarray
    distance: float


def assign_active_fingertips(
    target_key_row: np.ndarray,
    *,
    endpoint_hand_state: np.ndarray,
    key_xy: np.ndarray,
    kinematics: HandKinematics,
    threshold: float = 0.5,
) -> list[FingertipAssignment]:
    """Assign active keys to same-hand fingertips from the endpoint hand pose."""
    row = np.asarray(target_key_row, dtype=np.float32).reshape(-1)
    if row.size < NUM_PIANO_KEYS:
        raise ValueError(f"target_key_row must contain at least {NUM_PIANO_KEYS} values, got {row.size}")
    keys = np.asarray(key_xy, dtype=np.float32)
    if keys.shape != (NUM_PIANO_KEYS, 2):
        raise ValueError(f"key_xy must have shape [88, 2], got {keys.shape}")
    kinematics.set_hand_state(endpoint_hand_state)
    fingertips = np.asarray(kinematics.fingertip_xy(), dtype=np.float32)
    if fingertips.shape[0] < 10 or fingertips.shape[1] < 2:
        raise ValueError(f"kinematics returned invalid fingertip xy shape {fingertips.shape}")

    active_keys = np.flatnonzero(row[:NUM_PIANO_KEYS] > float(threshold))
    assignments: list[FingertipAssignment] = []
    for hand, candidate_fingers, candidate_keys in (
        ("left", LEFT_FINGERTIPS, [int(k) for k in active_keys if int(k) < KEY_SPLIT_LEFT_RIGHT]),
        ("right", RIGHT_FINGERTIPS, [int(k) for k in active_keys if int(k) >= KEY_SPLIT_LEFT_RIGHT]),
    ):
        if not candidate_keys:
            continue
        finger_indices = np.asarray(candidate_fingers, dtype=np.int64)
        key_indices = np.asarray(candidate_keys, dtype=np.int64)
        distances = np.linalg.norm(fingertips[finger_indices, None, :2] - keys[key_indices][None, :, :], axis=2)
        for fi, ki in _match_pairs(distances):
            finger = int(finger_indices[int(fi)])
            key = int(key_indices[int(ki)])
            assignments.append(
                FingertipAssignment(
                    fingertip_index=finger,
                    key_index=key,
                    hand=hand,
                    key_xy=keys[key].astype(np.float32, copy=True),
                    distance=float(distances[int(fi), int(ki)]),
                )
            )
    assignments.sort(key=lambda item: (item.key_index, item.fingertip_index))
    return assignments


def _match_pairs(cost: np.ndarray) -> list[tuple[int, int]]:
    if cost.size == 0:
        return []
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(cost)
        return [(int(r), int(c)) for r, c in zip(rows, cols)]
    except Exception:
        return _greedy_pairs(cost)


def _greedy_pairs(cost: np.ndarray) -> list[tuple[int, int]]:
    rows = range(cost.shape[0])
    cols = range(cost.shape[1])
    candidates = sorted((float(cost[r, c]), int(r), int(c)) for r in rows for c in cols)
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    out: list[tuple[int, int]] = []
    for _distance, row, col in candidates:
        if row in used_rows or col in used_cols:
            continue
        used_rows.add(row)
        used_cols.add(col)
        out.append((row, col))
        if len(used_rows) == cost.shape[0] or len(used_cols) == cost.shape[1]:
            break
    out.sort()
    return out
