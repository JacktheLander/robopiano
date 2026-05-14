from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


NUM_FINGERS = 10
FINGER_ORDER = tuple(range(NUM_FINGERS))


@dataclass(frozen=True)
class FingerAssignmentResult:
    active_keys: np.ndarray
    assigned_finger_indices: np.ndarray
    assigned_keys: np.ndarray
    assigned_key_positions: np.ndarray
    target_positions: np.ndarray
    unassigned_keys: np.ndarray
    cost_matrix: np.ndarray
    total_cost: float
    mean_cost: float

    @property
    def count(self) -> int:
        return int(self.assigned_keys.shape[0])

    def dense_key_by_finger(self, *, fill_value: int = -1) -> np.ndarray:
        out = np.full((NUM_FINGERS,), int(fill_value), dtype=np.int32)
        if self.count:
            out[self.assigned_finger_indices.astype(np.int64)] = self.assigned_keys.astype(np.int32)
        return out

    def dense_cost_by_finger(self) -> np.ndarray:
        out = np.full((NUM_FINGERS,), np.nan, dtype=np.float32)
        if self.count:
            for finger, key_position in zip(self.assigned_finger_indices, self.assigned_key_positions):
                out[int(finger)] = float(self.cost_matrix[int(finger), int(key_position)])
        return out

    def dense_targets_by_finger(self) -> np.ndarray:
        out = np.full((NUM_FINGERS, 3), np.nan, dtype=np.float32)
        if self.count:
            out[self.assigned_finger_indices.astype(np.int64)] = self.target_positions.astype(np.float32)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_keys": self.active_keys.astype(int).tolist(),
            "assigned_finger_indices": self.assigned_finger_indices.astype(int).tolist(),
            "assigned_keys": self.assigned_keys.astype(int).tolist(),
            "unassigned_keys": self.unassigned_keys.astype(int).tolist(),
            "total_cost": float(self.total_cost),
            "mean_cost": float(self.mean_cost),
        }


def _empty_result(active_keys: np.ndarray, cost_matrix: np.ndarray) -> FingerAssignmentResult:
    return FingerAssignmentResult(
        active_keys=active_keys.astype(np.int32),
        assigned_finger_indices=np.zeros((0,), dtype=np.int32),
        assigned_keys=np.zeros((0,), dtype=np.int32),
        assigned_key_positions=np.zeros((0,), dtype=np.int32),
        target_positions=np.zeros((0, 3), dtype=np.float32),
        unassigned_keys=np.zeros((0,), dtype=np.int32),
        cost_matrix=cost_matrix.astype(np.float32, copy=False),
        total_cost=0.0,
        mean_cost=0.0,
    )


def _sorted_unique_keys_and_targets(raw_keys: np.ndarray, key_targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    targets = np.asarray(key_targets, dtype=np.float32)
    if targets.ndim != 2 or targets.shape[1] != 3:
        raise ValueError(f"key_targets must have shape [N, 3] or [88, 3], got {targets.shape}")
    if targets.shape[0] >= 88:
        keys = np.unique(raw_keys.astype(np.int32))
        keys.sort()
        if np.any(keys < 0) or np.any(keys >= targets.shape[0]):
            raise ValueError("active_keys contains an index outside key_targets")
        return keys.astype(np.int32), np.ascontiguousarray(targets[keys], dtype=np.float32)
    if targets.shape[0] != raw_keys.shape[0]:
        raise ValueError(
            f"key_targets has {targets.shape[0]} rows but active_keys has {raw_keys.shape[0]} entries"
        )
    order = np.argsort(raw_keys, kind="stable")
    sorted_keys = raw_keys[order].astype(np.int32)
    sorted_targets = targets[order].astype(np.float32)
    if sorted_keys.size == 0:
        return sorted_keys, sorted_targets
    keep = np.concatenate([np.asarray([True]), sorted_keys[1:] != sorted_keys[:-1]])
    return sorted_keys[keep].astype(np.int32), np.ascontiguousarray(sorted_targets[keep], dtype=np.float32)


def assign_fingers_previous_pose(
    active_keys: np.ndarray,
    previous_fingertips: np.ndarray,
    key_targets: np.ndarray,
    config: object | None = None,
) -> FingerAssignmentResult:
    """Assign active piano keys to fingers by RP1M-style linear assignment.

    `previous_fingertips` must be ordered left hand sites first, then right hand sites.
    `key_targets` can either be `[88, 3]` indexed by key id or `[len(active_keys), 3]`
    in the same order as `active_keys`.
    """
    del config
    raw_keys = np.asarray(active_keys, dtype=np.int32).reshape(-1)
    keys, targets = _sorted_unique_keys_and_targets(raw_keys, key_targets)
    if np.any(keys < 0) or np.any(keys >= 88):
        raise ValueError(f"active_keys must be piano key indices in [0, 87], got {keys}")

    fingers = np.asarray(previous_fingertips, dtype=np.float32)
    if fingers.shape != (NUM_FINGERS, 3):
        raise ValueError(f"previous_fingertips must have shape [10, 3], got {fingers.shape}")

    if keys.size == 0:
        return _empty_result(keys, np.zeros((NUM_FINGERS, 0), dtype=np.float32))

    diff = fingers[:, None, :] - targets[None, :, :]
    cost = np.linalg.norm(diff, axis=2).astype(np.float64)
    # Deterministic tie break: lower finger index, then lower sorted-key position.
    cost += np.arange(NUM_FINGERS, dtype=np.float64)[:, None] * 1e-9
    cost += np.arange(keys.shape[0], dtype=np.float64)[None, :] * 1e-12

    row_ind, col_ind = linear_sum_assignment(cost)
    order = np.lexsort((keys[col_ind], row_ind))
    row_ind = row_ind[order].astype(np.int32)
    col_ind = col_ind[order].astype(np.int32)

    assigned_keys = keys[col_ind].astype(np.int32)
    assigned_targets = targets[col_ind].astype(np.float32)
    unassigned_mask = np.ones((keys.shape[0],), dtype=bool)
    unassigned_mask[col_ind] = False
    selected_costs = cost[row_ind, col_ind].astype(np.float32)

    return FingerAssignmentResult(
        active_keys=keys.astype(np.int32),
        assigned_finger_indices=row_ind,
        assigned_keys=assigned_keys,
        assigned_key_positions=col_ind,
        target_positions=assigned_targets,
        unassigned_keys=keys[unassigned_mask].astype(np.int32),
        cost_matrix=cost.astype(np.float32),
        total_cost=float(np.sum(selected_costs)),
        mean_cost=float(np.mean(selected_costs)) if selected_costs.size else 0.0,
    )
