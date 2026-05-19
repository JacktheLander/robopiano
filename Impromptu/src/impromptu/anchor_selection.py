from __future__ import annotations

import numpy as np

from impromptu.config import ImpromptuConfig


def _changed_target_frames(targets: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if targets.shape[0] <= 1:
        return np.zeros((0,), dtype=np.int64)
    finite = np.isfinite(targets).all(axis=2) & (weights > 0.0)
    changed: list[int] = []
    for frame in range(1, targets.shape[0]):
        if not np.array_equal(finite[frame], finite[frame - 1]):
            changed.append(frame)
            continue
        mask = finite[frame] & finite[frame - 1]
        if bool(mask.any()) and not np.allclose(targets[frame, mask], targets[frame - 1, mask], atol=1e-6, equal_nan=True):
            changed.append(frame)
    return np.asarray(changed, dtype=np.int64)


def select_ik_anchor_frames(
    *,
    fingertip_targets: np.ndarray,
    fingertip_weights: np.ndarray,
    waypoint_frames_dense: np.ndarray,
    config: ImpromptuConfig,
) -> np.ndarray:
    targets = np.asarray(fingertip_targets, dtype=np.float32)
    weights = np.asarray(fingertip_weights, dtype=np.float32)
    if targets.ndim != 3 or targets.shape[1:] != (10, 3):
        raise ValueError(f"fingertip_targets must have shape [T, 10, 3], got {targets.shape}")
    if weights.shape != targets.shape[:2]:
        raise ValueError(f"fingertip_weights must have shape [T, 10], got {weights.shape}")
    total = int(targets.shape[0])
    if total == 0:
        return np.zeros((0,), dtype=np.int64)

    anchors: set[int] = {0, total - 1}
    waypoints = np.asarray(waypoint_frames_dense, dtype=np.int64).reshape(-1)
    for frame in waypoints:
        if 0 <= int(frame) < total:
            anchors.add(int(frame))

    active = weights > 0.0
    transitions = np.flatnonzero(np.any(active[1:] != active[:-1], axis=1)).astype(np.int64) + 1
    for frame in transitions:
        anchors.add(int(frame))
        if int(frame) > 0:
            anchors.add(int(frame) - 1)

    for frame in _changed_target_frames(targets, weights):
        anchors.add(int(frame))

    if bool(config.include_midpoint_anchors) and waypoints.size >= 2:
        for left, right in zip(waypoints[:-1], waypoints[1:]):
            mid = int((int(left) + int(right)) // 2)
            if 0 <= mid < total:
                anchors.add(mid)

    stride = max(int(config.anchor_stride), 1)
    if bool(config.solve_contact_window_only):
        stride_frames = np.flatnonzero(np.any(active, axis=1))[::stride]
    else:
        stride_frames = np.arange(0, total, stride, dtype=np.int64)
    for frame in stride_frames:
        anchors.add(int(frame))

    return np.asarray(sorted(anchors), dtype=np.int64)
