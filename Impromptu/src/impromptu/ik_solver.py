from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from impromptu.config import ImpromptuConfig
from impromptu.paths import ensure_repo_paths

ensure_repo_paths()
from bagatelle.kinematics import HAND_STATE_DIM, IKResult  # noqa: E402


IK_ANCHOR_METRIC_COLUMNS = (
    "success",
    "optimizer_success",
    "nfev",
    "optimizer_cost",
    "mean_assigned_distance",
    "max_assigned_distance",
    "residual_norm",
    "active_finger_count",
)


@dataclass(frozen=True)
class AnchorIKSolution:
    anchor_frames: np.ndarray
    qpos: np.ndarray
    fingertips: np.ndarray
    metrics: np.ndarray
    results: tuple[IKResult, ...]


def _clip_qpos(kin: Any, qpos: np.ndarray) -> np.ndarray:
    if hasattr(kin, "clip_qpos"):
        return np.asarray(kin.clip_qpos(qpos), dtype=np.float32)
    values = np.asarray(qpos, dtype=np.float32).reshape(-1)
    lower = np.asarray(getattr(kin, "joint_lower", np.full(values.shape, -np.inf)), dtype=np.float32)
    upper = np.asarray(getattr(kin, "joint_upper", np.full(values.shape, np.inf)), dtype=np.float32)
    return np.clip(values, lower, upper).astype(np.float32)


def _result_from_pose(
    *,
    pose: np.ndarray,
    fingertips: np.ndarray,
    active_fingers: np.ndarray,
    optimizer_success: bool,
    optimizer_status: int,
    optimizer_message: str,
    optimizer_cost: float,
    nfev: int,
    threshold: float,
    target_positions: np.ndarray | None = None,
) -> IKResult:
    if active_fingers.size and target_positions is not None:
        distances = np.linalg.norm(fingertips[active_fingers.astype(np.int64)] - target_positions, axis=1).astype(np.float32)
    else:
        distances = np.zeros((0,), dtype=np.float32)
    max_residual = float(np.max(distances)) if distances.size else 0.0
    residual_norm = float(np.linalg.norm(distances)) if distances.size else 0.0
    return IKResult(
        pose=np.asarray(pose, dtype=np.float32).copy(),
        fingertip_positions=np.asarray(fingertips, dtype=np.float32).copy(),
        assigned_distances=distances,
        residual_norm=residual_norm,
        max_residual=max_residual,
        success=bool(optimizer_success and max_residual <= float(threshold)),
        optimizer_success=bool(optimizer_success),
        optimizer_status=int(optimizer_status),
        optimizer_message=str(optimizer_message),
        optimizer_cost=float(optimizer_cost),
        nfev=int(nfev),
        active_keys=np.zeros((0,), dtype=np.int32),
        assigned_keys=np.full((active_fingers.size,), -1, dtype=np.int32),
        assigned_finger_indices=active_fingers.astype(np.int32),
        unassigned_keys=np.zeros((0,), dtype=np.int32),
    )


def ik_metric_row(result: IKResult) -> np.ndarray:
    mean_distance = float(np.mean(result.assigned_distances)) if result.assigned_distances.size else 0.0
    return np.asarray(
        [
            float(result.success),
            float(result.optimizer_success),
            float(result.nfev),
            float(result.optimizer_cost),
            mean_distance,
            float(result.max_residual),
            float(result.residual_norm),
            float(result.assigned_finger_indices.size),
        ],
        dtype=np.float32,
    )


def solve_fingertip_frame(
    *,
    kin: Any,
    fingertip_targets: np.ndarray,
    fingertip_weights: np.ndarray,
    previous_qpos: np.ndarray,
    neutral_qpos: np.ndarray,
    config: ImpromptuConfig,
) -> IKResult:
    targets = np.asarray(fingertip_targets, dtype=np.float32)
    weights = np.asarray(fingertip_weights, dtype=np.float32).reshape(-1)
    if targets.shape != (10, 3):
        raise ValueError(f"fingertip_targets must have shape [10, 3], got {targets.shape}")
    if weights.shape != (10,):
        raise ValueError(f"fingertip_weights must have shape [10], got {weights.shape}")
    previous = _clip_qpos(kin, previous_qpos)
    neutral = _clip_qpos(kin, neutral_qpos)
    mask = np.isfinite(targets).all(axis=1) & (weights > 0.0)
    finger_indices = np.flatnonzero(mask).astype(np.int32)
    if finger_indices.size == 0:
        fingertips = np.asarray(kin.fingertip_positions_for_qpos(previous), dtype=np.float32)
        return _result_from_pose(
            pose=previous,
            fingertips=fingertips,
            active_fingers=finger_indices,
            optimizer_success=True,
            optimizer_status=0,
            optimizer_message="no active fingertip targets",
            optimizer_cost=0.0,
            nfev=0,
            threshold=float(config.residual_success_threshold),
        )

    target_positions = targets[finger_indices.astype(np.int64)].astype(np.float32)
    active_weights = weights[finger_indices.astype(np.int64)].astype(np.float64)

    def residual(values: np.ndarray) -> np.ndarray:
        q = _clip_qpos(kin, values)
        fingertips = np.asarray(kin.fingertip_positions_for_qpos(q), dtype=np.float32)
        fingertip_error = (fingertips[finger_indices.astype(np.int64)] - target_positions) * active_weights[:, None]
        parts = [
            fingertip_error.reshape(-1) * float(config.ik_fingertip_weight),
            (q - previous).reshape(-1) * float(config.ik_smoothness_weight),
            (q - neutral).reshape(-1) * float(config.ik_neutral_weight),
        ]
        return np.concatenate(parts, axis=0).astype(np.float64)

    try:
        opt = least_squares(
            residual,
            previous.astype(np.float64),
            bounds=(np.asarray(kin.joint_lower, dtype=np.float64), np.asarray(kin.joint_upper, dtype=np.float64)),
            max_nfev=max(int(config.ik_max_nfev), 1),
            ftol=float(config.ik_ftol),
            xtol=float(config.ik_xtol),
            gtol=float(config.ik_gtol),
        )
        pose = _clip_qpos(kin, opt.x)
        fingertips = np.asarray(kin.fingertip_positions_for_qpos(pose), dtype=np.float32)
        return _result_from_pose(
            pose=pose,
            fingertips=fingertips,
            active_fingers=finger_indices,
            optimizer_success=bool(opt.success),
            optimizer_status=int(opt.status),
            optimizer_message=str(opt.message),
            optimizer_cost=float(opt.cost),
            nfev=int(opt.nfev),
            threshold=float(config.residual_success_threshold),
            target_positions=target_positions,
        )
    except Exception as exc:
        fingertips = np.asarray(kin.fingertip_positions_for_qpos(previous), dtype=np.float32)
        return _result_from_pose(
            pose=previous,
            fingertips=fingertips,
            active_fingers=finger_indices,
            optimizer_success=False,
            optimizer_status=-1,
            optimizer_message=f"IK exception: {type(exc).__name__}: {exc}",
            optimizer_cost=float("nan"),
            nfev=0,
            threshold=float(config.residual_success_threshold),
            target_positions=target_positions,
        )


def solve_fingertip_trajectory_anchors(
    *,
    kin: Any,
    fingertip_targets: np.ndarray,
    fingertip_weights: np.ndarray,
    anchor_frames: np.ndarray,
    initial_qpos: np.ndarray,
    neutral_qpos: np.ndarray,
    config: ImpromptuConfig,
) -> AnchorIKSolution:
    frames = np.asarray(anchor_frames, dtype=np.int64).reshape(-1)
    targets = np.asarray(fingertip_targets, dtype=np.float32)
    weights = np.asarray(fingertip_weights, dtype=np.float32)
    previous = _clip_qpos(kin, initial_qpos)
    qpos_rows: list[np.ndarray] = []
    fingertip_rows: list[np.ndarray] = []
    metric_rows: list[np.ndarray] = []
    results: list[IKResult] = []
    for frame in frames:
        index = int(frame)
        result = solve_fingertip_frame(
            kin=kin,
            fingertip_targets=targets[index],
            fingertip_weights=weights[index],
            previous_qpos=previous,
            neutral_qpos=neutral_qpos,
            config=config,
        )
        qpos_rows.append(result.pose.astype(np.float32))
        fingertip_rows.append(result.fingertip_positions.astype(np.float32))
        metric_rows.append(ik_metric_row(result))
        results.append(result)
        previous = result.pose.astype(np.float32)
    if qpos_rows:
        qpos = np.stack(qpos_rows, axis=0).astype(np.float32)
        fingertips = np.stack(fingertip_rows, axis=0).astype(np.float32)
        metrics = np.stack(metric_rows, axis=0).astype(np.float32)
    else:
        qpos = np.zeros((0, HAND_STATE_DIM), dtype=np.float32)
        fingertips = np.zeros((0, 10, 3), dtype=np.float32)
        metrics = np.zeros((0, len(IK_ANCHOR_METRIC_COLUMNS)), dtype=np.float32)
    return AnchorIKSolution(anchor_frames=frames, qpos=qpos, fingertips=fingertips, metrics=metrics, results=tuple(results))


def interpolate_anchor_qpos(
    *,
    anchor_frames: np.ndarray,
    anchor_qpos: np.ndarray,
    dense_total: int,
) -> tuple[np.ndarray, np.ndarray]:
    frames = np.asarray(anchor_frames, dtype=np.int64).reshape(-1)
    qpos = np.asarray(anchor_qpos, dtype=np.float32)
    if qpos.ndim != 2:
        raise ValueError(f"anchor_qpos must have shape [N, D], got {qpos.shape}")
    if frames.shape[0] != qpos.shape[0]:
        raise ValueError("anchor_frames and anchor_qpos must have matching leading dimension")
    total = max(int(dense_total), 0)
    dim = int(qpos.shape[1]) if qpos.ndim == 2 else HAND_STATE_DIM
    dense = np.zeros((total, dim), dtype=np.float32)
    segment_ids = np.full((total,), -1, dtype=np.int32)
    if total == 0 or frames.size == 0:
        return dense, segment_ids
    order = np.argsort(frames, kind="stable")
    frames = frames[order]
    qpos = qpos[order]
    first = int(np.clip(frames[0], 0, total - 1))
    dense[: first + 1] = qpos[0]
    segment_ids[: first + 1] = 0
    for index in range(frames.size - 1):
        start = int(np.clip(frames[index], 0, total - 1))
        end = int(np.clip(frames[index + 1], 0, total - 1))
        if end < start:
            continue
        span = max(end - start, 1)
        for frame in range(start, end + 1):
            u = 0.0 if end == start else (frame - start) / span
            alpha = u * u * (3.0 - 2.0 * u)
            dense[frame] = (qpos[index] + alpha * (qpos[index + 1] - qpos[index])).astype(np.float32)
            segment_ids[frame] = int(index)
    last = int(np.clip(frames[-1], 0, total - 1))
    dense[last:] = qpos[-1]
    segment_ids[last:] = int(max(frames.size - 1, 0))
    return dense, segment_ids
