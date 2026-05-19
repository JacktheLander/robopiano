from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _summary(values: np.ndarray, prefix: str) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float32).reshape(-1)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_median": 0.0,
            f"{prefix}_p95": 0.0,
            f"{prefix}_max": 0.0,
        }
    return {
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_median": float(np.median(finite)),
        f"{prefix}_p95": float(np.percentile(finite, 95)),
        f"{prefix}_max": float(np.max(finite)),
    }


def evaluate_trajectory_payload(payload: dict[str, np.ndarray]) -> dict[str, Any]:
    anchor_frames = np.asarray(payload.get("ik_anchor_frames_dense", np.zeros((0,), dtype=np.int64)), dtype=np.int64)
    anchor_tips = np.asarray(payload.get("ik_anchor_fingertips", np.zeros((0, 10, 3), dtype=np.float32)), dtype=np.float32)
    targets = np.asarray(payload.get("fingertip_trajectory_targets", np.zeros((0, 10, 3), dtype=np.float32)), dtype=np.float32)
    weights = np.asarray(payload.get("fingertip_trajectory_weights", np.zeros((0, 10), dtype=np.float32)), dtype=np.float32)
    distances: list[np.ndarray] = []
    for row, frame in enumerate(anchor_frames):
        index = int(frame)
        if row >= anchor_tips.shape[0] or index < 0 or index >= targets.shape[0]:
            continue
        mask = np.isfinite(targets[index]).all(axis=1) & (weights[index] > 0.0)
        if bool(mask.any()):
            distances.append(np.linalg.norm(anchor_tips[row, mask] - targets[index, mask], axis=1).astype(np.float32))
    distance_values = np.concatenate(distances, axis=0) if distances else np.zeros((0,), dtype=np.float32)

    metrics = np.asarray(payload.get("ik_anchor_metrics", np.zeros((0, 8), dtype=np.float32)), dtype=np.float32)
    qpos = np.asarray(payload.get("planned_hand_joints", np.zeros((0, 46), dtype=np.float32)), dtype=np.float32)
    velocities = np.asarray(payload.get("planned_hand_velocities", np.zeros_like(qpos)), dtype=np.float32)
    joint_steps = np.linalg.norm(np.diff(qpos, axis=0), axis=1) if qpos.shape[0] > 1 else np.zeros((0,), dtype=np.float32)
    joint_velocity_norm = np.linalg.norm(velocities, axis=1) if velocities.ndim == 2 else np.zeros((0,), dtype=np.float32)

    out: dict[str, Any] = {}
    out.update(_summary(distance_values, "ik_anchor_fingertip_distance"))
    out["ik_anchor_success_rate"] = float(np.mean(metrics[:, 0])) if metrics.size else 0.0
    out["ik_anchor_nfev_mean"] = float(np.mean(metrics[:, 2])) if metrics.size else 0.0
    out["ik_anchor_nfev_p95"] = float(np.percentile(metrics[:, 2], 95)) if metrics.size else 0.0
    out["max_joint_step"] = float(np.max(joint_steps)) if joint_steps.size else 0.0
    out["mean_joint_step"] = float(np.mean(joint_steps)) if joint_steps.size else 0.0
    out["max_joint_velocity"] = float(np.max(joint_velocity_norm)) if joint_velocity_norm.size else 0.0
    out["mean_joint_velocity"] = float(np.mean(joint_velocity_norm)) if joint_velocity_norm.size else 0.0

    # Placeholders preserve schema compatibility for future rollout-integrated evaluation.
    out.setdefault("missed_key_presses", None)
    out.setdefault("mispresses", None)
    out.setdefault("matched_press_events", None)
    out.setdefault("target_press_events", None)
    out.setdefault("timing_abs_error_mean_s", None)
    out.setdefault("timing_abs_error_p95_s", None)
    return out


def evaluate_trajectory_npz(path: str | Path) -> dict[str, Any]:
    npz_path = Path(path).expanduser().resolve()
    data = np.load(npz_path, allow_pickle=False)
    return evaluate_trajectory_payload({name: data[name] for name in data.files})
