from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def finite_difference(q_ref: np.ndarray, dt: float) -> np.ndarray:
    q_ref = np.asarray(q_ref, dtype=np.float32)
    if q_ref.ndim != 2:
        raise ValueError(f"q_ref must have shape [T, D], got {q_ref.shape}")
    dt = float(dt)
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if q_ref.shape[0] == 0:
        raise ValueError("q_ref must have at least one timestep")
    if q_ref.shape[0] == 1:
        return np.zeros_like(q_ref, dtype=np.float32)
    return np.gradient(q_ref, dt, axis=0, edge_order=1).astype(np.float32)


def save_qpos_trajectory(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    q_ref = np.asarray(payload["q_ref"], dtype=np.float32)
    dt = float(payload.get("dt", 0.005))
    qdot_ref = payload.get("qdot_ref")
    if qdot_ref is None:
        qdot_ref = finite_difference(q_ref, dt)
    else:
        qdot_ref = np.asarray(qdot_ref, dtype=np.float32)
    metadata = json.dumps(payload.get("metadata", {}), sort_keys=True)
    np.savez_compressed(
        path,
        q_ref=q_ref,
        qdot_ref=qdot_ref,
        dt=np.asarray(dt, dtype=np.float32),
        metadata_json=np.asarray(metadata),
    )


def load_qpos_trajectory(path: str | Path) -> dict[str, Any]:
    with np.load(Path(path), allow_pickle=False) as data:
        if "q_ref" in data:
            q_ref = np.asarray(data["q_ref"], dtype=np.float32)
            dt = float(np.asarray(data["dt"]).item()) if "dt" in data else 0.005
            qdot_ref = (
                np.asarray(data["qdot_ref"], dtype=np.float32)
                if "qdot_ref" in data
                else finite_difference(q_ref, dt)
            )
            metadata_json = str(np.asarray(data["metadata_json"]).item()) if "metadata_json" in data else "{}"
            metadata = json.loads(metadata_json)
        elif "planned_hand_joints" in data or "planned_hand_joints_dense" in data:
            dense = "planned_hand_joints_dense" in data and "planned_hand_joints" not in data
            q_key = "planned_hand_joints_dense" if dense else "planned_hand_joints"
            qdot_key = "planned_hand_velocities_dense" if dense else "planned_hand_velocities"
            q_ref = np.asarray(data[q_key], dtype=np.float32)
            dt = float(np.asarray(data["dt"]).item()) if "dt" in data else 0.005
            qdot_ref = (
                np.asarray(data[qdot_key], dtype=np.float32)
                if qdot_key in data
                else finite_difference(q_ref, dt)
            )
            metadata = {"source_format": "impromptu_planner_npz", "dt": dt}
            for key in (
                "target_keys",
                "waypoint_frames",
                "waypoint_target_keys",
                "fingertip_targets",
                "waypoint_fingertips",
                "fingertip_trajectory_targets",
                "fingertip_trajectory_weights",
                "fingertip_trajectory_dense_frames",
                "ik_anchor_frames_dense",
                "ik_anchor_frames_control",
                "segment_ids",
                "segment_ids_dense",
                "active_window_crop_start_frame",
                "active_window_crop_end_frame",
                "active_window_original_steps",
                "active_window_cropped_steps",
            ):
                if key in data:
                    metadata[key] = np.asarray(data[key])
            if "target_keys" in metadata and np.asarray(metadata["target_keys"]).shape[0] == q_ref.shape[0]:
                metadata["target_keys"] = np.asarray(metadata["target_keys"], dtype=np.float32)
        else:
            raise KeyError(
                "Trajectory NPZ must contain q_ref/qdot_ref or an Impromptu planned_hand_joints payload"
            )
    return {
        "q_ref": q_ref,
        "qdot_ref": qdot_ref,
        "dt": dt,
        "metadata": metadata,
    }
