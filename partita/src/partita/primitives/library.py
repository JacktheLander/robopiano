from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from partita.primitives.features import resample_array


def build_primitive_library(
    data: dict[str, np.ndarray],
    segments: pd.DataFrame,
    assignments: pd.DataFrame,
    transformed_features: np.ndarray,
    scaler,
    pca,
    clusterer,
    feature_names: list[str],
    resample_len: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    primitives = {}
    merged = segments.merge(assignments[["global_segment_id", "primitive_id"]], on="global_segment_id", how="left")
    for primitive_id, group in merged.groupby("primitive_id"):
        pid = int(primitive_id)
        action_segments = []
        goal_segments = []
        piano_segments = []
        hand_joint_segments = []
        member_ids = []
        member_trajs = []
        for _, row in group.iterrows():
            traj_idx = int(row["trajectory_index"])
            start = int(row["start_t"])
            end = int(row["end_t"])
            member_ids.append(int(row["global_segment_id"]))
            member_trajs.append(int(row["trajectory_id"]))
            action_segments.append(resample_array(data["actions"][traj_idx, start:end], resample_len))
            if "goals" in data:
                goal_segments.append(resample_array(data["goals"][traj_idx, start:end], resample_len))
            if "piano_states" in data:
                piano_segments.append(resample_array(data["piano_states"][traj_idx, start:end], resample_len))
            if "hand_joints" in data:
                hand_joint_segments.append(resample_array(data["hand_joints"][traj_idx, start:end], resample_len))
        primitives[pid] = {
            "primitive_id": pid,
            "member_segment_ids": member_ids,
            "member_trajectory_ids": sorted(set(member_trajs)),
            "mean_duration": float(group["duration"].mean()),
            "mean_action_trajectory": np.mean(np.stack(action_segments, axis=0), axis=0).astype(np.float32),
            "mean_goal_profile": np.mean(np.stack(goal_segments, axis=0), axis=0).astype(np.float32) if goal_segments else None,
            "mean_piano_state_profile": np.mean(np.stack(piano_segments, axis=0), axis=0).astype(np.float32) if piano_segments else None,
            "mean_hand_joint_profile": np.mean(np.stack(hand_joint_segments, axis=0), axis=0).astype(np.float32) if hand_joint_segments else None,
            "feature_center": clusterer.cluster_centers_[pid].astype(np.float32),
        }
    return {
        "version": 1,
        "feature_names": feature_names,
        "scaler": scaler,
        "pca": pca,
        "clusterer": clusterer,
        "primitives": primitives,
        "resample_len": int(resample_len),
        "config": config,
    }


def save_library(path: str | Path, library: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        pickle.dump(library, f)


def load_library(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def primitive_summary(segments: pd.DataFrame, assignments: pd.DataFrame, num_training_trajectories: int) -> pd.DataFrame:
    merged = segments.merge(assignments[["global_segment_id", "primitive_id"]], on="global_segment_id", how="left")
    rows = []
    for pid, group in merged.groupby("primitive_id"):
        rows.append({
            "primitive_id": int(pid),
            "count": int(len(group)),
            "num_trajectories_used_in": int(group["trajectory_id"].nunique()),
            "trajectory_coverage_fraction": float(group["trajectory_id"].nunique() / max(num_training_trajectories, 1)),
            "mean_duration": float(group["duration"].mean()),
            "mean_action_energy": float(group["action_energy"].mean()),
            "mean_num_goal_keys": float(group["num_goal_keys"].mean()),
            "mean_num_played_keys": float(group["num_played_keys"].mean()),
        })
    return pd.DataFrame(rows).sort_values("primitive_id").reset_index(drop=True)


def primitive_usage_by_trajectory(assignments: pd.DataFrame) -> pd.DataFrame:
    table = assignments.pivot_table(index="trajectory_id", columns="primitive_id", values="global_segment_id", aggfunc="count", fill_value=0)
    table = table.sort_index(axis=0).sort_index(axis=1)
    table.columns = [f"primitive_{int(c)}" for c in table.columns]
    return table.reset_index()
