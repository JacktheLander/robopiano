from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from sonata.primitives.segmenters import load_segment_array
from sonata.primitives.slim_cache import feature_chunk_path, is_slim_chunk_name, resolve_slim_cache_paths
from sonata.utils.io import read_json, save_npz, write_json, write_table


def extract_segment_features(segment_df: pd.DataFrame, segments_dir: Path, output_dir: Path, config: dict[str, Any]) -> dict[str, Path]:
    features_dir = output_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    table_base = features_dir / "segment_features"
    bundle_path = features_dir / "segment_features_bundle.npz"
    manifest_path = features_dir / "segment_features_manifest.json"
    if table_base.with_suffix(".csv").exists() and bundle_path.exists() and manifest_path.exists() and not bool(config.get("force", False)):
        manifest = read_json(manifest_path)
        if int(manifest.get("num_segments", -1)) == int(len(segment_df)):
            return {"feature_table_base": table_base, "feature_bundle_path": bundle_path, "manifest_path": manifest_path}

    slim_paths = resolve_slim_cache_paths(output_dir, config)
    indexed_df = segment_df.reset_index(drop=True).copy()
    indexed_df["__feature_row_index"] = np.arange(len(indexed_df), dtype=np.int64)
    feature_rows: list[np.ndarray | None] = [None] * len(indexed_df)
    feature_names: list[str] = []
    grouped = indexed_df.groupby("chunk_path", sort=True)
    for chunk_name, rows in tqdm(grouped, total=grouped.ngroups, desc="Extract segment features"):
        ordered_rows = rows.sort_values("chunk_index", kind="stable")
        if is_slim_chunk_name(chunk_name) and feature_chunk_path(slim_paths, chunk_name).exists():
            bundle = np.load(feature_chunk_path(slim_paths, chunk_name), allow_pickle=True)
            chunk_names = [str(item) for item in bundle["feature_names"].tolist()]
            if not feature_names:
                feature_names = chunk_names
            elif feature_names != chunk_names:
                raise ValueError(f"Incompatible slim feature names in chunk {chunk_name}")
            chunk_ids = np.asarray(bundle["segment_ids"], dtype=object)
            chunk_matrix = np.asarray(bundle["feature_matrix"], dtype=np.float32)
            for row in ordered_rows.itertuples(index=False):
                idx = int(row.chunk_index)
                if idx >= len(chunk_ids) or str(chunk_ids[idx]) != str(row.segment_id):
                    raise ValueError(f"Slim feature segment id mismatch in chunk {chunk_name}")
                feature_rows[int(row.__feature_row_index)] = chunk_matrix[idx]
            continue
        bundle = np.load(segments_dir / str(chunk_name), allow_pickle=True)
        for row in ordered_rows.itertuples(index=False):
            vector, names = build_feature_vector(row=row, bundle=bundle, config=config)
            if not feature_names:
                feature_names = names
            feature_rows[int(row.__feature_row_index)] = vector
    if not feature_rows:
        feature_matrix = np.zeros((0, 0), dtype=np.float32)
    else:
        if any(vector is None for vector in feature_rows):
            raise ValueError("Missing feature vectors while materializing feature bundle.")
        feature_matrix = np.stack([np.asarray(vector, dtype=np.float32) for vector in feature_rows], axis=0).astype(np.float32)
    feature_df = pd.concat([segment_df.reset_index(drop=True), pd.DataFrame(feature_matrix, columns=feature_names)], axis=1)
    write_table(feature_df, table_base)
    save_npz(bundle_path, feature_matrix=feature_matrix, feature_names=np.asarray(feature_names, dtype=object), segment_ids=segment_df["segment_id"].astype(str).to_numpy(dtype=object))
    write_json(
        {
            "num_segments": int(len(segment_df)),
            "num_features": int(feature_matrix.shape[1]),
            "trajectory_resample_steps": int(config["trajectory_resample_steps"]),
            "feature_prefix_counts": prefix_counts(feature_names),
        },
        manifest_path,
    )
    return {"feature_table_base": table_base, "feature_bundle_path": bundle_path, "manifest_path": manifest_path}


def build_feature_vector(row, bundle: Any, config: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    idx = int(row.chunk_index)
    arrays = {
        "hand_joints": load_segment_array(bundle, "hand_joints", idx),
        "joint_velocities": load_segment_array(bundle, "joint_velocities", idx),
        "actions": load_segment_array(bundle, "actions", idx),
        "goals": load_segment_array(bundle, "goals", idx),
        "piano_states": load_segment_array(bundle, "piano_states", idx),
    }
    return build_feature_vector_from_arrays(row=row, arrays=arrays, config=config)


def build_feature_vector_from_arrays(row: Any, arrays: dict[str, np.ndarray | None], config: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    hand_joints = arrays.get("hand_joints")
    velocity = arrays.get("joint_velocities")
    actions = arrays.get("actions")
    goals = arrays.get("goals")
    piano_states = arrays.get("piano_states")
    if hand_joints is None:
        raise ValueError(f"Missing hand_joints for segment {_row_value(row, 'segment_id')}")
    if velocity is None:
        velocity = np.gradient(hand_joints, axis=0).astype(np.float32)
    acceleration = np.gradient(velocity, axis=0).astype(np.float32)
    score_context = json.loads(str(_row_value(row, "score_context_json")))
    contact_roll = goals if goals is not None else piano_states
    contact_roll = np.asarray(contact_roll[:, :-1] > 0.5, dtype=np.float32) if contact_roll is not None else np.zeros((hand_joints.shape[0], 88), dtype=np.float32)

    pieces: list[np.ndarray] = []
    names: list[str] = []

    joint_mean = hand_joints.mean(axis=0)
    joint_std = hand_joints.std(axis=0)
    joint_delta = hand_joints[-1] - hand_joints[0]
    vel_mean = velocity.mean(axis=0)
    vel_std = velocity.std(axis=0)
    accel_mean = acceleration.mean(axis=0)
    pieces.extend([joint_mean, joint_std, joint_delta, vel_mean, vel_std, accel_mean])
    names.extend([f"joint_mean_{idx:03d}" for idx in range(joint_mean.size)])
    names.extend([f"joint_std_{idx:03d}" for idx in range(joint_std.size)])
    names.extend([f"joint_delta_{idx:03d}" for idx in range(joint_delta.size)])
    names.extend([f"velocity_mean_{idx:03d}" for idx in range(vel_mean.size)])
    names.extend([f"velocity_std_{idx:03d}" for idx in range(vel_std.size)])
    names.extend([f"accel_mean_{idx:03d}" for idx in range(accel_mean.size)])

    if actions is not None:
        action_mean = actions.mean(axis=0)
        action_std = actions.std(axis=0)
        action_delta = actions[-1] - actions[0]
    else:
        action_dim = int(config.get("fallback_action_dim", 39))
        action_mean = np.zeros((action_dim,), dtype=np.float32)
        action_std = np.zeros((action_dim,), dtype=np.float32)
        action_delta = np.zeros((action_dim,), dtype=np.float32)
    pieces.extend([action_mean, action_std, action_delta])
    names.extend([f"action_mean_{idx:03d}" for idx in range(action_mean.size)])
    names.extend([f"action_std_{idx:03d}" for idx in range(action_std.size)])
    names.extend([f"action_delta_{idx:03d}" for idx in range(action_delta.size)])

    resampled_joints = resample_time_axis(hand_joints, int(config["trajectory_resample_steps"])).reshape(-1)
    pieces.append(resampled_joints)
    names.extend([f"traj_joint_{idx:04d}" for idx in range(resampled_joints.size)])

    if bool(config.get("include_action_trajectory", True)) and actions is not None:
        resampled_actions = resample_time_axis(actions, int(config["trajectory_resample_steps"])).reshape(-1)
        pieces.append(resampled_actions)
        names.extend([f"traj_action_{idx:04d}" for idx in range(resampled_actions.size)])

    histogram = np.asarray(score_context.get("goal_histogram", [0.0] * 12), dtype=np.float32)
    scalar_context = np.asarray(
        [
            float(score_context.get("active_ratio", 0.0)),
            float(score_context.get("future_density", 0.0)),
            float(_row_value(row, "duration_steps")),
            float(_row_value(row, "motion_energy")),
            float(_row_value(row, "chord_size")),
            float(_row_value(row, "key_center")),
            float(_row_value(row, "start_state_norm")),
            float(_row_value(row, "end_state_norm")),
        ],
        dtype=np.float32,
    )
    contact_summary = np.asarray(
        [
            float(contact_roll.mean()),
            float(contact_roll.sum(axis=1).mean() / max(contact_roll.shape[1], 1)),
            float((contact_roll.sum(axis=1) > 0).mean()),
        ],
        dtype=np.float32,
    )
    pieces.extend([histogram, scalar_context, contact_summary])
    names.extend([f"score_hist_{idx:02d}" for idx in range(histogram.size)])
    names.extend(
        [
            "score_active_ratio",
            "score_future_density",
            "duration_steps",
            "motion_energy",
            "chord_size",
            "key_center",
            "start_state_norm",
            "end_state_norm",
        ]
    )
    names.extend(["contact_mean", "contact_density", "contact_nonzero_ratio"])
    return np.concatenate(pieces).astype(np.float32), names


def _row_value(row: Any, name: str) -> Any:
    if isinstance(row, dict):
        return row[name]
    return getattr(row, name)


def resample_time_axis(array: np.ndarray, steps: int) -> np.ndarray:
    if array.shape[0] == steps:
        return array.astype(np.float32)
    x_old = np.linspace(0.0, 1.0, array.shape[0], dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, steps, dtype=np.float32)
    output = np.zeros((steps, array.shape[1]), dtype=np.float32)
    for dim in range(array.shape[1]):
        output[:, dim] = np.interp(x_new, x_old, array[:, dim])
    return output


def prefix_counts(names: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name in names:
        prefix = name.split("_", 1)[0]
        counts[prefix] = counts.get(prefix, 0) + 1
    return counts
