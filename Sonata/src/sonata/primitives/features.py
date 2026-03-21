from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from sonata.primitives.segmenters import load_segment_array
from sonata.primitives.slim_cache import feature_chunk_path, is_slim_chunk_name, resolve_slim_cache_paths
from sonata.utils.io import save_npz, write_json, write_table

def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def _atomic_save_npz(path: Path, **payload: Any) -> None:
    tmp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
    saved_path = save_npz(tmp_path, **payload)
    os.replace(saved_path, path)


def _load_feature_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "new", "completed_chunk_paths": [], "num_rows": 0}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload.setdefault("status", "partial")
    payload.setdefault("completed_chunk_paths", [])
    payload.setdefault("num_rows", 0)
    return payload

def extract_segment_features(segment_df: pd.DataFrame, segments_dir: Path, output_dir: Path, config: dict[str, Any]) -> dict[str, Path]:
    features_dir = output_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    table_base = features_dir / "segment_features"
    bundle_path = features_dir / "segment_features_bundle.npz"
    manifest_path = features_dir / "segment_features_manifest.json"

    partial_table_csv = features_dir / "segment_features.partial.csv"
    chunk_feature_dir = features_dir / "chunk_features"
    chunk_feature_dir.mkdir(parents=True, exist_ok=True)

    final_table_csv = table_base.with_suffix(".csv")
    force = bool(config.get("force", False))

    if force:
        for path in [partial_table_csv, final_table_csv, bundle_path, manifest_path]:
            if path.exists():
                path.unlink()

    if final_table_csv.exists() and bundle_path.exists() and not force:
        return {"feature_table_base": table_base, "feature_bundle_path": bundle_path, "manifest_path": manifest_path}

    chunk_paths = (
        segment_df["chunk_path"].astype(str).fillna("").tolist()
        if "chunk_path" in segment_df.columns
        else []
    )
    if chunk_paths and all(is_slim_chunk_name(chunk_path) for chunk_path in chunk_paths):
        feature_matrix, feature_names = load_feature_matrix_from_store(
            segment_df=segment_df,
            output_dir=output_dir,
            config=config,
        )
        if feature_names:
            feature_df = pd.concat(
                [
                    segment_df.reset_index(drop=True),
                    pd.DataFrame(feature_matrix, columns=feature_names),
                ],
                axis=1,
            )
        else:
            feature_df = segment_df.reset_index(drop=True).copy()
        write_table(feature_df, table_base)
        _atomic_save_npz(
            bundle_path,
            feature_matrix=np.asarray(feature_matrix, dtype=np.float32),
            feature_names=np.asarray(feature_names, dtype=object),
            segment_ids=feature_df["segment_id"].astype(str).to_numpy(dtype=object)
            if "segment_id" in feature_df.columns
            else np.asarray([], dtype=object),
        )
        write_json(
            {
                "status": "completed",
                "num_segments": int(feature_matrix.shape[0]),
                "num_features": int(feature_matrix.shape[1]) if feature_matrix.ndim == 2 else 0,
                "feature_prefix_counts": prefix_counts(feature_names),
                "source": "slim_store",
            },
            manifest_path,
        )
        return {
            "feature_table_base": table_base,
            "feature_bundle_path": bundle_path,
            "manifest_path": manifest_path,
        }

    manifest = _load_feature_manifest(manifest_path)
    completed_chunk_paths = set(str(item) for item in manifest.get("completed_chunk_paths", []))

    grouped = segment_df.groupby("chunk_path", sort=True)
    feature_names: list[str] = []

    for chunk_name, rows in tqdm(grouped, total=grouped.ngroups, desc="Extract segment features"):
        chunk_name = str(chunk_name)
        if chunk_name in completed_chunk_paths:
            continue

        bundle = np.load(segments_dir / chunk_name, allow_pickle=True)
        chunk_rows: list[dict[str, Any]] = []

        for row in rows.itertuples(index=False):
            vector, names = build_feature_vector(row=row, bundle=bundle, config=config)
            if not feature_names:
                feature_names = names
            merged = dict(pd.Series(row._asdict()))
            merged.update({name: float(value) for name, value in zip(feature_names, vector.tolist())})
            chunk_rows.append(merged)

        if chunk_rows:
            chunk_df = pd.DataFrame(chunk_rows)
            chunk_df.to_csv(
                partial_table_csv,
                mode="a",
                header=not partial_table_csv.exists() or partial_table_csv.stat().st_size == 0,
                index=False,
            )

            chunk_matrix = chunk_df[feature_names].to_numpy(dtype=np.float32)
            chunk_segment_ids = chunk_df["segment_id"].astype(str).to_numpy(dtype=object)
            chunk_feature_path = chunk_feature_dir / f"{Path(chunk_name).stem}.features.npz"
            _atomic_save_npz(
                chunk_feature_path,
                feature_matrix=chunk_matrix,
                feature_names=np.asarray(feature_names, dtype=object),
                segment_ids=chunk_segment_ids,
            )

        completed_chunk_paths.add(chunk_name)
        _atomic_write_json(
            manifest_path,
            {
                "status": "running",
                "completed_chunk_paths": sorted(completed_chunk_paths),
                "num_rows": int(pd.read_csv(partial_table_csv).shape[0]) if partial_table_csv.exists() else 0,
            },
        )

    feature_df = pd.read_csv(partial_table_csv) if partial_table_csv.exists() else pd.DataFrame()
    write_table(feature_df, table_base)

    if feature_df.empty:
        feature_matrix = np.zeros((0, 0), dtype=np.float32)
        feature_names = []
        segment_ids = np.asarray([], dtype=object)
    else:
        inferred_feature_names = [col for col in feature_df.columns if col not in segment_df.columns]
        feature_names = feature_names or inferred_feature_names
        feature_matrix = feature_df[feature_names].to_numpy(dtype=np.float32)
        segment_ids = feature_df["segment_id"].astype(str).to_numpy(dtype=object)

    _atomic_save_npz(
        bundle_path,
        feature_matrix=feature_matrix,
        feature_names=np.asarray(feature_names, dtype=object),
        segment_ids=segment_ids,
    )

    write_json(
        {
            "status": "completed",
            "num_segments": int(feature_matrix.shape[0]),
            "num_features": int(feature_matrix.shape[1]) if feature_matrix.ndim == 2 else 0,
            "feature_prefix_counts": prefix_counts(feature_names),
        },
        manifest_path,
    )

    return {"feature_table_base": table_base, "feature_bundle_path": bundle_path, "manifest_path": manifest_path}


def load_feature_matrix_from_store(
    segment_df: pd.DataFrame,
    output_dir: Path,
    config: dict[str, Any],
) -> tuple[np.ndarray, list[str]]:
    if segment_df.empty:
        return np.zeros((0, 0), dtype=np.float32), []
    if "chunk_path" not in segment_df.columns or "chunk_index" not in segment_df.columns:
        raise ValueError("Segment table must include chunk_path and chunk_index to load slim feature chunks.")

    paths = resolve_slim_cache_paths(output_dir, config)
    ordered = segment_df.reset_index(drop=True).copy()
    ordered["_row_position"] = np.arange(len(ordered), dtype=np.int64)
    grouped_items = list(ordered.groupby("chunk_path", sort=False))
    num_workers = _positive_int(config.get("feature_num_workers")) or 0
    if num_workers > 1 and len(grouped_items) > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(
                executor.map(
                    lambda item: _load_feature_rows_from_store_chunk(
                        paths=paths,
                        chunk_name=str(item[0]),
                        rows=item[1],
                    ),
                    grouped_items,
                )
            )
    else:
        results = [
            _load_feature_rows_from_store_chunk(paths=paths, chunk_name=str(chunk_name), rows=rows)
            for chunk_name, rows in grouped_items
        ]

    first_names = next((names for _, _, names in results if names), [])
    feature_dim = len(first_names)
    if feature_dim == 0:
        return np.zeros((len(ordered), 0), dtype=np.float32), []
    feature_matrix = np.zeros((len(ordered), feature_dim), dtype=np.float32)
    for positions, chunk_matrix, feature_names in results:
        if list(feature_names) != list(first_names):
            raise ValueError("Incompatible feature names found across slim feature chunks.")
        feature_matrix[np.asarray(positions, dtype=np.int64)] = np.asarray(chunk_matrix, dtype=np.float32)
    return feature_matrix, list(first_names)


def _load_feature_rows_from_store_chunk(
    *,
    paths,
    chunk_name: str,
    rows: pd.DataFrame,
) -> tuple[list[int], np.ndarray, list[str]]:
    if not is_slim_chunk_name(chunk_name):
        raise ValueError(f"Expected slim chunk path, found: {chunk_name}")
    chunk_path = feature_chunk_path(paths, chunk_name)
    if not chunk_path.exists():
        raise FileNotFoundError(f"Missing slim feature chunk: {chunk_path}")
    bundle = np.load(chunk_path, allow_pickle=True)
    segment_ids = np.asarray(bundle["segment_ids"], dtype=object)
    feature_matrix = np.asarray(bundle["feature_matrix"], dtype=np.float32)
    feature_names = [str(item) for item in np.asarray(bundle["feature_names"], dtype=object).tolist()]

    ordered_rows = rows.sort_values("chunk_index", kind="stable")
    positions: list[int] = []
    values: list[np.ndarray] = []
    for row_position, chunk_index, segment_id in ordered_rows[
        ["_row_position", "chunk_index", "segment_id"]
    ].itertuples(index=False, name=None):
        chunk_index = int(chunk_index)
        if chunk_index < 0 or chunk_index >= len(segment_ids):
            raise IndexError(f"Chunk index {chunk_index} is out of range for {chunk_name}")
        expected_segment_id = str(segment_id)
        actual_segment_id = str(segment_ids[chunk_index])
        if actual_segment_id != expected_segment_id:
            raise ValueError(
                f"Slim feature segment id mismatch in {chunk_name}: "
                f"expected {expected_segment_id}, found {actual_segment_id}"
            )
        positions.append(int(row_position))
        values.append(feature_matrix[chunk_index])
    if values:
        return positions, np.stack(values, axis=0).astype(np.float32), feature_names
    return positions, np.zeros((0, len(feature_names)), dtype=np.float32), feature_names


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


def build_gmr_target_from_arrays(arrays: dict[str, np.ndarray | None], config: dict[str, Any]) -> tuple[np.ndarray, str]:
    target_name = "actions" if bool(config.get("gmr_target_actions", True)) and arrays.get("actions") is not None else "hand_joints"
    trajectory = arrays.get(target_name)
    if trajectory is None:
        trajectory = arrays.get("hand_joints")
        target_name = "hand_joints"
    if trajectory is None:
        raise ValueError("Cannot build GMR target without actions or hand_joints.")
    return resample_time_axis(np.asarray(trajectory, dtype=np.float32), resolve_gmr_resample_steps(config)), target_name


def resolve_gmr_resample_steps(config: dict[str, Any]) -> int:
    return int(config.get("gmr_resample_steps", config.get("gmr_horizon", 32)))


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


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None
