from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from sonata.primitives.parallel import iter_parallel_map, resolve_worker_count
from sonata.primitives.segmenters import load_segment_array
from sonata.primitives.slim_cache import feature_chunk_path, is_slim_chunk_name, resolve_slim_cache_paths
from sonata.utils.io import save_npz, write_json, write_table

FINGER_LABELS = (
    "right_thumb",
    "right_index",
    "right_middle",
    "right_ring",
    "right_pinky",
    "left_thumb",
    "left_index",
    "left_middle",
    "left_ring",
    "left_pinky",
)
MAX_CONTEXT_KEYS = 8
KEY_WIDTH_METERS = 0.0235


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

    chunk_paths = segment_df["chunk_path"].astype(str).fillna("").tolist() if "chunk_path" in segment_df.columns else []
    if chunk_paths and all(is_slim_chunk_name(chunk_path) for chunk_path in chunk_paths):
        feature_matrix, feature_names = load_feature_matrix_from_store(segment_df=segment_df, output_dir=output_dir, config=config)
        feature_df = (
            pd.concat([segment_df.reset_index(drop=True), pd.DataFrame(feature_matrix, columns=feature_names)], axis=1)
            if feature_names
            else segment_df.reset_index(drop=True).copy()
        )
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
        return {"feature_table_base": table_base, "feature_bundle_path": bundle_path, "manifest_path": manifest_path}

    manifest = _load_feature_manifest(manifest_path)
    completed_chunk_paths = set(str(item) for item in manifest.get("completed_chunk_paths", []))
    grouped_items = [(str(chunk_name), rows.copy()) for chunk_name, rows in segment_df.groupby("chunk_path", sort=True)]
    grouped_items = [(chunk_name, rows) for chunk_name, rows in grouped_items if chunk_name not in completed_chunk_paths]
    feature_names: list[str] = []

    jobs = [
        {
            "chunk_name": chunk_name,
            "rows": rows.to_dict(orient="records"),
            "segments_dir": str(segments_dir),
            "config": config,
        }
        for chunk_name, rows in grouped_items
    ]
    num_workers = resolve_worker_count(config.get("feature_num_workers"), default=0)
    use_process_pool = bool(config.get("use_process_pool", True))
    start_method = str(config.get("process_start_method", "spawn"))
    iterator = (
        iter_parallel_map(
            _build_feature_chunk_job,
            jobs,
            max_workers=num_workers,
            use_process_pool=use_process_pool,
            in_flight_multiplier=2,
            start_method=start_method,
        )
        if num_workers > 1 and len(jobs) > 1
        else (_build_feature_chunk_job(job) for job in jobs)
    )

    for result in tqdm(iterator, total=len(jobs), desc="Extract segment features"):
        chunk_name = str(result["chunk_name"])
        chunk_rows = result["rows"]
        if chunk_rows:
            chunk_df = pd.DataFrame(chunk_rows)
            inferred_names = [column for column in chunk_df.columns if column not in segment_df.columns]
            if not feature_names:
                feature_names = inferred_names
            chunk_df.to_csv(
                partial_table_csv,
                mode="a",
                header=not partial_table_csv.exists() or partial_table_csv.stat().st_size == 0,
                index=False,
            )
            chunk_feature_path = chunk_feature_dir / f"{Path(chunk_name).stem}.features.npz"
            _atomic_save_npz(
                chunk_feature_path,
                feature_matrix=np.asarray(result["feature_matrix"], dtype=np.float32),
                feature_names=np.asarray(feature_names, dtype=object),
                segment_ids=np.asarray(result["segment_ids"], dtype=object),
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
        inferred_feature_names = [column for column in feature_df.columns if column not in segment_df.columns]
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


def _build_feature_chunk_job(job_payload: dict[str, Any]) -> dict[str, Any]:
    chunk_name = str(job_payload["chunk_name"])
    bundle = np.load(Path(job_payload["segments_dir"]) / chunk_name, allow_pickle=True)
    chunk_rows: list[dict[str, Any]] = []
    feature_names: list[str] = []
    feature_vectors: list[np.ndarray] = []
    segment_ids: list[str] = []
    for row in job_payload["rows"]:
        vector, names = build_feature_vector(row=row, bundle=bundle, config=job_payload["config"])
        feature_names = feature_names or list(names)
        merged = dict(row)
        merged.update({name: float(value) for name, value in zip(feature_names, vector.tolist())})
        chunk_rows.append(merged)
        feature_vectors.append(np.asarray(vector, dtype=np.float32))
        segment_ids.append(str(row["segment_id"]))
    feature_matrix = np.stack(feature_vectors, axis=0).astype(np.float32) if feature_vectors else np.zeros((0, 0), dtype=np.float32)
    return {
        "chunk_name": chunk_name,
        "rows": chunk_rows,
        "feature_matrix": feature_matrix,
        "feature_names": feature_names,
        "segment_ids": segment_ids,
    }


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
    grouped_items = [(str(chunk_name), rows.copy()) for chunk_name, rows in ordered.groupby("chunk_path", sort=False)]
    jobs = [
        {
            "chunk_name": chunk_name,
            "rows": rows[["_row_position", "chunk_index", "segment_id"]].to_dict(orient="records"),
            "paths_root": str(paths.root),
            "config": config,
        }
        for chunk_name, rows in grouped_items
    ]
    num_workers = resolve_worker_count(config.get("feature_num_workers"), default=0)
    iterator = (
        iter_parallel_map(
            _load_feature_rows_from_store_chunk_job,
            jobs,
            max_workers=num_workers,
            use_process_pool=False,
            in_flight_multiplier=2,
            start_method=str(config.get("process_start_method", "spawn")),
        )
        if num_workers > 1 and len(jobs) > 1
        else (_load_feature_rows_from_store_chunk_job(job) for job in jobs)
    )
    results = list(iterator)

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


def _load_feature_rows_from_store_chunk_job(job_payload: dict[str, Any]) -> tuple[list[int], np.ndarray, list[str]]:
    paths = resolve_slim_cache_paths(Path(job_payload["paths_root"]).parent, job_payload["config"])
    chunk_name = str(job_payload["chunk_name"])
    if not is_slim_chunk_name(chunk_name):
        raise ValueError(f"Expected slim chunk path, found: {chunk_name}")
    chunk_path = feature_chunk_path(paths, chunk_name)
    if not chunk_path.exists():
        raise FileNotFoundError(f"Missing slim feature chunk: {chunk_path}")
    bundle = np.load(chunk_path, allow_pickle=True)
    segment_ids = np.asarray(bundle["segment_ids"], dtype=object)
    feature_matrix = np.asarray(bundle["feature_matrix"], dtype=np.float32)
    feature_names = [str(item) for item in np.asarray(bundle["feature_names"], dtype=object).tolist()]
    ordered_rows = sorted(job_payload["rows"], key=lambda item: int(item["chunk_index"]))

    positions: list[int] = []
    values: list[np.ndarray] = []
    for row in ordered_rows:
        row_position = int(row["_row_position"])
        chunk_index = int(row["chunk_index"])
        segment_id = str(row["segment_id"])
        if chunk_index < 0 or chunk_index >= len(segment_ids):
            raise IndexError(f"Chunk index {chunk_index} is out of range for {chunk_name}")
        if str(segment_ids[chunk_index]) != segment_id:
            raise ValueError(
                f"Slim feature segment id mismatch in {chunk_name}: expected {segment_id}, found {segment_ids[chunk_index]}"
            )
        positions.append(row_position)
        values.append(feature_matrix[chunk_index])
    if values:
        return positions, np.stack(values, axis=0).astype(np.float32), feature_names
    return positions, np.zeros((0, len(feature_names)), dtype=np.float32), feature_names


def build_feature_vector(row, bundle: Any, config: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    idx = int(row["chunk_index"] if isinstance(row, dict) else row.chunk_index)
    arrays = {
        "hand_joints": load_segment_array(bundle, "hand_joints", idx),
        "joint_velocities": load_segment_array(bundle, "joint_velocities", idx),
        "actions": load_segment_array(bundle, "actions", idx),
        "goals": load_segment_array(bundle, "goals", idx),
        "piano_states": load_segment_array(bundle, "piano_states", idx),
        "hand_fingertips": load_segment_array(bundle, "hand_fingertips", idx),
        "wrist_pose": load_segment_array(bundle, "wrist_pose", idx),
        "hand_pose": load_segment_array(bundle, "hand_pose", idx),
    }
    return build_feature_vector_from_arrays(row=row, arrays=arrays, config=config)


def build_feature_vector_from_arrays(row: Any, arrays: dict[str, np.ndarray | None], config: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    hand_joints = arrays.get("hand_joints")
    velocity = arrays.get("joint_velocities")
    actions = arrays.get("actions")
    goals = arrays.get("goals")
    piano_states = arrays.get("piano_states")
    fingertips = arrays.get("hand_fingertips")
    wrist_pose = arrays.get("wrist_pose")
    hand_pose = arrays.get("hand_pose")
    if hand_joints is None:
        raise ValueError(f"Missing hand_joints for segment {_row_value(row, 'segment_id')}")

    hand_joints = np.asarray(hand_joints, dtype=np.float32)
    velocity = np.asarray(velocity if velocity is not None else np.gradient(hand_joints, axis=0), dtype=np.float32)
    acceleration = np.gradient(velocity, axis=0).astype(np.float32)
    normalized_joints, frame_metadata = normalize_hand_frame(hand_joints=hand_joints, wrist_pose=wrist_pose, row=row, config=config)
    normalized_velocity = np.gradient(normalized_joints, axis=0).astype(np.float32)
    normalized_acceleration = np.gradient(normalized_velocity, axis=0).astype(np.float32)
    action_delta = np.diff(actions, axis=0, prepend=actions[:1]).astype(np.float32) if actions is not None else None
    contact_roll = goals if goals is not None else piano_states
    contact_roll = np.asarray(contact_roll, dtype=np.float32) if contact_roll is not None else np.zeros((hand_joints.shape[0], 88), dtype=np.float32)
    contact_keys = contact_roll[:, :88] if contact_roll.shape[1] >= 88 else np.zeros((hand_joints.shape[0], 88), dtype=np.float32)
    pedal_state = contact_roll[:, 88:] if contact_roll.shape[1] > 88 else np.zeros((hand_joints.shape[0], 1), dtype=np.float32)

    score_context = _safe_json(_row_value(row, "score_context_json", default="{}"))
    primitive_context = build_primitive_context_features(row=row, arrays=arrays, config=config)
    pieces: list[np.ndarray] = []
    names: list[str] = []

    _append_feature_block(
        pieces,
        names,
        "joint_rel_mean",
        normalized_joints.mean(axis=0),
    )
    _append_feature_block(pieces, names, "joint_rel_std", normalized_joints.std(axis=0))
    _append_feature_block(pieces, names, "joint_rel_delta", normalized_joints[-1] - normalized_joints[0])
    _append_feature_block(pieces, names, "joint_rel_velocity_mean", normalized_velocity.mean(axis=0))
    _append_feature_block(pieces, names, "joint_rel_velocity_std", normalized_velocity.std(axis=0))
    _append_feature_block(pieces, names, "joint_rel_accel_mean", normalized_acceleration.mean(axis=0))
    _append_feature_block(pieces, names, "joint_abs_speed_summary", np.asarray(_trajectory_summary(velocity), dtype=np.float32))
    _append_feature_block(pieces, names, "joint_abs_accel_summary", np.asarray(_trajectory_summary(acceleration), dtype=np.float32))

    if actions is not None:
        _append_feature_block(pieces, names, "action_mean", actions.mean(axis=0))
        _append_feature_block(pieces, names, "action_std", actions.std(axis=0))
        _append_feature_block(pieces, names, "action_delta", actions[-1] - actions[0])
        if action_delta is not None:
            _append_feature_block(pieces, names, "action_delta_summary", np.asarray(_trajectory_summary(action_delta), dtype=np.float32))
    else:
        action_dim = int(config.get("fallback_action_dim", 39))
        zeros = np.zeros((action_dim,), dtype=np.float32)
        _append_feature_block(pieces, names, "action_mean", zeros)
        _append_feature_block(pieces, names, "action_std", zeros)
        _append_feature_block(pieces, names, "action_delta", zeros)
        _append_feature_block(pieces, names, "action_delta_summary", np.zeros((4,), dtype=np.float32))

    if fingertips is not None:
        fingertip_speed = np.gradient(np.asarray(fingertips, dtype=np.float32), axis=0)
        _append_feature_block(pieces, names, "fingertip_motion", np.asarray(_trajectory_summary(fingertip_speed), dtype=np.float32))
    if wrist_pose is not None:
        wrist_pose = np.asarray(wrist_pose, dtype=np.float32)
        _append_feature_block(pieces, names, "wrist_pose_delta", wrist_pose[-1] - wrist_pose[0])
        _append_feature_block(pieces, names, "wrist_pose_summary", np.asarray(_trajectory_summary(np.gradient(wrist_pose, axis=0)), dtype=np.float32))
    if hand_pose is not None:
        hand_pose = np.asarray(hand_pose, dtype=np.float32)
        _append_feature_block(pieces, names, "hand_pose_delta", hand_pose[-1] - hand_pose[0])

    normalized_traj = resample_time_axis(normalized_joints, int(config["trajectory_resample_steps"])).reshape(-1)
    _append_feature_block(pieces, names, "traj_joint_rel", normalized_traj)
    velocity_traj = resample_time_axis(normalized_velocity, int(config["trajectory_resample_steps"])).reshape(-1)
    _append_feature_block(pieces, names, "traj_joint_velocity_rel", velocity_traj)
    if bool(config.get("include_action_trajectory", True)) and action_delta is not None:
        action_traj = resample_time_axis(action_delta, int(config["trajectory_resample_steps"])).reshape(-1)
        _append_feature_block(pieces, names, "traj_action_delta", action_traj)

    histogram = np.asarray(score_context.get("goal_histogram", [0.0] * 12), dtype=np.float32)
    _append_feature_block(pieces, names, "score_hist", histogram)
    target_keys = _parse_key_signature(
        str(_row_value(row, "target_key_signature", default="") or _row_value(row, "key_signature", default=""))
    )
    _append_feature_block(pieces, names, "target_keyset", _target_keyset_vector(target_keys))
    _append_feature_block(pieces, names, "target_keyset_summary", _target_keyset_summary(target_keys))
    _append_feature_block(pieces, names, "target_interval_hist", _target_interval_histogram(target_keys))
    for block_name, values in primitive_context["feature_blocks"].items():
        _append_feature_block(pieces, names, block_name, values)
    scalar_context = np.asarray(
        [
            float(score_context.get("active_ratio", 0.0)),
            float(score_context.get("future_density", 0.0)),
            float(_row_value(row, "duration_steps", default=0.0)),
            float(_row_value(row, "motion_energy", default=0.0)),
            float(_row_value(row, "chord_size", default=0.0)),
            float(_row_value(row, "key_center", default=0.0)),
            float(_row_value(row, "start_state_norm", default=0.0)),
            float(_row_value(row, "end_state_norm", default=0.0)),
            float(_row_value(row, "boundary_energy", default=0.0)),
            float(_row_value(row, "boundary_alignment_score", default=0.0)),
            float(_row_value(row, "proposal_size", default=1.0)),
            float(_row_value(row, "proposal_span_steps", default=_row_value(row, "duration_steps", default=0.0))),
            float(_row_value(row, "target_key_count", default=len(target_keys))),
            float(_row_value(row, "next_onset_gap_steps", default=-1)),
            float(bool(_row_value(row, "truncated_by_next_onset", default=False))),
            float(frame_metadata["used_wrist_frame"]),
            float(frame_metadata["key_center_offset"]),
        ],
        dtype=np.float32,
    )
    _append_feature_block(pieces, names, "score_scalar", scalar_context)

    contact_summary = np.asarray(
        [
            float(contact_keys.mean()) if contact_keys.size else 0.0,
            float((contact_keys > 0.5).sum(axis=1).mean() / max(contact_keys.shape[1], 1)) if contact_keys.size else 0.0,
            float((np.abs(np.diff(np.asarray(contact_keys > 0.5, dtype=np.float32), axis=0)).sum(axis=1).mean() / max(contact_keys.shape[1], 1))) if contact_keys.shape[0] > 1 else 0.0,
            float(np.abs(np.diff(pedal_state, axis=0)).mean()) if pedal_state.shape[0] > 1 else 0.0,
        ],
        dtype=np.float32,
    )
    _append_feature_block(pieces, names, "contact_summary", contact_summary)

    return np.concatenate(pieces).astype(np.float32), names


def enrich_segment_row_with_primitive_context(
    row: dict[str, Any],
    arrays: dict[str, np.ndarray | None],
    config: dict[str, Any],
) -> dict[str, Any]:
    context = build_primitive_context_features(row=row, arrays=arrays, config=config)
    output = dict(row)
    output.update(context["metadata"])
    return output


def build_condition_feature_vector(
    feature_vector: np.ndarray,
    feature_names: list[str],
    config: dict[str, Any],
) -> tuple[np.ndarray, list[str], list[int]]:
    selectors = config.get("gmr_condition_features")
    if not selectors:
        selectors = [
            "relative_wrist_anchor",
            "chord_center_key_id_normalized",
            "interval_pattern_embedding",
            "finger_set_id",
            "fingertip_to_target_key_offsets",
            "start_joint_state",
            "duration_bucket",
            "dynamics_bucket",
        ]
    indices = select_feature_indices(feature_names=feature_names, selectors=[str(item) for item in selectors])
    values = np.asarray(feature_vector, dtype=np.float32)
    if not indices:
        return np.zeros((0,), dtype=np.float32), [], []
    return values[np.asarray(indices, dtype=np.int64)].astype(np.float32), [feature_names[index] for index in indices], indices


def select_feature_indices(feature_names: list[str], selectors: list[str]) -> list[int]:
    prefixes = []
    exact = []
    aliases = {
        "relative_wrist_anchor": ["context_relative_wrist_anchor"],
        "chord_center_key_id_normalized": ["context_chord_geometry_0001"],
        "interval_pattern_embedding": ["context_interval_pattern"],
        "finger_set_id": ["context_finger_set_onehot"],
        "fingertip_to_target_key_offsets": ["context_fingertip_to_target"],
        "start_joint_state": ["context_start_joint_state", "joint_rel_mean"],
        "duration_bucket": ["context_duration_dynamics_0000"],
        "dynamics_bucket": ["context_duration_dynamics_0001"],
        "relative_geometry": ["context_interval_pattern", "context_key_offsets", "context_fingertip_to_target"],
        "wrist_relative_features": ["context_relative_wrist_anchor", "context_wrist_velocity"],
        "finger_set": ["context_finger_set_onehot"],
        "motion_family": ["context_motion_family_onehot"],
        "chord_interval_pattern": ["context_interval_pattern", "target_interval_hist"],
    }
    for selector in selectors:
        value = str(selector).strip()
        if not value:
            continue
        for alias in aliases.get(value, [value]):
            if alias.endswith("_"):
                prefixes.append(alias)
            elif alias in feature_names:
                exact.append(alias)
            else:
                prefixes.append(alias)
    selected: list[int] = []
    for index, name in enumerate(feature_names):
        if name in exact or any(name.startswith(prefix) for prefix in prefixes):
            selected.append(index)
    return selected


def build_primitive_context_features(row: Any, arrays: dict[str, np.ndarray | None], config: dict[str, Any]) -> dict[str, Any]:
    target_keys = _parse_key_signature(
        str(_row_value(row, "target_key_signature", default="") or _row_value(row, "key_signature", default=""))
    )
    chord_center_key_id = float(np.mean(target_keys)) if target_keys else float(_row_value(row, "key_center", default=0.0)) * 87.0
    chord_center_position = key_world_position(chord_center_key_id)
    key_positions = np.stack([key_world_position(key) for key in target_keys], axis=0).astype(np.float32) if target_keys else np.zeros((0, 3), dtype=np.float32)
    interval_pattern = [float(key - chord_center_key_id) for key in target_keys]
    white_black_pattern = [int(int(key) % 12 in {1, 3, 6, 8, 10}) for key in target_keys]
    chord_span = float(max(target_keys) - min(target_keys)) if target_keys else 0.0

    wrist_pose = arrays.get("wrist_pose")
    wrist_start = _first_vector(wrist_pose, 3)
    wrist_velocity = _first_velocity(wrist_pose, 3)
    wrist_orientation = _first_vector(wrist_pose, max(_array_width(wrist_pose) - 3, 0), offset=3)
    relative_wrist_anchor = wrist_start - chord_center_position

    fingertips = arrays.get("hand_fingertips")
    fingertip_start = _reshape_fingertips(fingertips)
    fingertip_offsets, nearest_finger_indices, target_offsets = _finger_key_offsets(
        fingertip_start=fingertip_start,
        key_positions=key_positions,
    )
    finger_set = _finger_set_from_indices(nearest_finger_indices, fallback_hand=_hand_side_from_keys(target_keys))
    hand_side = _hand_side_from_finger_set(finger_set, fallback=_hand_side_from_keys(target_keys))
    motion_family = _motion_family(row=row, chord_size=len(target_keys))
    dynamics_value = _dynamics_value(arrays.get("actions"), row=row)
    duration_steps = float(_row_value(row, "duration_steps", default=0.0))
    duration_bucket = _bucket_scalar(duration_steps, bins=float(config.get("num_duration_buckets", 8)), scale=32.0)
    dynamics_bucket = _bucket_scalar(dynamics_value, bins=float(config.get("num_dynamics_buckets", 6)), scale=1.0)

    start_joint_state = _fixed_vector(_first_vector(arrays.get("hand_joints"), int(config.get("fallback_action_dim", 39))), int(config.get("fallback_action_dim", 39)))
    start_joint_velocity = _fixed_vector(_first_vector(arrays.get("joint_velocities"), int(config.get("fallback_action_dim", 39))), int(config.get("fallback_action_dim", 39)))

    feature_blocks = {
        "context_chord_geometry": np.asarray(
            [
                float(len(target_keys)),
                float(chord_center_key_id / 87.0),
                float(chord_span / 87.0),
                float(np.mean(white_black_pattern)) if white_black_pattern else 0.0,
            ],
            dtype=np.float32,
        ),
        "context_interval_pattern": _fixed_vector(np.asarray(interval_pattern, dtype=np.float32) / 12.0, MAX_CONTEXT_KEYS),
        "context_key_offsets": _fixed_vector((key_positions - chord_center_position[None, :]).reshape(-1), MAX_CONTEXT_KEYS * 3),
        "context_white_black_pattern": _fixed_vector(np.asarray(white_black_pattern, dtype=np.float32), MAX_CONTEXT_KEYS),
        "context_wrist_start": wrist_start.astype(np.float32),
        "context_wrist_velocity": wrist_velocity.astype(np.float32),
        "context_wrist_orientation": _fixed_vector(wrist_orientation, 4),
        "context_relative_wrist_anchor": relative_wrist_anchor.astype(np.float32),
        "context_wrist_to_each_target": _fixed_vector((key_positions - wrist_start[None, :]).reshape(-1), MAX_CONTEXT_KEYS * 3),
        "context_fingertip_start": _fixed_vector(fingertip_start.reshape(-1), len(FINGER_LABELS) * 3),
        "context_fingertip_to_target": _fixed_vector(fingertip_offsets.reshape(-1), MAX_CONTEXT_KEYS * 3),
        "context_fingertip_target_lateral_height": _fixed_vector(target_offsets.reshape(-1), MAX_CONTEXT_KEYS * 3),
        "context_nearest_finger": _fixed_vector(np.asarray(nearest_finger_indices, dtype=np.float32) / max(len(FINGER_LABELS) - 1, 1), MAX_CONTEXT_KEYS),
        "context_finger_set_onehot": _finger_set_onehot(finger_set),
        "context_hand_side_onehot": _hand_side_onehot(hand_side),
        "context_start_joint_state": start_joint_state,
        "context_start_joint_velocity": start_joint_velocity,
        "context_duration_dynamics": np.asarray([duration_bucket, dynamics_bucket, duration_steps / 64.0, dynamics_value], dtype=np.float32),
        "context_motion_family_onehot": _motion_family_onehot(motion_family),
    }

    metadata = {
        "target_key_ids": json.dumps([int(key) for key in target_keys]),
        "target_key_ids_json": json.dumps([int(key) for key in target_keys]),
        "chord_size": int(len(target_keys) or int(_row_value(row, "chord_size", default=0))),
        "chord_center_key_id": float(chord_center_key_id),
        "chord_center_key_id_normalized": float(chord_center_key_id / 87.0),
        "chord_span_semitones": float(chord_span),
        "interval_pattern": json.dumps([round(float(value), 4) for value in interval_pattern]),
        "interval_pattern_bucket": _interval_pattern_bucket(interval_pattern),
        "white_black_pattern": json.dumps([int(value) for value in white_black_pattern]),
        "key_world_positions": json.dumps(key_positions.round(6).tolist()),
        "wrist_world_position": json.dumps(wrist_start.round(6).tolist()),
        "wrist_world_orientation": json.dumps(_fixed_vector(wrist_orientation, 4).round(6).tolist()),
        "wrist_velocity": json.dumps(wrist_velocity.round(6).tolist()),
        "wrist_to_chord_center_offset": json.dumps(relative_wrist_anchor.round(6).tolist()),
        "relative_wrist_anchor": json.dumps(relative_wrist_anchor.round(6).tolist()),
        "wrist_to_each_target_key_offset": json.dumps((key_positions - wrist_start[None, :]).round(6).tolist()),
        "fingertip_world_positions": json.dumps(fingertip_start.round(6).tolist()),
        "fingertip_to_target_key_offsets": json.dumps(fingertip_offsets.round(6).tolist()),
        "nearest_finger_to_each_target_key": json.dumps([int(value) for value in nearest_finger_indices]),
        "nearest_finger_labels": json.dumps([FINGER_LABELS[int(value)] for value in nearest_finger_indices if 0 <= int(value) < len(FINGER_LABELS)]),
        "contact_finger_ids": json.dumps([int(value) for value in nearest_finger_indices]),
        "finger_set_id": finger_set,
        "finger_set": finger_set,
        "fingertip_height_above_key": json.dumps(target_offsets[:, 2].round(6).tolist() if target_offsets.size else []),
        "lateral_fingertip_key_offsets": json.dumps(target_offsets[:, :2].round(6).tolist() if target_offsets.size else []),
        "start_joint_state": json.dumps(start_joint_state.round(6).tolist()),
        "start_joint_velocity": json.dumps(start_joint_velocity.round(6).tolist()),
        "normalized_joint_state": json.dumps(start_joint_state.round(6).tolist()),
        "hand_side": hand_side,
        "segment_alignment": str(_row_value(row, "segment_alignment", default="prepress_to_onset")),
        "key_inactive_at_segment_start": bool(_row_value(row, "inactive_start", default=False)),
        "key_activation_onset_step": int(_row_value(row, "target_onset_step", default=-1)),
        "contact_step": -1,
        "duration_steps": int(_row_value(row, "duration_steps", default=0)),
        "dynamics_value": float(dynamics_value),
        "dynamics_bucket": int(round(dynamics_bucket)),
        "duration_bucket": int(round(duration_bucket)),
        "motion_family": motion_family,
        "primitive_frame_mode": str(config.get("primitive_frame", {}).get("mode", config.get("primitive_frame_mode", "absolute"))),
    }
    return {"feature_blocks": feature_blocks, "metadata": metadata}


def build_gmr_target_from_arrays(arrays: dict[str, np.ndarray | None], config: dict[str, Any]) -> tuple[np.ndarray, str]:
    if bool(config.get("gmr_target_actions", True)):
        if arrays.get("actions") is None:
            raise ValueError("gmr_target_actions=true requires action trajectories; refusing hand-joint fallback.")
        target_name = "actions"
    else:
        target_name = "hand_joints"
    trajectory = arrays.get(target_name)
    if trajectory is None:
        trajectory = arrays.get("hand_joints")
        target_name = "hand_joints"
    if trajectory is None:
        raise ValueError("Cannot build GMR target without actions or hand_joints.")
    return resample_time_axis(np.asarray(trajectory, dtype=np.float32), resolve_gmr_resample_steps(config)), target_name


def resolve_gmr_resample_steps(config: dict[str, Any]) -> int:
    return int(config.get("gmr_resample_steps", config.get("gmr_horizon", 32)))


def normalize_hand_frame(hand_joints: np.ndarray, wrist_pose: np.ndarray | None, row: Any, config: dict[str, Any]) -> tuple[np.ndarray, dict[str, float]]:
    normalized = np.asarray(hand_joints, dtype=np.float32).copy()
    used_wrist_frame = 0.0
    frame_config = config.get("primitive_frame", {}) if isinstance(config.get("primitive_frame"), dict) else {}
    normalize_wrist_translation = bool(frame_config.get("normalize_wrist_translation", config.get("relative_wrist_frame", True)))
    normalize_chord_center = bool(frame_config.get("normalize_chord_center", config.get("relative_key_center_frame", True)))
    if normalize_wrist_translation and wrist_pose is not None and normalized.shape[1] % 3 == 0 and np.asarray(wrist_pose).shape[1] >= 3:
        wrist_pose = np.asarray(wrist_pose, dtype=np.float32)
        repeats = normalized.shape[1] // 3
        normalized -= np.tile(wrist_pose[:, :3], (1, repeats))
        used_wrist_frame = 1.0
    else:
        normalized -= normalized[:1]

    key_center_offset = float(_row_value(row, "key_center", default=0.0)) - 0.5
    if normalize_chord_center and normalized.shape[1] % 3 == 0:
        normalized[:, 0::3] -= key_center_offset

    if bool(config.get("hand_specific_normalization", True)):
        scale = np.clip(normalized.std(axis=0), 1e-4, None)
        normalized /= scale
    return normalized.astype(np.float32), {"used_wrist_frame": used_wrist_frame, "key_center_offset": key_center_offset}


def _append_feature_block(pieces: list[np.ndarray], names: list[str], prefix: str, values: np.ndarray) -> None:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    pieces.append(array)
    names.extend([f"{prefix}_{index:04d}" for index in range(array.size)])


def _trajectory_summary(array: np.ndarray) -> list[float]:
    values = np.linalg.norm(np.asarray(array, dtype=np.float32), axis=1) if np.asarray(array).ndim == 2 else np.asarray(array, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(values.mean()), float(values.std()), float(values.max()), float(values[-1] - values[0])]


def _parse_key_signature(signature: str) -> list[int]:
    keys: list[int] = []
    for group in str(signature or "").replace("|", "-").split("-"):
        text = group.strip()
        if not text or text == "none":
            continue
        try:
            key = int(text)
        except ValueError:
            continue
        if 0 <= key < 88:
            keys.append(key)
    return sorted(set(keys))


def key_world_position(key_id: float | int) -> np.ndarray:
    x = (float(key_id) - 43.5) * KEY_WIDTH_METERS
    return np.asarray([x, 0.0, 0.0], dtype=np.float32)


def _target_keyset_vector(keys: list[int]) -> np.ndarray:
    vector = np.zeros((88,), dtype=np.float32)
    for key in keys:
        vector[int(key)] = 1.0
    return vector


def _target_keyset_summary(keys: list[int]) -> np.ndarray:
    if not keys:
        return np.zeros((7,), dtype=np.float32)
    values = np.asarray(keys, dtype=np.float32)
    black_key_classes = {1, 3, 6, 8, 10}
    black_keys = sum(int(key % 12 in black_key_classes) for key in keys)
    left_hand_proxy = sum(int(key < 44) for key in keys)
    right_hand_proxy = len(keys) - left_hand_proxy
    return np.asarray(
        [
            float(len(keys)),
            float(values.min() / 87.0),
            float(values.max() / 87.0),
            float(values.mean() / 87.0),
            float((values.max() - values.min()) / 87.0),
            float(black_keys / max(len(keys), 1)),
            float((right_hand_proxy - left_hand_proxy) / max(len(keys), 1)),
        ],
        dtype=np.float32,
    )


def _target_interval_histogram(keys: list[int]) -> np.ndarray:
    histogram = np.zeros((12,), dtype=np.float32)
    if len(keys) < 2:
        return histogram
    for left, right in zip(keys[:-1], keys[1:]):
        interval = int(abs(right - left) % 12)
        histogram[interval] += 1.0
    total = float(histogram.sum())
    return histogram / total if total > 0 else histogram


def _safe_json(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    try:
        return json.loads(str(payload))
    except Exception:
        return {}


def _row_value(row: Any, name: str, default: Any | None = None) -> Any:
    if isinstance(row, dict):
        return row.get(name, default)
    return getattr(row, name, default)


def _array_width(array: np.ndarray | None) -> int:
    if array is None:
        return 0
    values = np.asarray(array)
    return int(values.shape[1]) if values.ndim >= 2 else int(values.size)


def _first_vector(array: np.ndarray | None, size: int, offset: int = 0) -> np.ndarray:
    if size <= 0:
        return np.zeros((0,), dtype=np.float32)
    if array is None:
        return np.zeros((size,), dtype=np.float32)
    values = np.asarray(array, dtype=np.float32)
    if values.size == 0:
        return np.zeros((size,), dtype=np.float32)
    first = values.reshape(values.shape[0], -1)[0]
    return _fixed_vector(first[offset : offset + size], size)


def _first_velocity(array: np.ndarray | None, size: int) -> np.ndarray:
    if array is None:
        return np.zeros((size,), dtype=np.float32)
    values = np.asarray(array, dtype=np.float32).reshape(np.asarray(array).shape[0], -1)
    if values.shape[0] < 2:
        return np.zeros((size,), dtype=np.float32)
    return _fixed_vector(values[1, :size] - values[0, :size], size)


def _fixed_vector(values: np.ndarray, size: int) -> np.ndarray:
    output = np.zeros((int(size),), dtype=np.float32)
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    width = min(output.size, array.size)
    if width > 0:
        output[:width] = array[:width]
    return output


def _reshape_fingertips(fingertips: np.ndarray | None) -> np.ndarray:
    if fingertips is None:
        return np.zeros((len(FINGER_LABELS), 3), dtype=np.float32)
    values = np.asarray(fingertips, dtype=np.float32)
    if values.size == 0 or values.shape[-1] < 3:
        return np.zeros((len(FINGER_LABELS), 3), dtype=np.float32)
    start = values.reshape(values.shape[0], -1, 3)[0]
    output = np.zeros((len(FINGER_LABELS), 3), dtype=np.float32)
    count = min(output.shape[0], start.shape[0])
    output[:count] = start[:count]
    return output


def _finger_key_offsets(
    *,
    fingertip_start: np.ndarray,
    key_positions: np.ndarray,
) -> tuple[np.ndarray, list[int], np.ndarray]:
    if key_positions.size == 0:
        return np.zeros((0, 3), dtype=np.float32), [], np.zeros((0, 3), dtype=np.float32)
    distances = np.linalg.norm(fingertip_start[:, None, :] - key_positions[None, :, :], axis=-1)
    nearest = np.argmin(distances, axis=0).astype(int).tolist()
    offsets = []
    lateral_height = []
    for key_index, finger_index in enumerate(nearest):
        offset = fingertip_start[int(finger_index)] - key_positions[key_index]
        offsets.append(offset)
        lateral_height.append(offset)
    return np.asarray(offsets, dtype=np.float32), nearest, np.asarray(lateral_height, dtype=np.float32)


def _finger_set_from_indices(indices: list[int], fallback_hand: str) -> str:
    if not indices:
        return f"{fallback_hand}_unknown" if fallback_hand in {"left", "right"} else "unknown"
    labels = sorted({FINGER_LABELS[int(index)] for index in indices if 0 <= int(index) < len(FINGER_LABELS)})
    return "+".join(labels) if labels else "unknown"


def _hand_side_from_finger_set(finger_set: str, fallback: str) -> str:
    labels = [item for item in str(finger_set).split("+") if item]
    has_right = any(item.startswith("right_") for item in labels)
    has_left = any(item.startswith("left_") for item in labels)
    if has_right and not has_left:
        return "right"
    if has_left and not has_right:
        return "left"
    if has_left and has_right:
        return "both"
    return fallback


def _hand_side_from_keys(keys: list[int]) -> str:
    if not keys:
        return "unknown"
    return "left" if float(np.mean(keys)) < 44.0 else "right"


def _finger_set_onehot(finger_set: str) -> np.ndarray:
    values = np.zeros((len(FINGER_LABELS),), dtype=np.float32)
    labels = set(str(finger_set).split("+"))
    for index, label in enumerate(FINGER_LABELS):
        if label in labels:
            values[index] = 1.0
    return values


def _hand_side_onehot(hand_side: str) -> np.ndarray:
    labels = ["left", "right", "both", "unknown"]
    values = np.zeros((len(labels),), dtype=np.float32)
    values[labels.index(hand_side if hand_side in labels else "unknown")] = 1.0
    return values


def _motion_family(row: Any, chord_size: int) -> str:
    value = str(_row_value(row, "motion_family", default="") or _row_value(row, "coarse_family", default=""))
    if value:
        return value
    if chord_size <= 1:
        return "single_press"
    if chord_size == 2:
        return "dyad_press"
    if chord_size == 3:
        return "triad_press"
    return "chord_press"


def _motion_family_onehot(motion_family: str) -> np.ndarray:
    labels = ["single_press", "dyad_press", "triad_press", "chord_press", "repeat_press", "release", "transition", "reposition", "mixed_unknown"]
    values = np.zeros((len(labels),), dtype=np.float32)
    index = labels.index(motion_family) if motion_family in labels else labels.index("mixed_unknown")
    values[index] = 1.0
    return values


def _dynamics_value(actions: np.ndarray | None, row: Any) -> float:
    if actions is not None and np.asarray(actions).size:
        values = np.asarray(actions, dtype=np.float32)
        delta = np.diff(values, axis=0, prepend=values[:1])
        return float(np.linalg.norm(delta, axis=1).mean())
    return float(_row_value(row, "motion_energy", default=0.0))


def _bucket_scalar(value: float, *, bins: float, scale: float) -> float:
    if bins <= 1:
        return 0.0
    clipped = np.clip(float(value) / max(float(scale), 1e-6), 0.0, 1.0)
    return float(np.floor(clipped * (float(bins) - 1.0)))


def _interval_pattern_bucket(interval_pattern: list[float]) -> str:
    if not interval_pattern:
        return "none"
    return ",".join(str(int(round(value))) for value in interval_pattern)


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
