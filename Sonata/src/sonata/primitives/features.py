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


def normalize_hand_frame(hand_joints: np.ndarray, wrist_pose: np.ndarray | None, row: Any, config: dict[str, Any]) -> tuple[np.ndarray, dict[str, float]]:
    normalized = np.asarray(hand_joints, dtype=np.float32).copy()
    used_wrist_frame = 0.0
    if bool(config.get("relative_wrist_frame", True)) and wrist_pose is not None and normalized.shape[1] % 3 == 0 and np.asarray(wrist_pose).shape[1] >= 3:
        wrist_pose = np.asarray(wrist_pose, dtype=np.float32)
        repeats = normalized.shape[1] // 3
        normalized -= np.tile(wrist_pose[:, :3], (1, repeats))
        used_wrist_frame = 1.0
    else:
        normalized -= normalized[:1]

    key_center_offset = float(_row_value(row, "key_center", default=0.0)) - 0.5
    if bool(config.get("relative_key_center_frame", True)) and normalized.shape[1] % 3 == 0:
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
