from __future__ import annotations

import numpy as np
import pandas as pd


def resample_array(arr, length: int) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[0] == length:
        return x.copy()
    if x.shape[0] <= 1:
        return np.repeat(x[:1], length, axis=0) if x.shape[0] else np.zeros((length, 1), dtype=np.float32)
    src = np.linspace(0.0, 1.0, x.shape[0])
    dst = np.linspace(0.0, 1.0, length)
    cols = [np.interp(dst, src, x[:, i]) for i in range(x.shape[1])]
    return np.stack(cols, axis=1).astype(np.float32)


def _stats(prefix: str, arr) -> tuple[list[float], list[str]]:
    if arr is None:
        return [], []
    x = np.asarray(arr, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    vals = np.concatenate([mean, std])
    names = [f"{prefix}_mean_{i}" for i in range(mean.shape[0])] + [f"{prefix}_std_{i}" for i in range(std.shape[0])]
    return vals.astype(np.float32).tolist(), names


def segment_feature(data: dict[str, np.ndarray], row, include_relative_song_time: bool, relative_song_time_weight: float) -> tuple[np.ndarray, list[str]]:
    traj_idx = int(row["trajectory_index"])
    start = int(row["start_t"])
    end = int(row["end_t"])
    actions = np.asarray(data["actions"][traj_idx, start:end], dtype=np.float32)
    feats: list[float] = []
    names: list[str] = []

    vals, ns = _stats("action", actions)
    feats.extend(vals); names.extend(ns)
    delta = actions[-1] - actions[0] if actions.shape[0] else np.zeros(data["actions"].shape[-1], dtype=np.float32)
    feats.extend(delta.astype(np.float32).tolist())
    names.extend([f"action_delta_{i}" for i in range(delta.shape[0])])
    feats.extend([float(row.get("action_energy", np.mean(actions * actions))), float(row["duration"])])
    names.extend(["action_energy", "duration"])

    for key, prefix in [("goals", "goal"), ("piano_states", "piano"), ("hand_joints", "joint"), ("hand_fingertips", "fingertip")]:
        if key in data:
            vals, ns = _stats(prefix, data[key][traj_idx, start:end])
            feats.extend(vals); names.extend(ns)

    if include_relative_song_time:
        total_t = max(int(data["actions"].shape[1]), 1)
        feats.append(float(start) / total_t * float(relative_song_time_weight))
        names.append("relative_song_time_weighted")
    return np.asarray(feats, dtype=np.float32), names


def features_for_segments(data: dict[str, np.ndarray], segments: pd.DataFrame, include_relative_song_time: bool = True, relative_song_time_weight: float = 0.2) -> tuple[np.ndarray, list[str]]:
    vectors = []
    feature_names = None
    for _, row in segments.iterrows():
        vec, names = segment_feature(data, row, include_relative_song_time, relative_song_time_weight)
        vectors.append(vec)
        if feature_names is None:
            feature_names = names
    if not vectors:
        raise RuntimeError("No segments available for feature extraction.")
    lengths = {len(v) for v in vectors}
    if len(lengths) != 1:
        raise RuntimeError(f"Inconsistent feature lengths across segments: {sorted(lengths)}")
    return np.stack(vectors, axis=0), (feature_names or [])


def feature_for_target_segment(traj: dict[str, np.ndarray], row, feature_names: list[str], include_relative_song_time: bool, relative_song_time_weight: float) -> np.ndarray:
    data = {k: v[None, ...] for k, v in traj.items() if isinstance(v, np.ndarray) and k != "trajectory_id" and np.asarray(v).ndim >= 2}
    data["actions"] = np.asarray(traj["actions"])[None, ...]
    row = dict(row)
    row["trajectory_index"] = 0
    vec, names = segment_feature(data, row, include_relative_song_time, relative_song_time_weight)
    if len(vec) != len(feature_names):
        raise RuntimeError(f"Target feature length {len(vec)} does not match training length {len(feature_names)}")
    return vec
