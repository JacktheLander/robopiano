from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from partita.evaluation.metrics import key_metrics


@dataclass
class SegmentConfig:
    min_segment_len: int = 4
    max_segment_len: int = 32
    use_goal_boundaries: bool = True
    use_piano_state_boundaries: bool = True
    use_action_derivative_boundaries: bool = True
    action_derivative_percentile: float = 90.0
    key_threshold: float = 0.5


def _event_from_key_changes(arr, threshold: float) -> np.ndarray | None:
    if arr is None:
        return None
    keys = np.asarray(arr) > threshold
    if keys.ndim != 2 or keys.shape[0] < 2:
        return None
    changes = np.any(keys[1:] != keys[:-1], axis=1)
    return np.concatenate([[False], changes])


def _event_from_action_derivative(actions, percentile: float) -> np.ndarray | None:
    if actions is None:
        return None
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return None
    mag = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    if not np.any(np.isfinite(mag)):
        return None
    threshold = np.percentile(mag[np.isfinite(mag)], percentile)
    spikes = mag >= threshold
    return np.concatenate([[False], spikes])


def segment_boundaries(actions, goals=None, piano_states=None, config: SegmentConfig | None = None) -> list[tuple[int, int]]:
    cfg = config or SegmentConfig()
    if actions is None:
        raise RuntimeError("actions are required for segmentation")
    T = int(np.asarray(actions).shape[0])
    if T <= 0:
        return []
    events = np.zeros(T, dtype=bool)
    if cfg.use_goal_boundaries:
        ev = _event_from_key_changes(goals, cfg.key_threshold)
        if ev is not None:
            events |= ev
    if cfg.use_piano_state_boundaries:
        ev = _event_from_key_changes(piano_states, cfg.key_threshold)
        if ev is not None:
            events |= ev
    if cfg.use_action_derivative_boundaries:
        ev = _event_from_action_derivative(actions, cfg.action_derivative_percentile)
        if ev is not None:
            events |= ev

    min_len = max(1, int(cfg.min_segment_len))
    max_len = max(min_len, int(cfg.max_segment_len))
    boundaries = [0]
    last = 0
    for t in range(1, T):
        duration = t - last
        if duration >= max_len or (events[t] and duration >= min_len):
            boundaries.append(t)
            last = t
    if boundaries[-1] != T:
        if T - boundaries[-1] < min_len and len(boundaries) > 1:
            boundaries[-1] = T
        else:
            boundaries.append(T)
    return [(int(a), int(b)) for a, b in zip(boundaries[:-1], boundaries[1:]) if b > a]


def segment_metrics(actions, goals, piano_states, start: int, end: int, threshold: float) -> dict[str, float]:
    a = np.asarray(actions[start:end], dtype=np.float32)
    g = None if goals is None else np.asarray(goals[start:end])
    p = None if piano_states is None else np.asarray(piano_states[start:end])
    km = key_metrics(g, p, threshold=threshold)
    return {
        "duration": int(end - start),
        "num_goal_keys": float(np.sum(g > threshold) / max(end - start, 1)) if g is not None else float("nan"),
        "num_played_keys": float(np.sum(p > threshold) / max(end - start, 1)) if p is not None else float("nan"),
        "action_energy": float(np.mean(a * a)) if a.size else 0.0,
        "key_f1_for_segment": float(km["key_f1"]),
    }


def segment_dataset(data: dict[str, np.ndarray], segmentation_cfg: dict, key_threshold: float = 0.5) -> pd.DataFrame:
    actions = data.get("actions")
    if actions is None:
        raise RuntimeError("selected_trajectories.npz does not contain required actions array")
    goals = data.get("goals")
    piano_states = data.get("piano_states")
    trajectory_ids = data.get("trajectory_ids", np.arange(actions.shape[0]))
    cfg = SegmentConfig(
        min_segment_len=int(segmentation_cfg.get("min_segment_len", 4)),
        max_segment_len=int(segmentation_cfg.get("max_segment_len", 32)),
        use_goal_boundaries=bool(segmentation_cfg.get("use_goal_boundaries", True)),
        use_piano_state_boundaries=bool(segmentation_cfg.get("use_piano_state_boundaries", True)),
        use_action_derivative_boundaries=bool(segmentation_cfg.get("use_action_derivative_boundaries", True)),
        action_derivative_percentile=float(segmentation_cfg.get("action_derivative_percentile", 90)),
        key_threshold=float(key_threshold),
    )
    rows = []
    global_id = 0
    for traj_idx in range(actions.shape[0]):
        traj_goals = None if goals is None else goals[traj_idx]
        traj_piano = None if piano_states is None else piano_states[traj_idx]
        bounds = segment_boundaries(actions[traj_idx], traj_goals, traj_piano, cfg)
        for local_id, (start, end) in enumerate(bounds):
            metrics = segment_metrics(actions[traj_idx], traj_goals, traj_piano, start, end, cfg.key_threshold)
            rows.append({
                "global_segment_id": global_id,
                "trajectory_id": int(trajectory_ids[traj_idx]),
                "trajectory_index": int(traj_idx),
                "local_segment_id": int(local_id),
                "start_t": int(start),
                "end_t": int(end),
                **metrics,
            })
            global_id += 1
    return pd.DataFrame(rows)


def segment_single_trajectory(traj: dict[str, np.ndarray], segmentation_cfg: dict, key_threshold: float = 0.5) -> pd.DataFrame:
    cfg = SegmentConfig(
        min_segment_len=int(segmentation_cfg.get("min_segment_len", 4)),
        max_segment_len=int(segmentation_cfg.get("max_segment_len", 32)),
        use_goal_boundaries=bool(segmentation_cfg.get("use_goal_boundaries", True)),
        use_piano_state_boundaries=bool(segmentation_cfg.get("use_piano_state_boundaries", True)),
        use_action_derivative_boundaries=bool(segmentation_cfg.get("use_action_derivative_boundaries", True)),
        action_derivative_percentile=float(segmentation_cfg.get("action_derivative_percentile", 90)),
        key_threshold=float(key_threshold),
    )
    bounds = segment_boundaries(traj.get("actions"), traj.get("goals"), traj.get("piano_states"), cfg)
    trajectory_id = int(np.asarray(traj.get("trajectory_id", 0)).reshape(-1)[0])
    rows = []
    for local_id, (start, end) in enumerate(bounds):
        rows.append({
            "global_segment_id": int(local_id),
            "trajectory_id": trajectory_id,
            "trajectory_index": 0,
            "local_segment_id": int(local_id),
            "start_t": int(start),
            "end_t": int(end),
            **segment_metrics(traj["actions"], traj.get("goals"), traj.get("piano_states"), start, end, cfg.key_threshold),
        })
    return pd.DataFrame(rows)
