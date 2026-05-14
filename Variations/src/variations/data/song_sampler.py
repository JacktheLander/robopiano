from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from variations.data.rp1m_loader import list_songs, read_trajectory, trajectory_count


@dataclass
class SongRanking:
    song_id: str
    trajectory_ids: list[int]
    rows: list[dict[str, float | int | str]]
    best_key_f1: float
    dropped: bool
    drop_reason: str | None = None


def stride_pick_songs(song_ids: list[str], num_songs: int | None) -> list[str]:
    ordered = sorted(str(song_id) for song_id in song_ids)
    if num_songs is None or num_songs >= len(ordered):
        return ordered
    if num_songs <= 0:
        return []
    step = len(ordered) / float(num_songs)
    indices = [min(int(idx * step), len(ordered) - 1) for idx in range(num_songs)]
    selected = [ordered[idx] for idx in indices]
    return sorted(selected)


def _seed_for_song(seed: int, song_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{song_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


def sample_trajectory_ids(num_trajectories: int, sample_size: int, seed: int, song_id: str) -> list[int]:
    if num_trajectories <= 0:
        return []
    if sample_size <= 0 or sample_size >= num_trajectories:
        return list(range(num_trajectories))
    rng = np.random.default_rng(_seed_for_song(seed, song_id))
    return [int(x) for x in rng.choice(num_trajectories, size=sample_size, replace=False)]


def key_metrics(goals: np.ndarray, piano_states: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    goal = np.asarray(goals)[..., :88] > threshold
    played = np.asarray(piano_states)[..., :88] > threshold
    t = min(goal.shape[0], played.shape[0])
    goal = goal[:t]
    played = played[:t]
    tp = np.logical_and(goal, played).sum(dtype=np.float64)
    fp = np.logical_and(~goal, played).sum(dtype=np.float64)
    fn = np.logical_and(goal, ~played).sum(dtype=np.float64)
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    mispress = fp / max(played.sum(dtype=np.float64), 1.0)
    return {
        "key_precision": float(precision),
        "key_recall": float(recall),
        "key_f1": float(f1),
        "mispress_rate": float(mispress),
        "score": float(f1 - mispress),
    }


def score_trajectory(song_group, song_id: str, trajectory_id: int, threshold: float) -> dict[str, float | int | str]:
    traj = read_trajectory(song_group, trajectory_id, arrays=["goals", "piano_states"])
    metrics = key_metrics(traj["goals"], traj["piano_states"], threshold=threshold)
    return {"song_id": song_id, "trajectory_id": int(trajectory_id), **metrics}


def rank_song_trajectories(
    song_group,
    song_id: str,
    *,
    seed: int,
    score_sample_size: int,
    top_k_trajectories: int,
    min_song_key_f1: float,
    threshold: float,
) -> SongRanking:
    try:
        n = trajectory_count(song_group)
        sampled_ids = sample_trajectory_ids(n, score_sample_size, seed, song_id)
        rows = [score_trajectory(song_group, song_id, trajectory_id, threshold) for trajectory_id in sampled_ids]
    except Exception as exc:
        return SongRanking(song_id=song_id, trajectory_ids=[], rows=[], best_key_f1=float("nan"), dropped=True, drop_reason=str(exc))

    rows = sorted(rows, key=lambda row: (-float(row["score"]), int(row["trajectory_id"])))
    for rank, row in enumerate(rows):
        row["rank"] = rank
    best_key_f1 = float(rows[0]["key_f1"]) if rows else float("nan")
    if not rows:
        return SongRanking(song_id=song_id, trajectory_ids=[], rows=rows, best_key_f1=best_key_f1, dropped=True, drop_reason="no sampled trajectories")
    if np.isfinite(best_key_f1) and best_key_f1 < min_song_key_f1:
        return SongRanking(
            song_id=song_id,
            trajectory_ids=[],
            rows=rows,
            best_key_f1=best_key_f1,
            dropped=True,
            drop_reason=f"best key_f1 {best_key_f1:.4f} < min_song_key_f1 {min_song_key_f1:.4f}",
        )
    kept = [int(row["trajectory_id"]) for row in rows[:top_k_trajectories]]
    return SongRanking(song_id=song_id, trajectory_ids=kept, rows=rows, best_key_f1=best_key_f1, dropped=False)


def select_and_rank_songs(
    root,
    *,
    num_songs: int,
    seed: int,
    score_sample_size: int,
    top_k_trajectories: int,
    min_song_key_f1: float,
    threshold: float,
) -> tuple[list[SongRanking], list[SongRanking]]:
    selected = stride_pick_songs(list_songs(root), num_songs)
    kept: list[SongRanking] = []
    dropped: list[SongRanking] = []
    for song_id in selected:
        ranking = rank_song_trajectories(
            root[song_id],
            song_id,
            seed=seed,
            score_sample_size=score_sample_size,
            top_k_trajectories=top_k_trajectories,
            min_song_key_f1=min_song_key_f1,
            threshold=threshold,
        )
        if ranking.dropped:
            dropped.append(ranking)
        else:
            kept.append(ranking)
    kept.sort(key=lambda item: item.song_id)
    return kept, dropped

