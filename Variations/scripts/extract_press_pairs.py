from __future__ import annotations

import argparse
import pickle
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

VARIATIONS_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = VARIATIONS_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from variations.data.press_extractor import PressCandidates, extract_press_candidates, goal_fingerprint
from variations.data.rp1m_loader import open_rp1m_root, read_trajectory
from variations.data.song_sampler import SongRanking, select_and_rank_songs
from variations.utils.config import extraction_root, load_config, rp1m_root
from variations.utils.io import ensure_dir, load_json, safe_stem, save_csv, save_json


STATE_VERSION = 1


def empty_song_npz(path: Path, song_id: str) -> None:
    if path.exists():
        return
    ensure_dir(path.parent)
    np.savez_compressed(
        path,
        target_keys=np.zeros((0, 88), dtype=np.float32),
        hand_state=np.zeros((0, 76), dtype=np.float32),
        song_id=np.asarray(song_id),
        trajectory_ids=np.zeros((0,), dtype=np.int32),
        steps=np.zeros((0,), dtype=np.int32),
    )


def append_song_npz(path: Path, song_id: str, target_keys: np.ndarray, hand_state: np.ndarray, trajectory_ids: np.ndarray, steps: np.ndarray) -> None:
    empty_song_npz(path, song_id)
    old = np.load(path, allow_pickle=False)
    out = {
        "target_keys": np.concatenate([old["target_keys"], target_keys.astype(np.float32)], axis=0),
        "hand_state": np.concatenate([old["hand_state"], hand_state.astype(np.float32)], axis=0),
        "song_id": np.asarray(song_id),
        "trajectory_ids": np.concatenate([old["trajectory_ids"], trajectory_ids.astype(np.int32)], axis=0),
        "steps": np.concatenate([old["steps"], steps.astype(np.int32)], axis=0),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    np.savez_compressed(tmp, **out)
    tmp_npz = tmp if tmp.exists() else tmp.with_suffix(tmp.suffix + ".npz")
    tmp_npz.replace(path)


def load_seen(path: Path) -> set[bytes]:
    if not path.exists():
        return set()
    with path.open("rb") as f:
        return pickle.load(f)


def save_seen(path: Path, seen: set[bytes]) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        pickle.dump(seen, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def accept_candidates(
    candidates: PressCandidates,
    *,
    seen_goal_fp: set[bytes],
    dedupe: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    accepted_indices = []
    duplicates = 0
    for idx, target in enumerate(candidates.target_keys):
        fp = goal_fingerprint(target)
        if dedupe and fp in seen_goal_fp:
            duplicates += 1
            continue
        seen_goal_fp.add(fp)
        accepted_indices.append(idx)
    if not accepted_indices:
        return (
            np.zeros((0, 88), dtype=np.float32),
            np.zeros((0, 76), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
            duplicates,
        )
    idx = np.asarray(accepted_indices, dtype=np.int64)
    return candidates.target_keys[idx], candidates.hand_state[idx], candidates.steps[idx], candidates.num_onsets[idx], duplicates


def ranking_rows(rankings: list[SongRanking], dropped: list[SongRanking]) -> list[dict[str, Any]]:
    rows = []
    for ranking in rankings + dropped:
        for row in ranking.rows:
            rows.append({**row, "dropped_song": ranking.dropped, "drop_reason": ranking.drop_reason or ""})
        if not ranking.rows:
            rows.append({
                "song_id": ranking.song_id,
                "trajectory_id": "",
                "rank": "",
                "key_precision": "",
                "key_recall": "",
                "key_f1": "",
                "mispress_rate": "",
                "score": "",
                "dropped_song": ranking.dropped,
                "drop_reason": ranking.drop_reason or "",
            })
    return rows


def save_manifest(root: Path, rankings: list[SongRanking], stats: dict[str, dict[str, int]]) -> None:
    rows = []
    for ranking in rankings:
        safe = safe_stem(ranking.song_id)
        song_stats = stats.setdefault(ranking.song_id, {"candidates_seen": 0, "rows_accepted": 0, "goal_duplicates_skipped": 0})
        rows.append({
            "song_id": ranking.song_id,
            "safe_song_id": safe,
            "path": f"song_{safe}.npz",
            "num_ranked_trajectories": len(ranking.trajectory_ids),
            "candidates_seen": song_stats["candidates_seen"],
            "rows_accepted": song_stats["rows_accepted"],
            "goal_duplicates_skipped": song_stats["goal_duplicates_skipped"],
        })
    save_csv(root / "manifest.csv", rows, fieldnames=[
        "song_id",
        "safe_song_id",
        "path",
        "num_ranked_trajectories",
        "candidates_seen",
        "rows_accepted",
        "goal_duplicates_skipped",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract unique successful RP1M press pairs for Variations.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    out_root = extraction_root(config)
    if args.force_rebuild and out_root.exists():
        shutil.rmtree(out_root)
    ensure_dir(out_root)
    state_dir = ensure_dir(out_root / "extraction_state")

    sampling = config.get("sampling", {})
    extraction = config.get("extraction", {})
    if not bool(extraction.get("round_robin_trajectories", True)):
        raise ValueError("round_robin_trajectories must remain true for the Variations extraction policy.")

    root = open_rp1m_root(rp1m_root(config))
    rankings, dropped = select_and_rank_songs(
        root,
        num_songs=int(sampling.get("num_songs", 5)),
        seed=int(config.get("seed", 42)),
        score_sample_size=int(sampling.get("score_sample_size", 8)),
        top_k_trajectories=int(sampling.get("top_k_trajectories", 4)),
        min_song_key_f1=float(sampling.get("min_song_key_f1", 0.0)),
        threshold=float(sampling.get("key_threshold", 0.5)),
    )
    save_csv(out_root / "trajectory_rankings.csv", ranking_rows(rankings, dropped))
    save_json(out_root / "selected_songs.json", {
        "kept": [{"song_id": r.song_id, "trajectory_ids": r.trajectory_ids, "best_key_f1": r.best_key_f1} for r in rankings],
        "dropped": [{"song_id": r.song_id, "reason": r.drop_reason, "best_key_f1": r.best_key_f1} for r in dropped],
    })

    for ranking in rankings:
        empty_song_npz(out_root / f"song_{safe_stem(ranking.song_id)}.npz", ranking.song_id)

    resume_path = state_dir / "resume.json"
    stats_path = state_dir / "song_stats.json"
    seen_path = state_dir / "seen_goal_fp.pkl"
    if args.no_resume:
        resume = {"rank": 0, "song_index": 0, "completed": False}
        stats: dict[str, dict[str, int]] = {}
        seen_goal_fp: set[bytes] = set()
    else:
        resume = load_json(resume_path) if resume_path.exists() else {"rank": 0, "song_index": 0, "completed": False}
        stats = load_json(stats_path) if stats_path.exists() else {}
        seen_goal_fp = load_seen(seen_path)
    if resume.get("completed"):
        print(f"Extraction already completed at {out_root}")
        return

    top_k = int(sampling.get("top_k_trajectories", 4))
    threshold = float(sampling.get("key_threshold", 0.5))
    dedupe = bool(extraction.get("deduplicate_unique_goal_keys", True))
    start_rank = int(resume.get("rank", 0))
    start_song = int(resume.get("song_index", 0))

    for rank in range(start_rank, top_k):
        song_start = start_song if rank == start_rank else 0
        for song_index in range(song_start, len(rankings)):
            ranking = rankings[song_index]
            if rank >= len(ranking.trajectory_ids):
                next_resume = {"version": STATE_VERSION, "rank": rank, "song_index": song_index + 1, "completed": False}
                save_json(resume_path, next_resume)
                continue
            trajectory_id = ranking.trajectory_ids[rank]
            traj = read_trajectory(root[ranking.song_id], trajectory_id, arrays=["goals", "piano_states", "hand_joints", "hand_fingertips"])
            candidates = extract_press_candidates(traj, threshold=threshold)
            target_keys, hand_state, steps, _num_onsets, duplicates = accept_candidates(candidates, seen_goal_fp=seen_goal_fp, dedupe=dedupe)
            stat = stats.setdefault(ranking.song_id, {"candidates_seen": 0, "rows_accepted": 0, "goal_duplicates_skipped": 0})
            stat["candidates_seen"] += candidates.count
            stat["rows_accepted"] += int(target_keys.shape[0])
            stat["goal_duplicates_skipped"] += int(duplicates)
            if target_keys.shape[0]:
                append_song_npz(
                    out_root / f"song_{safe_stem(ranking.song_id)}.npz",
                    ranking.song_id,
                    target_keys,
                    hand_state,
                    np.full((target_keys.shape[0],), trajectory_id, dtype=np.int32),
                    steps,
                )
            save_seen(seen_path, seen_goal_fp)
            save_json(stats_path, stats)
            next_rank = rank
            next_song_index = song_index + 1
            if next_song_index >= len(rankings):
                next_rank = rank + 1
                next_song_index = 0
            save_json(resume_path, {"version": STATE_VERSION, "rank": next_rank, "song_index": next_song_index, "completed": False})
            print(
                f"rank={rank} song={song_index + 1}/{len(rankings)} {ranking.song_id} "
                f"traj={trajectory_id} candidates={candidates.count} accepted={target_keys.shape[0]} dup={duplicates}"
            )

    save_manifest(out_root, rankings, stats)
    summary = {
        "num_songs_requested": int(sampling.get("num_songs", 5)),
        "num_songs_kept": len(rankings),
        "num_songs_dropped": len(dropped),
        "score_sample_size": int(sampling.get("score_sample_size", 8)),
        "top_k_trajectories": top_k,
        "round_robin_trajectories": True,
        "deduplicate_unique_goal_keys": dedupe,
        "unique_goal_fingerprints": len(seen_goal_fp),
        "total_candidates_seen": int(sum(row["candidates_seen"] for row in stats.values())),
        "total_rows_accepted": int(sum(row["rows_accepted"] for row in stats.values())),
        "total_goal_duplicates_skipped": int(sum(row["goal_duplicates_skipped"] for row in stats.values())),
    }
    save_json(out_root / "summary.json", summary)
    save_json(state_dir / "goal_dedupe_stats.json", summary)
    save_json(resume_path, {"version": STATE_VERSION, "rank": top_k, "song_index": 0, "completed": True})
    print(f"Saved extraction outputs: {out_root}")


if __name__ == "__main__":
    main()

