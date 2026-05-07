from __future__ import annotations

import sys
from pathlib import Path

PARTITA_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PARTITA_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse
import numpy as np

from partita.data.rp1m_loader import open_rp1m_root, trajectory_count
from partita.data.song_selector import choose_best_song, score_song_trajectories
from partita.utils.config import experiment_name, load_config, output_root
from partita.utils.io import ensure_dir, save_csv, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Select one RP1M song and successful trajectories for Partita.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    sel_cfg = config.get("selection", {})
    threshold = float(sel_cfg.get("key_threshold", 0.5))
    root = open_rp1m_root(config["rp1m_root"])

    selection_notes = []
    if config.get("song_name"):
        song_name = str(config["song_name"])
        if song_name not in root:
            raise RuntimeError(f"Configured song_name not found in RP1M root: {song_name}")
        selection_meta = {"selection_mode": "configured", "chosen": {"song_name": song_name}}
    else:
        song_name, selection_meta = choose_best_song(
            root,
            list(config.get("song_search_terms", [])),
            threshold=threshold,
            fallback_scan_songs=int(sel_cfg.get("fallback_scan_songs", 50)),
        )
    print(f"Selected song: {song_name}")

    try:
        scores = score_song_trajectories(root[song_name], song_name, threshold=threshold)
    except Exception as exc:
        n = trajectory_count(root[song_name])
        selection_notes.append(f"Metric scoring failed ({exc}); falling back to rank by trajectory_id.")
        scores = []
        for i in range(n):
            scores.append({
                "song_name": song_name,
                "trajectory_id": i,
                "rank": i,
                "key_precision": np.nan,
                "key_recall": np.nan,
                "key_f1": np.nan,
                "mispress_rate": np.nan,
                "action_smoothness": np.nan,
                "score": -float(i),
            })
        import pandas as pd
        scores = pd.DataFrame(scores)

    scores = scores.sort_values("score", ascending=False).reset_index(drop=True)
    scores["rank"] = np.arange(len(scores), dtype=int)
    reconstruction_rank = int(sel_cfg.get("reconstruction_rank", config.get("reconstruction", {}).get("target_trajectory_rank", 0)))
    reconstruction_rank = min(max(reconstruction_rank, 0), len(scores) - 1)
    target_id = int(scores.loc[reconstruction_rank, "trajectory_id"])

    filtered = scores.copy()
    min_f1 = sel_cfg.get("min_key_f1")
    max_mispress = sel_cfg.get("max_mispress_rate")
    if min_f1 is not None and "key_f1" in filtered:
        keep = filtered["key_f1"].isna() | (filtered["key_f1"] >= float(min_f1))
        filtered = filtered[keep]
    if max_mispress is not None and "mispress_rate" in filtered:
        keep = filtered["mispress_rate"].isna() | (filtered["mispress_rate"] <= float(max_mispress))
        filtered = filtered[keep]

    if bool(sel_cfg.get("exclude_reconstruction_from_training", True)):
        filtered = filtered[filtered["trajectory_id"] != target_id]

    top_k = int(sel_cfg.get("top_k_train", 64))
    if len(filtered) < top_k:
        selection_notes.append(f"Only {len(filtered)} trajectories passed filters; using best available.")
    train_ids = [int(x) for x in filtered.head(top_k)["trajectory_id"].tolist()]
    if not train_ids:
        fallback = scores[scores["trajectory_id"] != target_id].head(top_k)
        train_ids = [int(x) for x in fallback["trajectory_id"].tolist()]
        selection_notes.append("No trajectories passed filters; using best non-target trajectories.")
    if not train_ids and len(scores):
        train_ids = [target_id]
        selection_notes.append("Only target trajectory available; including target in training.")

    scores["selected_for_training"] = scores["trajectory_id"].isin(train_ids)
    scores["selected_as_target"] = scores["trajectory_id"] == target_id

    out_dir = ensure_dir(output_root(config) / "data" / experiment_name(config))
    save_csv(out_dir / "trajectory_scores.csv", scores)
    selection = {
        "song_name": song_name,
        "rp1m_root": config["rp1m_root"],
        "experiment_name": experiment_name(config),
        "selection_meta": selection_meta,
        "selection_notes": selection_notes,
        "top_k_train_requested": top_k,
        "num_training_trajectories": len(train_ids),
        "train_trajectory_ids": train_ids,
        "target_trajectory_id": target_id,
        "target_rank": reconstruction_rank,
        "key_threshold": threshold,
    }
    save_json(out_dir / "selection.json", selection)
    save_json(out_dir / "train_trajectory_ids.json", train_ids)
    save_json(out_dir / "reconstruction_target.json", {"song_name": song_name, "trajectory_id": target_id, "rank": reconstruction_rank})
    print(f"Training trajectories: {len(train_ids)}")
    print(f"Target trajectory: {target_id} (rank {reconstruction_rank})")
    print(f"Saved selection outputs: {out_dir}")


if __name__ == "__main__":
    main()
