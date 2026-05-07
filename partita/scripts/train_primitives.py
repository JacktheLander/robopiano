from __future__ import annotations

import sys
from pathlib import Path

PARTITA_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PARTITA_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse
import numpy as np
import pandas as pd

from partita.primitives.clustering import fit_kmeans_features
from partita.primitives.features import features_for_segments
from partita.primitives.library import build_primitive_library, primitive_summary, primitive_usage_by_trajectory, save_library
from partita.utils.config import experiment_name, load_config, output_root
from partita.utils.io import ensure_dir, save_csv
from partita.utils.plotting import save_primitive_timeline, save_primitive_usage


def main() -> None:
    parser = argparse.ArgumentParser(description="Train same-song Partita primitive library.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    exp = experiment_name(config)
    data_dir = output_root(config) / "data" / exp
    prim_dir = ensure_dir(output_root(config) / "primitives" / exp)
    data_npz = np.load(data_dir / "selected_trajectories.npz")
    data = {k: data_npz[k] for k in data_npz.files}
    segments = pd.read_csv(prim_dir / "segments.csv")
    cfg = config.get("primitives", {})
    features, feature_names = features_for_segments(
        data,
        segments,
        include_relative_song_time=bool(cfg.get("include_relative_song_time", True)),
        relative_song_time_weight=float(cfg.get("relative_song_time_weight", 0.2)),
    )
    scaler, pca, clusterer, labels, transformed = fit_kmeans_features(
        features,
        num_primitives=int(cfg.get("num_primitives", 32)),
        pca_dim=int(cfg.get("pca_dim", 24)),
        random_seed=int(cfg.get("random_seed", 42)),
    )
    assignments = segments[["global_segment_id", "trajectory_id", "trajectory_index", "local_segment_id", "start_t", "end_t", "duration"]].copy()
    assignments["primitive_id"] = labels.astype(int)
    assignments = assignments[["global_segment_id", "trajectory_id", "trajectory_index", "local_segment_id", "primitive_id", "start_t", "end_t", "duration"]]
    save_csv(prim_dir / "primitive_assignments.csv", assignments)
    summary = primitive_summary(segments, assignments, num_training_trajectories=int(data["actions"].shape[0]))
    save_csv(prim_dir / "primitive_summary.csv", summary)
    usage = primitive_usage_by_trajectory(assignments)
    save_csv(prim_dir / "primitive_usage_by_trajectory.csv", usage)
    library = build_primitive_library(
        data,
        segments,
        assignments,
        transformed,
        scaler,
        pca,
        clusterer,
        feature_names,
        resample_len=int(cfg.get("resample_len", 16)),
        config=config,
    )
    save_library(prim_dir / "primitive_library.pkl", library)
    save_primitive_usage(summary, prim_dir / "primitive_usage.png")
    save_primitive_timeline(assignments, prim_dir / "primitive_timeline_by_trajectory.png")
    print(f"Saved primitive library with {len(summary)} primitives to {prim_dir}")


if __name__ == "__main__":
    main()
