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

from partita.primitives.features import feature_for_target_segment, resample_array
from partita.primitives.library import load_library
from partita.reconstruction.nearest import assign_nearest_primitives
from partita.segmentation.segmenter import segment_single_trajectory
from partita.utils.config import experiment_name, load_config, output_root
from partita.utils.io import ensure_dir, save_csv
from partita.utils.plotting import save_action_comparison


def _target_npz_to_dict(npz) -> dict:
    return {k: npz[k] for k in npz.files}


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct target trajectory from Partita primitives.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    exp = experiment_name(config)
    data_dir = output_root(config) / "data" / exp
    prim_dir = output_root(config) / "primitives" / exp
    recon_dir = ensure_dir(output_root(config) / "reconstruction" / exp)
    target = _target_npz_to_dict(np.load(data_dir / "target_trajectory.npz"))
    library = load_library(prim_dir / "primitive_library.pkl")
    threshold = float(config.get("selection", {}).get("key_threshold", 0.5))
    segments = segment_single_trajectory(target, config.get("segmentation", {}), key_threshold=threshold)
    save_csv(recon_dir / "target_segments.csv", segments)

    prim_cfg = config.get("primitives", {})
    feature_vectors = []
    for _, row in segments.iterrows():
        feature_vectors.append(feature_for_target_segment(
            target,
            row,
            library["feature_names"],
            include_relative_song_time=bool(prim_cfg.get("include_relative_song_time", True)),
            relative_song_time_weight=float(prim_cfg.get("relative_song_time_weight", 0.2)),
        ))
    features = np.stack(feature_vectors, axis=0)
    labels = assign_nearest_primitives(features, library)
    assignments = segments[["global_segment_id", "trajectory_id", "trajectory_index", "local_segment_id", "start_t", "end_t", "duration"]].copy()
    assignments["primitive_id"] = labels.astype(int)
    assignments = assignments[["global_segment_id", "trajectory_id", "trajectory_index", "local_segment_id", "primitive_id", "start_t", "end_t", "duration"]]
    save_csv(recon_dir / "target_primitive_assignments.csv", assignments)

    actions = np.asarray(target["actions"], dtype=np.float32)
    recon_actions = np.zeros_like(actions, dtype=np.float32)
    recon_piano = None
    if "piano_states" in target:
        recon_piano = np.zeros_like(np.asarray(target["piano_states"], dtype=np.float32), dtype=np.float32)
    recon_hand_joints = None
    if "hand_joints" in target:
        recon_hand_joints = np.zeros_like(np.asarray(target["hand_joints"], dtype=np.float32), dtype=np.float32)
    timeline_rows = []
    for _, row in assignments.iterrows():
        start = int(row["start_t"])
        end = int(row["end_t"])
        duration = max(end - start, 1)
        pid = int(row["primitive_id"])
        primitive = library["primitives"][pid]
        recon_actions[start:end] = resample_array(primitive["mean_action_trajectory"], duration)[:, : actions.shape[-1]]
        if recon_piano is not None and primitive.get("mean_piano_state_profile") is not None:
            piano = resample_array(primitive["mean_piano_state_profile"], duration)
            recon_piano[start:end] = piano[:, : recon_piano.shape[-1]]
        if recon_hand_joints is not None and primitive.get("mean_hand_joint_profile") is not None:
            hand_joints = resample_array(primitive["mean_hand_joint_profile"], duration)
            recon_hand_joints[start:end] = hand_joints[:, : recon_hand_joints.shape[-1]]
        timeline_rows.append({"start_t": start, "end_t": end, "duration": duration, "primitive_id": pid})

    np.save(recon_dir / "reconstructed_actions.npy", recon_actions)
    np.save(recon_dir / "original_actions.npy", actions)
    if recon_piano is not None:
        np.save(recon_dir / "reconstructed_piano_states.npy", recon_piano)
        np.save(recon_dir / "original_piano_states.npy", np.asarray(target["piano_states"], dtype=np.float32))
    if recon_hand_joints is not None:
        np.save(recon_dir / "reconstructed_hand_joints.npy", recon_hand_joints)
        np.save(recon_dir / "original_hand_joints.npy", np.asarray(target["hand_joints"], dtype=np.float32))
    save_csv(recon_dir / "primitive_timeline.csv", pd.DataFrame(timeline_rows))
    save_action_comparison(actions, recon_actions, recon_dir / "original_vs_reconstructed_actions.png")
    print(f"Saved reconstruction outputs to {recon_dir}")


if __name__ == "__main__":
    main()
