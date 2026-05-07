from __future__ import annotations

import sys
from pathlib import Path

PARTITA_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PARTITA_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse
import numpy as np

from partita.data.rp1m_loader import available_arrays, open_rp1m_root, read_trajectories, read_trajectory
from partita.utils.config import experiment_name, load_config, output_root
from partita.utils.io import ensure_dir, load_json, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract selected RP1M trajectories into compact NPZ files.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    data_dir = ensure_dir(output_root(config) / "data" / experiment_name(config))
    selection = load_json(data_dir / "selection.json")
    train_ids = [int(x) for x in load_json(data_dir / "train_trajectory_ids.json")]
    target = load_json(data_dir / "reconstruction_target.json")
    song_name = selection["song_name"]

    root = open_rp1m_root(selection["rp1m_root"])
    song_group = root[song_name]
    arrays = [a for a in available_arrays(song_group) if a in {"actions", "goals", "piano_states", "hand_joints", "hand_fingertips"}]
    if "actions" not in arrays:
        raise RuntimeError(f"Song {song_name} does not contain required actions array.")

    selected = read_trajectories(song_group, train_ids, arrays=arrays)
    target_data = read_trajectory(song_group, int(target["trajectory_id"]), arrays=arrays)
    np.savez_compressed(data_dir / "selected_trajectories.npz", **selected)
    np.savez_compressed(data_dir / "target_trajectory.npz", **target_data)
    summary = {
        "song_name": song_name,
        "available_arrays": arrays,
        "num_training_trajectories": len(train_ids),
        "train_trajectory_ids": train_ids,
        "target_trajectory_id": int(target["trajectory_id"]),
        "selected_shapes": {k: list(v.shape) for k, v in selected.items() if hasattr(v, "shape")},
        "target_shapes": {k: list(v.shape) for k, v in target_data.items() if hasattr(v, "shape")},
    }
    save_json(data_dir / "song_summary.json", summary)
    print(f"Saved selected and target trajectory NPZ files to {data_dir}")


if __name__ == "__main__":
    main()
