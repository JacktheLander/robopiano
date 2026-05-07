from __future__ import annotations

import os
import sys
from pathlib import Path
import subprocess

PARTITA_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PARTITA_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from partita.utils.config import experiment_name, load_config, output_root

WAVE_ZARR_PYTHON = Path("/WAVE/users2/unix/jlanders/.conda/envs/sonata/bin/python")


def choose_python() -> str:
    explicit = os.environ.get("PARTITA_PIPELINE_PYTHON") or os.environ.get("PARTITA_ZARR_PYTHON")
    if explicit:
        return explicit
    if WAVE_ZARR_PYTHON.exists():
        return str(WAVE_ZARR_PYTHON)
    return sys.executable


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the full Partita debug pipeline.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    runner = choose_python()
    scripts = [
        ["inspect_rp1m.py", "--rp1m-root", config["rp1m_root"], "--max-songs", "5"],
        ["select_song_trajectories.py", "--config", args.config],
        ["extract_song_trajectories.py", "--config", args.config],
        ["segment_trajectories.py", "--config", args.config],
        ["train_primitives.py", "--config", args.config],
        ["reconstruct_trajectory.py", "--config", args.config],
        ["evaluate_reconstruction.py", "--config", args.config],
    ]
    for cmd in scripts:
        full = [runner, str(PARTITA_ROOT / "scripts" / cmd[0]), *cmd[1:]]
        print("\n==>", " ".join(full), flush=True)
        subprocess.run(full, check=True)

    root = output_root(config)
    exp = experiment_name(config)
    paths = {
        "trajectory_scores": root / "data" / exp / "trajectory_scores.csv",
        "train_trajectory_ids": root / "data" / exp / "train_trajectory_ids.json",
        "reconstruction_target": root / "data" / exp / "reconstruction_target.json",
        "segments": root / "primitives" / exp / "segments.csv",
        "primitive_summary": root / "primitives" / exp / "primitive_summary.csv",
        "primitive_usage_by_trajectory": root / "primitives" / exp / "primitive_usage_by_trajectory.csv",
        "metrics": root / "evaluation" / exp / "metrics.json",
        "pianoroll_comparison": root / "evaluation" / exp / "pianoroll_comparison.png",
    }
    print("\nPartita debug pipeline complete.")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
