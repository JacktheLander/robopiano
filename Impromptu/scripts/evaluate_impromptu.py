#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "Impromptu" / "src"
INTERMEZZO_SRC = REPO_ROOT / "Intermezzo" / "src"
for _path in (SRC_ROOT, INTERMEZZO_SRC):
    _text = str(_path)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from impromptu.evaluation import evaluate_trajectory_npz  # noqa: E402
from intermezzo.io import atomic_save_json, create_unique_run_dir  # noqa: E402


DEFAULT_EVALUATION_ROOT = Path("/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/evaluation")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate an Impromptu trajectory NPZ.")
    parser.add_argument("--trajectory-npz", required=True)
    parser.add_argument("--output-root", default=str(DEFAULT_EVALUATION_ROOT))
    parser.add_argument("--run-name", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics = evaluate_trajectory_npz(args.trajectory_npz)
    run_dir = create_unique_run_dir(Path(args.output_root).expanduser(), run_name=args.run_name, prefix="impromptu_eval")
    metrics_path = run_dir / "metrics.json"
    atomic_save_json(
        metrics_path,
        {
            "trajectory_npz": str(Path(args.trajectory_npz).expanduser().resolve()),
            "run_dir": str(run_dir),
            "metrics": metrics,
        },
    )
    print(f"Wrote Impromptu evaluation: {run_dir}")
    print(
        "ik_anchor_fingertip_distance_p95="
        f"{metrics['ik_anchor_fingertip_distance_p95']:.6f} "
        f"max_joint_velocity={metrics['max_joint_velocity']:.6f}"
    )


if __name__ == "__main__":
    main()
