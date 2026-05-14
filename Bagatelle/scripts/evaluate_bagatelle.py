#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (
    REPO_ROOT / "Bagatelle" / "src",
    REPO_ROOT / "Intermezzo" / "src",
    REPO_ROOT / "Variations" / "src",
    REPO_ROOT / "Variations",
    REPO_ROOT / "partita" / "src",
    REPO_ROOT,
):
    _text = str(_path)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from bagatelle.config import BagatelleConfig  # noqa: E402
from bagatelle.evaluation import DEFAULT_OUTPUT_ROOT, evaluate_bagatelle_trajectory  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a Bagatelle trajectory with headless RoboPianist pose injection.")
    parser.add_argument("--trajectory-npz", required=True, help="Path to Bagatelle trajectory.npz.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Parent directory for unique evaluation run dirs.")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--label", default="bagatelle")
    parser.add_argument("--control-timestep", type=float, default=0.05)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--environment-name", default="RoboPianist-debug-TwinkleTwinkleLittleStar-v0")
    parser.add_argument("--timing-tolerance-s", type=float, default=0.15)
    parser.add_argument("--residual-success-threshold", type=float, default=0.02)
    parser.add_argument("--settle-steps", type=int, default=3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = BagatelleConfig(
        control_timestep=float(args.control_timestep),
        threshold=float(args.threshold),
        seed=int(args.seed),
        environment_name=str(args.environment_name),
        residual_success_threshold=float(args.residual_success_threshold),
        settle_steps=int(args.settle_steps),
    )
    summary = evaluate_bagatelle_trajectory(
        args.trajectory_npz,
        output_root=args.output_root,
        run_name=args.run_name,
        label=str(args.label),
        config=config,
        timing_tolerance_s=float(args.timing_tolerance_s),
    )
    score = summary["rollout"]["score"]
    print(f"Wrote Bagatelle evaluation: {summary['run_dir']}")
    print(
        f"missed={score['missed_key_presses']} "
        f"mispresses={score['mispresses']} "
        f"matched={score['matched_press_events']}/{score['target_press_events']} "
        f"fingertip_success={summary['fingertips']['fingertip_success_rate']}"
    )


if __name__ == "__main__":
    main()
