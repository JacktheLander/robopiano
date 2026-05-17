#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

INTERMEZZO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = INTERMEZZO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from intermezzo.online_eval import (  # noqa: E402
    DEFAULT_MAESTRO_ROOT,
    DEFAULT_OUTPUT_ROOT,
    REQUESTED_FULL_DIFFUSION_CHECKPOINT,
    evaluate_variations_vs_intermezzo,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Headless online RoboPianist rollout comparison for Variations direct poses and Intermezzo planned trajectories.",
    )
    parser.add_argument("--midi-path", default=None, help="Explicit MAESTRO .mid/.midi file. If omitted, use --maestro-root selection.")
    parser.add_argument("--maestro-root", default=str(DEFAULT_MAESTRO_ROOT))
    parser.add_argument("--midi-selection", choices=["shortest", "index"], default="shortest")
    parser.add_argument("--piece-index", type=int, default=0)
    parser.add_argument("--checkpoint", default=str(REQUESTED_FULL_DIFFUSION_CHECKPOINT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--control-timestep", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-duration-s", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--diffusion-steps", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--timing-tolerance-s", type=float, default=0.15)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = evaluate_variations_vs_intermezzo(
        midi_path=args.midi_path,
        maestro_root=args.maestro_root,
        midi_selection=args.midi_selection,
        piece_index=int(args.piece_index),
        checkpoint=args.checkpoint,
        output_root=args.output_root,
        control_timestep=float(args.control_timestep),
        max_steps=args.max_steps,
        max_duration_s=args.max_duration_s,
        batch_size=int(args.batch_size),
        diffusion_steps=args.diffusion_steps,
        device=str(args.device),
        seed=int(args.seed),
        threshold=float(args.threshold),
        timing_tolerance_s=float(args.timing_tolerance_s),
    )
    print(f"Wrote Intermezzo online evaluation: {summary['run_dir']}")
    for model_name, model in summary["models"].items():
        score = model["score"]
        print(
            f"{model_name}: missed={score['missed_key_presses']} "
            f"mispresses={score['mispresses']} "
            f"matched={score['matched_press_events']}/{score['target_press_events']} "
            f"timing_mean_abs_s={score['timing_abs_error_mean_s']}"
        )


if __name__ == "__main__":
    main()

