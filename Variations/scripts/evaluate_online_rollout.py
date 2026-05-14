#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys


VARIATIONS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = VARIATIONS_ROOT.parent
for path in (
    VARIATIONS_ROOT / "src",
    VARIATIONS_ROOT,
    REPO_ROOT / "Intermezzo" / "src",
    REPO_ROOT / "partita" / "src",
    REPO_ROOT,
):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from variations.online_eval import (  # noqa: E402
    DEFAULT_MAESTRO_ROOT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_VARIATIONS_CHECKPOINT_ROOT,
    evaluate_variations_models_online,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Headless online RoboPianist rollout comparison across Variations "
            "MLP baseline, latent MDN, and diffusion models."
        )
    )
    parser.add_argument("--midi-path", default=None, help="Explicit MAESTRO .mid/.midi file.")
    parser.add_argument("--maestro-root", default=str(DEFAULT_MAESTRO_ROOT))
    parser.add_argument("--midi-selection", choices=["shortest", "index"], default="shortest")
    parser.add_argument("--piece-index", type=int, default=0)
    parser.add_argument("--mlp-checkpoint", default=None)
    parser.add_argument("--latent-mdn-checkpoint", default=None)
    parser.add_argument("--diffusion-checkpoint", default=None)
    parser.add_argument("--checkpoint-search-root", default=str(DEFAULT_VARIATIONS_CHECKPOINT_ROOT))
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
    summary = evaluate_variations_models_online(
        midi_path=args.midi_path,
        maestro_root=args.maestro_root,
        midi_selection=args.midi_selection,
        piece_index=int(args.piece_index),
        mlp_checkpoint=args.mlp_checkpoint,
        latent_mdn_checkpoint=args.latent_mdn_checkpoint,
        diffusion_checkpoint=args.diffusion_checkpoint,
        checkpoint_search_root=args.checkpoint_search_root,
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
    print(f"Wrote Variations online evaluation: {summary['run_dir']}")
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
