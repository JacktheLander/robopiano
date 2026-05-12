#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

INTERMEZZO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = INTERMEZZO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from intermezzo.io import atomic_save_json, atomic_save_npz, create_unique_run_dir  # noqa: E402
from intermezzo.keys import load_target_keys_npz  # noqa: E402
from intermezzo.midi import load_target_keys_from_midi  # noqa: E402
from intermezzo.planner import PlannerConfig, build_intermezzo_trajectory  # noqa: E402
from intermezzo.variations_bridge import load_variations_diffusion_predictor  # noqa: E402


DEFAULT_OUTPUT_ROOT = Path("/WAVE/datasets/ccoelho_lab-jlanders/Intermezzo/runs")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan Intermezzo hand-state trajectories from MIDI or target_keys using a Variations diffusion checkpoint.",
    )
    parser.add_argument("--checkpoint", required=True, help="Variations diffusion checkpoint path, typically checkpoints/best.pt.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--midi-path", default=None, help="Path to a .mid/.midi file to quantize into target_keys[T, 88].")
    source.add_argument("--target-keys-npz", default=None, help="NPZ containing `target_keys` or a first array shaped [T, 88+].")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Parent directory for unique Intermezzo run dirs.")
    parser.add_argument("--run-name", default=None, help="Optional filesystem-safe run name prefix. A numeric suffix is added on collision.")
    parser.add_argument("--control-timestep", type=float, default=0.05)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--diffusion-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser


def _set_seed(seed: int) -> None:
    import random

    import numpy as np

    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import torch
    except Exception:
        return
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _load_target_keys(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    if args.midi_path:
        target_keys, meta = load_target_keys_from_midi(
            args.midi_path,
            control_timestep=float(args.control_timestep),
        )
        return target_keys, {"source_type": "midi", "midi_path": str(Path(args.midi_path).expanduser().resolve()), "midi": meta}
    target_keys = load_target_keys_npz(args.target_keys_npz)
    return target_keys, {
        "source_type": "target_keys_npz",
        "target_keys_npz": str(Path(args.target_keys_npz).expanduser().resolve()),
    }


def main() -> None:
    args = build_parser().parse_args()
    _set_seed(int(args.seed))

    target_keys, source_meta = _load_target_keys(args)
    predictor = load_variations_diffusion_predictor(
        args.checkpoint,
        device=str(args.device),
        diffusion_steps=args.diffusion_steps,
    )
    config = PlannerConfig(
        control_timestep=float(args.control_timestep),
        threshold=float(args.threshold),
    )
    plan = build_intermezzo_trajectory(
        target_keys,
        predictor=lambda keys: predictor.predict(keys, batch_size=int(args.batch_size)),
        config=config,
        batch_size=int(args.batch_size),
    )

    run_dir = create_unique_run_dir(Path(args.output_root).expanduser(), run_name=args.run_name)
    trajectory_path = run_dir / "trajectory.npz"
    metadata_path = run_dir / "metadata.json"

    atomic_save_npz(
        trajectory_path,
        target_keys=plan.target_keys,
        waypoint_frames=plan.waypoint_frames,
        waypoint_target_keys=plan.waypoint_target_keys,
        waypoint_hand_joints=plan.waypoint_hand_joints,
        planned_hand_joints=plan.planned_hand_joints,
        planned_hand_velocities=plan.planned_hand_velocities,
        segment_ids=plan.segment_ids,
    )

    metadata: dict[str, Any] = {
        **source_meta,
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "run_dir": str(run_dir),
        "trajectory_npz": str(trajectory_path),
        "seed": int(args.seed),
        "device": str(predictor.device),
        "diffusion_steps": int(predictor.diffusion_steps),
        "batch_size": int(args.batch_size),
        **plan.metadata,
    }
    atomic_save_json(metadata_path, metadata)
    print(f"Wrote Intermezzo trajectory: {run_dir}")


if __name__ == "__main__":
    main()
