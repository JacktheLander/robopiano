#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

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
from bagatelle.planner import plan_target_keys  # noqa: E402
from intermezzo.io import atomic_save_json, atomic_save_npz, create_unique_run_dir  # noqa: E402
from intermezzo.keys import load_target_keys_npz  # noqa: E402
from intermezzo.midi import load_target_keys_from_midi  # noqa: E402


DEFAULT_OUTPUT_ROOT = Path("/WAVE/datasets/ccoelho_lab-jlanders/Bagatelle/runs")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan Bagatelle OT+IK hand-state trajectories from MIDI or target_keys.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--midi-path", default=None, help="Path to a .mid/.midi file to quantize into target_keys[T, 88].")
    source.add_argument("--target-keys-npz", default=None, help="NPZ containing `target_keys` or a first array shaped [T, 88+].")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Parent directory for unique Bagatelle run dirs.")
    parser.add_argument("--run-name", default=None, help="Optional filesystem-safe run name. A numeric suffix is added on collision.")
    parser.add_argument("--control-timestep", type=float, default=0.05)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--environment-name", default="RoboPianist-debug-TwinkleTwinkleLittleStar-v0")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional MIDI quantization step cap.")
    parser.add_argument("--max-duration-s", type=float, default=None, help="Optional MIDI quantization duration cap.")
    parser.add_argument("--ik-fingertip-weight", type=float, default=1.0)
    parser.add_argument("--ik-smoothness-weight", type=float, default=0.05)
    parser.add_argument("--ik-neutral-weight", type=float, default=0.01)
    parser.add_argument("--ik-max-nfev", type=int, default=120)
    parser.add_argument("--residual-success-threshold", type=float, default=0.02)
    parser.add_argument("--key-press-depth", type=float, default=0.008)
    parser.add_argument("--clearance-height", type=float, default=0.02)
    return parser


def _load_target_keys(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    if args.midi_path:
        target_keys, meta = load_target_keys_from_midi(
            args.midi_path,
            control_timestep=float(args.control_timestep),
            max_steps=args.max_steps,
            max_duration_s=args.max_duration_s,
        )
        return target_keys, {
            "source_type": "midi",
            "midi_path": str(Path(args.midi_path).expanduser().resolve()),
            "midi": meta,
        }
    target_keys = load_target_keys_npz(args.target_keys_npz)
    return target_keys, {
        "source_type": "target_keys_npz",
        "target_keys_npz": str(Path(args.target_keys_npz).expanduser().resolve()),
    }


def main() -> None:
    args = build_parser().parse_args()
    target_keys, source_meta = _load_target_keys(args)
    config = BagatelleConfig(
        control_timestep=float(args.control_timestep),
        threshold=float(args.threshold),
        seed=int(args.seed),
        environment_name=str(args.environment_name),
        ik_fingertip_weight=float(args.ik_fingertip_weight),
        ik_smoothness_weight=float(args.ik_smoothness_weight),
        ik_neutral_weight=float(args.ik_neutral_weight),
        ik_max_nfev=int(args.ik_max_nfev),
        residual_success_threshold=float(args.residual_success_threshold),
        key_press_depth=float(args.key_press_depth),
        clearance_height=float(args.clearance_height),
    )
    plan = plan_target_keys(target_keys, config=config)
    run_dir = create_unique_run_dir(Path(args.output_root).expanduser(), run_name=args.run_name, prefix="bagatelle")
    trajectory_path = run_dir / "trajectory.npz"
    metadata_path = run_dir / "metadata.json"
    atomic_save_npz(trajectory_path, **plan.npz_payload())
    metadata = {
        **source_meta,
        "run_dir": str(run_dir),
        "trajectory_npz": str(trajectory_path),
        "seed": int(args.seed),
        **plan.metadata,
    }
    atomic_save_json(metadata_path, metadata)
    print(f"Wrote Bagatelle trajectory: {run_dir}")
    print(
        f"waypoints={plan.waypoint_frames.size} "
        f"ik_success={metadata['ik_success_count']}/{plan.waypoint_frames.size} "
        f"unassigned_keys={metadata['ik_unassigned_key_count']}"
    )


if __name__ == "__main__":
    main()
