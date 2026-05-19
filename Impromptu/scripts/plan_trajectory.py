#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (
    REPO_ROOT / "Impromptu" / "src",
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

from impromptu.config import ImpromptuConfig  # noqa: E402
from impromptu.planner import plan_target_keys  # noqa: E402
from intermezzo.io import atomic_save_json, atomic_save_npz, create_unique_run_dir  # noqa: E402
from intermezzo.keys import load_target_keys_npz  # noqa: E402
from intermezzo.midi import load_target_keys_from_midi  # noqa: E402


DEFAULT_OUTPUT_ROOT = Path("/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/runs")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan Impromptu fingertip-space Bagatelle-IK trajectories.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--midi-path", default=None)
    source.add_argument("--target-keys-npz", default=None)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--control-timestep", type=float, default=0.05)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--environment-name", default="RoboPianist-debug-TwinkleTwinkleLittleStar-v0")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-duration-s", type=float, default=None)
    parser.add_argument("--interpolation-substeps", type=int, default=4)
    parser.add_argument("--approach-s", type=float, default=0.08)
    parser.add_argument("--hold-s", type=float, default=0.02)
    parser.add_argument("--release-s", type=float, default=0.06)
    parser.add_argument("--clearance-height", type=float, default=0.02)
    parser.add_argument("--key-press-depth", type=float, default=0.008)
    parser.add_argument("--anchor-stride", type=int, default=2)
    parser.add_argument("--solve-contact-window-only", action="store_true", default=True)
    parser.add_argument("--solve-all-stride-anchors", action="store_true")
    parser.add_argument("--include-midpoint-anchors", action="store_true", default=True)
    parser.add_argument("--no-include-midpoint-anchors", action="store_true")
    parser.add_argument("--ik-fingertip-weight", type=float, default=1.0)
    parser.add_argument("--ik-smoothness-weight", type=float, default=0.05)
    parser.add_argument("--ik-neutral-weight", type=float, default=0.01)
    parser.add_argument("--ik-max-nfev", type=int, default=40)
    parser.add_argument("--residual-success-threshold", type=float, default=0.02)
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
    config = ImpromptuConfig(
        control_timestep=float(args.control_timestep),
        threshold=float(args.threshold),
        environment_name=str(args.environment_name),
        seed=int(args.seed),
        interpolation_substeps=int(args.interpolation_substeps),
        approach_s=float(args.approach_s),
        hold_s=float(args.hold_s),
        release_s=float(args.release_s),
        clearance_height=float(args.clearance_height),
        key_press_depth=float(args.key_press_depth),
        anchor_stride=int(args.anchor_stride),
        solve_contact_window_only=not bool(args.solve_all_stride_anchors),
        include_midpoint_anchors=not bool(args.no_include_midpoint_anchors),
        ik_fingertip_weight=float(args.ik_fingertip_weight),
        ik_smoothness_weight=float(args.ik_smoothness_weight),
        ik_neutral_weight=float(args.ik_neutral_weight),
        ik_max_nfev=int(args.ik_max_nfev),
        residual_success_threshold=float(args.residual_success_threshold),
        output_root=str(Path(args.output_root).expanduser()),
    )
    plan = plan_target_keys(target_keys, config=config)
    run_dir = create_unique_run_dir(Path(args.output_root).expanduser(), run_name=args.run_name, prefix="impromptu")
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
    print(f"Wrote Impromptu trajectory: {run_dir}")
    print(f"waypoints={plan.waypoint_frames.size} anchors={plan.ik_anchor_frames_dense.size} ik_success={metadata['ik_success_count']}")


if __name__ == "__main__":
    main()
