#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np

INTERMEZZO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = INTERMEZZO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from intermezzo.evaluation import compute_two_state_metrics, save_metrics  # noqa: E402
from intermezzo.io import atomic_save_json, atomic_save_npz, create_unique_run_dir  # noqa: E402
from intermezzo.key_geometry import load_key_geometry  # noqa: E402
from intermezzo.keys import load_target_keys_npz  # noqa: E402
from intermezzo.planner import PlannerConfig, build_two_state_trajectory  # noqa: E402
from intermezzo.variations_bridge import load_variations_diffusion_predictor  # noqa: E402


DEFAULT_OUTPUT_ROOT = Path("/WAVE/datasets/ccoelho_lab-jlanders/Intermezzo/two_state_runs")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan a two-state Intermezzo hand trajectory between two target keysets.")
    parser.add_argument("--checkpoint", required=True, help="Variations diffusion checkpoint path.")
    parser.add_argument("--keyset-a", default=None, help="Comma-separated key indices for endpoint A.")
    parser.add_argument("--keyset-b", default=None, help="Comma-separated key indices for endpoint B.")
    parser.add_argument("--target-keys-npz", default=None, help="NPZ containing target_keys for selecting frame A/B keysets.")
    parser.add_argument("--frame-a", type=int, default=None)
    parser.add_argument("--frame-b", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=40)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--diffusion-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enable-key-magnetism", action="store_true")
    parser.add_argument("--render", action="store_true", help="Render the planned rollout after writing the trajectory bundle.")
    return parser


def _set_seed(seed: int) -> None:
    import random

    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import torch
    except Exception:
        return
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _parse_keyset(text: str | None, *, name: str) -> np.ndarray:
    out = np.zeros((88,), dtype=np.float32)
    if text is None or not str(text).strip():
        return out
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        key = int(token)
        if key < 0 or key >= 88:
            raise ValueError(f"{name} contains key index {key}; valid range is [0, 87]")
        out[key] = 1.0
    return out


def _load_keysets(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if args.target_keys_npz:
        if args.frame_a is None or args.frame_b is None:
            raise ValueError("--target-keys-npz requires both --frame-a and --frame-b")
        keys = load_target_keys_npz(args.target_keys_npz)
        for frame_name, frame in (("frame_a", args.frame_a), ("frame_b", args.frame_b)):
            if int(frame) < 0 or int(frame) >= keys.shape[0]:
                raise ValueError(f"--{frame_name.replace('_', '-')}={frame} outside target_keys length {keys.shape[0]}")
        return (
            keys[int(args.frame_a)].astype(np.float32),
            keys[int(args.frame_b)].astype(np.float32),
            {
                "source_type": "target_keys_npz",
                "target_keys_npz": str(Path(args.target_keys_npz).expanduser().resolve()),
                "frame_a": int(args.frame_a),
                "frame_b": int(args.frame_b),
            },
        )
    if args.keyset_a is None or args.keyset_b is None:
        raise ValueError("Provide either --target-keys-npz with frames, or both --keyset-a and --keyset-b")
    return (
        _parse_keyset(args.keyset_a, name="keyset_a"),
        _parse_keyset(args.keyset_b, name="keyset_b"),
        {"source_type": "literal_keysets", "keyset_a": args.keyset_a, "keyset_b": args.keyset_b},
    )


def main() -> None:
    args = build_parser().parse_args()
    _set_seed(int(args.seed))
    keyset_a, keyset_b, source_meta = _load_keysets(args)

    predictor = load_variations_diffusion_predictor(
        args.checkpoint,
        device=str(args.device),
        diffusion_steps=args.diffusion_steps,
    )
    endpoint_hand_joints = predictor.predict(np.stack([keyset_a, keyset_b], axis=0), batch_size=int(args.batch_size))
    key_xy = load_key_geometry(allow_approximate=True)
    config = PlannerConfig(enable_key_magnetism=bool(args.enable_key_magnetism))
    plan = build_two_state_trajectory(
        keyset_a,
        keyset_b,
        endpoint_hand_joints=np.asarray(endpoint_hand_joints, dtype=np.float32),
        num_steps=int(args.num_steps),
        config=config,
        key_geometry=key_xy,
    )

    run_dir = create_unique_run_dir(Path(args.output_root).expanduser(), prefix="intermezzo_two_state")
    trajectory_path = run_dir / "two_state_trajectory.npz"
    metadata_path = run_dir / "metadata.json"
    metrics_path = run_dir / "rollout_metrics.json"

    atomic_save_npz(
        trajectory_path,
        target_keys=plan.target_keys,
        keyset_a=keyset_a,
        keyset_b=keyset_b,
        waypoint_frames=plan.waypoint_frames,
        waypoint_target_keys=plan.waypoint_target_keys,
        endpoint_hand_joints=np.asarray(endpoint_hand_joints, dtype=np.float32),
        waypoint_hand_joints=plan.waypoint_hand_joints,
        planned_hand_joints=plan.planned_hand_joints,
        planned_hand_velocities=plan.planned_hand_velocities,
        segment_ids=plan.segment_ids,
        key_geometry=key_xy,
    )
    metrics = compute_two_state_metrics(
        planned_hand_joints=plan.planned_hand_joints,
        endpoint_hand_joints=np.asarray(endpoint_hand_joints, dtype=np.float32),
        final_target_keys=keyset_b,
        key_xy=key_xy,
        control_timestep=float(config.control_timestep),
        threshold=float(config.threshold),
    )
    save_metrics(metrics_path, metrics)

    metadata: dict[str, Any] = {
        **source_meta,
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "run_dir": str(run_dir),
        "trajectory_npz": str(trajectory_path),
        "rollout_metrics_json": str(metrics_path),
        "seed": int(args.seed),
        "device": str(predictor.device),
        "diffusion_steps": int(predictor.diffusion_steps),
        "batch_size": int(args.batch_size),
        "num_steps": int(args.num_steps),
        **plan.metadata,
    }
    atomic_save_json(metadata_path, metadata)

    if args.render:
        from render_two_state_rollout import render_two_state_rollout

        render_two_state_rollout(trajectory_path=trajectory_path, output_dir=run_dir, control_timestep=float(config.control_timestep), seed=int(args.seed))
    print(f"Wrote Intermezzo two-state trajectory: {run_dir}")


if __name__ == "__main__":
    main()
