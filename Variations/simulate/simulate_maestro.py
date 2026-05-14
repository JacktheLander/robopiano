from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
VARIATIONS_ROOT = REPO_ROOT / "Variations"
for path in (
    VARIATIONS_ROOT / "src",
    VARIATIONS_ROOT,
    REPO_ROOT / "partita" / "src",
    REPO_ROOT,
):
    s = str(path)
    if s not in sys.path:
        sys.path.insert(0, s)

from simulate.midi_keysets import (  # noqa: E402
    discover_midi_files,
    midi_to_target_key_roll,
    piece_id_from_path,
    simulation_slug,
)
from simulate.model_loader import load_simulation_model  # noqa: E402
from simulate.render import rollout_variations_maestro_prediction, write_simulation_bundle  # noqa: E402

DEFAULT_OUTPUT_ROOT = Path("/WAVE/datasets/ccoelho_lab-jlanders/Variations/simulation")


def _resolve_midi_path(args: argparse.Namespace) -> tuple[Path, str]:
    if args.midi_path:
        p = Path(args.midi_path).expanduser().resolve()
        slug = simulation_slug(p.stem)
        return p, slug
    if args.maestro_root:
        root = Path(args.maestro_root).expanduser().resolve()
        files = discover_midi_files(root)
        if args.limit_songs is not None:
            files = files[: int(args.limit_songs)]
        if not files:
            raise RuntimeError(f"No MIDI files found under {root}")
        idx = int(args.piece_index)
        if idx < 0 or idx >= len(files):
            raise IndexError(f"piece_index {idx} out of range (found {len(files)} MIDI files)")
        path = files[idx]
        slug = simulation_slug(Path(piece_id_from_path(path, root)).stem.replace("/", "_"))
        return path, slug
    raise SystemExit("Provide --midi-path or --maestro-root (and optionally --piece-index).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate Variations predicted poses on MAESTRO MIDI and render RoboPianist video.")
    parser.add_argument("--model-type", required=True, help="One of: mlp_baseline, diffusion, latent_mdn")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to best.pt (or trained checkpoint).")
    parser.add_argument("--midi-path", type=str, default=None, help="Path to a single .mid / .midi file.")
    parser.add_argument("--maestro-root", type=str, default=None, help="MAESTRO dataset root (recursive .mid discovery).")
    parser.add_argument("--piece-index", type=int, default=0, help="Index into sorted MAESTRO midis when using --maestro-root.")
    parser.add_argument("--limit-songs", type=int, default=None, help="Optional cap when scanning --maestro-root (debug).")
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT), help="Parent directory for run outputs.")
    parser.add_argument("--control-timestep", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-duration-s", type=float, default=None)
    parser.add_argument(
        "--environment-name",
        type=str,
        default="RoboPianist-debug-TwinkleTwinkleLittleStar-v0",
        help="Registered RoboPianist name passed to loader (midi_file proto overrides score).",
    )
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--render-every", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--diffusion-steps", type=int, default=None, help="Override DDIM-style steps for diffusion models.")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--prefer-canonical-midi", action="store_true")

    args = parser.parse_args()
    midi_path, piece_slug = _resolve_midi_path(args)

    target_keys, midi_meta = midi_to_target_key_roll(
        midi_path,
        control_timestep=args.control_timestep,
        max_steps=args.max_steps,
        max_duration_s=args.max_duration_s,
    )
    steps = target_keys.shape[0]

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    run_name = f"{ts}_{simulation_slug(args.model_type)}_{piece_slug}"
    run_dir = Path(args.output_root).expanduser().resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_simulation_model(
        args.checkpoint,
        args.model_type,
        device=args.device,
        diffusion_steps=args.diffusion_steps,
    )
    hand_joints = loaded.predict_hand_states(target_keys, batch_size=int(args.batch_size))

    rollout_meta = rollout_variations_maestro_prediction(
        target_keys=target_keys,
        hand_joints=hand_joints,
        song_name=str(args.environment_name),
        output_dir=run_dir,
        label="simulation",
        control_timestep=float(args.control_timestep),
        fps=int(args.fps),
        width=int(args.width),
        height=int(args.height),
        max_steps=int(args.max_steps) if args.max_steps is not None else None,
        render_every=int(args.render_every),
        seed=int(args.seed),
        threshold=float(args.threshold),
        prefer_canonical_midi=bool(args.prefer_canonical_midi),
    )

    vp = rollout_meta.get("video_path")
    if vp:
        src = Path(vp)
        if src.is_file():
            dst = run_dir / f"simulation{src.suffix.lower()}"
            shutil.copy2(src, dst)
            rollout_meta["simulation_media"] = str(dst)

    proto = rollout_meta.get("midi_proto_path")
    if proto:
        pth = Path(proto)
        if pth.is_file():
            shutil.copy2(pth, run_dir / "target_goals.proto")

    run_meta = {
        "run_name": run_name,
        "model_type": str(args.model_type),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "midi_path": str(midi_path),
        "piece_slug": piece_slug,
        "environment_name": str(args.environment_name),
        "control_timestep": float(args.control_timestep),
        "max_steps": args.max_steps,
        "max_duration_s": args.max_duration_s,
        "midi_quantization": midi_meta,
        "inference": {
            "batch_size": int(args.batch_size),
            "device": str(loaded.device),
            "diffusion_steps": int(loaded.diffusion_steps) if loaded.model_type == "diffusion" else None,
        },
        "target_keys_shape": list(target_keys.shape),
        "hand_joints_shape": list(hand_joints.shape),
    }
    write_simulation_bundle(
        run_dir,
        target_keys=target_keys,
        hand_joints=hand_joints,
        rollout_meta=rollout_meta,
        run_meta=run_meta,
    )
    print(f"Wrote simulation under {run_dir}")


if __name__ == "__main__":
    main()
