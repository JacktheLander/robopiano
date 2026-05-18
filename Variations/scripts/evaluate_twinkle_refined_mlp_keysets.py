#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
VARIATIONS_ROOT = REPO_ROOT / "Variations"
for path in (
    VARIATIONS_ROOT / "src",
    VARIATIONS_ROOT,
    REPO_ROOT / "partita" / "src",
    REPO_ROOT,
):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from partita.utils.io import ensure_dir, save_json  # noqa: E402
from simulate.midi_keysets import discover_midi_files, midi_to_target_key_roll  # noqa: E402
from simulate.model_loader import load_simulation_model  # noqa: E402
from simulate.render import rollout_variations_maestro_prediction  # noqa: E402


DEFAULT_TARGET_KEYS_NPZ = Path(
    "/WAVE/datasets/ccoelho_lab-jlanders/Variations/variations_rp1m_full_20260510_152232/"
    "variations/extraction/full/song_RoboPianist-debug-TwinkleTwinkleLittleStar-v0_0.npz"
)
DEFAULT_CHECKPOINT = Path(
    "/WAVE/datasets/ccoelho_lab-jlanders/Variations/variations_tuned_20260512_190752/"
    "variations/contact_refinement/mlp_reuse_labels_20260514_104844/mlp_baseline_contact.pt"
)
DEFAULT_OUTPUT_PARENT = Path(
    "/WAVE/datasets/ccoelho_lab-jlanders/Variations/variations_tuned_20260512_190752/"
    "variations/twinkle_refined_mlp_keyset_eval"
)
DEFAULT_ENVIRONMENT_NAME = "RoboPianist-debug-TwinkleTwinkleLittleStar-v0"
DEFAULT_MAESTRO_ROOT = Path("/WAVE/datasets/ccoelho_lab-jlanders/maestro-v3.0.0/maestro-v3.0.0")


def load_target_keys_npz(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=False)
    source_key = None
    if "target_keys" in data.files:
        source_key = "target_keys"
    elif "target_key" in data.files:
        source_key = "target_key"
    else:
        for key in data.files:
            value = np.asarray(data[key])
            if value.ndim == 2 and value.shape[1] >= 88:
                source_key = key
                break
    if source_key is None:
        raise ValueError(f"No 2D target key array found in {path}; available arrays: {data.files}")
    target_keys = np.asarray(data[source_key], dtype=np.float32)
    if target_keys.ndim != 2 or target_keys.shape[1] < 88:
        raise ValueError(f"{path}:{source_key} must have shape [T, 88+], got {target_keys.shape}")
    meta = {"source": "npz", "target_keys_npz": str(path), "npz_array_key": source_key, "npz_arrays": list(data.files)}
    return target_keys[:, :88], meta


def load_midi_target_keys(args: argparse.Namespace, midi_path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    target_keys, midi_meta = midi_to_target_key_roll(
        midi_path,
        control_timestep=float(args.control_timestep),
        max_steps=int(args.max_steps) if args.max_steps is not None else None,
        max_duration_s=float(args.max_duration_s) if args.max_duration_s is not None else None,
    )
    return target_keys[:, :88], {"source": "midi", "midi": midi_meta}


def random_maestro_midi(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    root = Path(args.maestro_root).expanduser().resolve()
    files = discover_midi_files(root)
    if not files:
        raise RuntimeError(f"No MIDI files found under MAESTRO root: {root}")
    rng = random.Random(int(args.seed))
    index = rng.randrange(len(files))
    return files[index], {
        "maestro_root": str(root),
        "selection": "random_seeded_fallback",
        "seed": int(args.seed),
        "selected_index": int(index),
        "candidate_count": int(len(files)),
    }


def load_target_keys(args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    if args.midi_path:
        return load_midi_target_keys(args, args.midi_path)

    npz_meta: dict[str, Any] | None = None
    try:
        target_keys, npz_meta = load_target_keys_npz(Path(args.target_keys_npz).expanduser().resolve())
    except FileNotFoundError:
        if not bool(args.fallback_random_maestro):
            raise
        target_keys = np.zeros((0, 88), dtype=np.float32)

    if target_keys.shape[0] == 0 and bool(args.fallback_random_maestro):
        midi_path, fallback_meta = random_maestro_midi(args)
        target_keys, midi_meta = midi_to_target_key_roll(
            midi_path,
            control_timestep=float(args.control_timestep),
            max_steps=int(args.max_steps) if args.max_steps is not None else None,
            max_duration_s=float(args.max_duration_s) if args.max_duration_s is not None else None,
        )
        return target_keys[:, :88], {
            "source": "maestro_random_fallback",
            "empty_or_missing_target_keys_npz": str(args.target_keys_npz),
            "npz_meta": npz_meta,
            "fallback": fallback_meta,
            "midi": midi_meta,
        }

    if args.max_steps is not None:
        target_keys = target_keys[: int(args.max_steps)]
    if args.max_duration_s is not None:
        max_steps = int(np.ceil(float(args.max_duration_s) / float(args.control_timestep)))
        target_keys = target_keys[:max_steps]
    if target_keys.shape[0] == 0:
        raise ValueError(
            f"{args.target_keys_npz} contains zero target-key frames. Provide --midi-path for Twinkle MIDI playback "
            "or point --target-keys-npz at a non-empty [T, 88] target-key roll."
        )
    return target_keys, npz_meta or {"source": "npz", "target_keys_npz": str(args.target_keys_npz)}


def unique_keysets(target_keys: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    binary = (np.asarray(target_keys, dtype=np.float32)[:, :88] > float(threshold)).astype(np.uint8)
    unique, inverse = np.unique(binary, axis=0, return_inverse=True)
    active_mask = unique.sum(axis=1) > 0
    frame_counts = np.bincount(inverse, minlength=unique.shape[0])
    rest_ids = np.flatnonzero(~active_mask)
    return unique.astype(np.float32), inverse.astype(np.int32), {
        "unique_keyset_count": int(unique.shape[0]),
        "active_unique_keyset_count": int(active_mask.sum()),
        "rest_unique_keyset_count": int((~active_mask).sum()),
        "active_frame_count": int(binary.sum(axis=1).astype(bool).sum()),
        "rest_frame_count": int((binary.sum(axis=1) == 0).sum()),
        "frame_counts_by_keyset": frame_counts.astype(int).tolist(),
        "rest_keyset_ids": rest_ids.astype(int).tolist(),
    }


def active_key_text(keyset: np.ndarray) -> str:
    return " ".join(str(int(idx)) for idx in np.flatnonzero(np.asarray(keyset) > 0.5))


def write_keysets_csv(
    path: Path,
    *,
    unique: np.ndarray,
    inverse: np.ndarray,
    unique_hand_joints: np.ndarray,
) -> None:
    ensure_dir(path.parent)
    fieldnames = [
        "keyset_id",
        "active_key_count",
        "active_key_indices",
        "frame_count",
        "first_frame",
        "last_frame",
        "pose_l2_norm",
        "pose_min",
        "pose_max",
        "pose_mean",
        "pose_std",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for keyset_id, keyset in enumerate(unique):
            frames = np.flatnonzero(inverse == keyset_id)
            pose = np.asarray(unique_hand_joints[keyset_id], dtype=np.float32)
            writer.writerow(
                {
                    "keyset_id": int(keyset_id),
                    "active_key_count": int(np.sum(keyset > 0.5)),
                    "active_key_indices": active_key_text(keyset),
                    "frame_count": int(frames.size),
                    "first_frame": int(frames[0]) if frames.size else "",
                    "last_frame": int(frames[-1]) if frames.size else "",
                    "pose_l2_norm": float(np.linalg.norm(pose)),
                    "pose_min": float(np.min(pose)),
                    "pose_max": float(np.max(pose)),
                    "pose_mean": float(np.mean(pose)),
                    "pose_std": float(np.std(pose)),
                }
            )


def output_dir_from_args(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir).expanduser().resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    return DEFAULT_OUTPUT_PARENT / f"{timestamp}_twinkle_refined_mlp_keysets"


def copy_playback_media(run_dir: Path, rollout: dict[str, Any]) -> str | None:
    source = rollout.get("video_path")
    if not source:
        return None
    src = Path(str(source))
    if not src.is_file():
        return None
    dst = run_dir / f"twinkle_refined_mlp_keyset_playback{src.suffix.lower()}"
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    return str(dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate refined Variations MLP playback on Twinkle unique keysets.")
    parser.add_argument("--target-keys-npz", default=str(DEFAULT_TARGET_KEYS_NPZ))
    parser.add_argument("--midi-path", default=None, help="Optional MIDI fallback/source for target-key roll.")
    parser.add_argument("--maestro-root", default=str(DEFAULT_MAESTRO_ROOT))
    parser.add_argument("--fallback-random-maestro", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--environment-name", default=DEFAULT_ENVIRONMENT_NAME)
    parser.add_argument("--control-timestep", type=float, default=0.05)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-duration-s", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--render-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    run_dir = ensure_dir(output_dir_from_args(args))
    target_keys, source_meta = load_target_keys(args)
    unique, frame_to_keyset_id, keyset_meta = unique_keysets(target_keys, float(args.threshold))

    model = load_simulation_model(str(checkpoint), "mlp_baseline", device=str(args.device))
    unique_hand_joints = model.predict_hand_states(unique, batch_size=int(args.batch_size)).astype(np.float32)
    frame_hand_joints = unique_hand_joints[frame_to_keyset_id].astype(np.float32)

    np.savez_compressed(
        run_dir / "inputs_and_predictions.npz",
        target_keys=np.asarray(target_keys, dtype=np.float32),
        unique_keysets=np.asarray(unique, dtype=np.float32),
        frame_to_keyset_id=np.asarray(frame_to_keyset_id, dtype=np.int32),
        unique_hand_joints=unique_hand_joints,
        frame_hand_joints=frame_hand_joints,
    )
    write_keysets_csv(run_dir / "keysets.csv", unique=unique, inverse=frame_to_keyset_id, unique_hand_joints=unique_hand_joints)

    rollout = rollout_variations_maestro_prediction(
        target_keys=target_keys,
        hand_joints=frame_hand_joints,
        song_name=str(args.environment_name),
        output_dir=run_dir,
        label="twinkle_refined_mlp_keyset",
        control_timestep=float(args.control_timestep),
        fps=int(args.fps),
        width=int(args.width),
        height=int(args.height),
        max_steps=int(args.max_steps) if args.max_steps is not None else None,
        render_every=int(args.render_every),
        seed=int(args.seed),
        threshold=float(args.threshold),
    )
    media_path = copy_playback_media(run_dir, rollout)

    summary = {
        "checkpoint": str(checkpoint),
        "model_type": "mlp_baseline",
        "target_source": source_meta,
        "environment_name": str(args.environment_name),
        "control_timestep": float(args.control_timestep),
        "threshold": float(args.threshold),
        "target_keys_shape": list(target_keys.shape),
        "unique_keysets_shape": list(unique.shape),
        "unique_hand_joints_shape": list(unique_hand_joints.shape),
        "frame_hand_joints_shape": list(frame_hand_joints.shape),
        **keyset_meta,
        "playback_json": str(run_dir / "twinkle_refined_mlp_keyset_playback.json"),
        "playback_media": media_path,
        "against_goals": rollout.get("against_goals"),
        "rollout": rollout,
    }
    save_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote Twinkle refined MLP keyset evaluation under {run_dir}")


if __name__ == "__main__":
    main()
