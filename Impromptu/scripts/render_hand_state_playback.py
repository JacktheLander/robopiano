#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (
    REPO_ROOT / "Impromptu" / "src",
    REPO_ROOT / "Intermezzo" / "src",
    REPO_ROOT / "partita" / "src",
    REPO_ROOT,
):
    _text = str(_path)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from intermezzo.io import atomic_save_json, atomic_save_npz  # noqa: E402
from intermezzo.online_eval import score_rollout  # noqa: E402
from partita.evaluation.rollout import (  # noqa: E402
    _capture_piano_activation,
    _load_env,
    _locate_task_physics_piano,
    _set_reduced_hand_qpos,
    candidate_environment_names,
    piano_roll_to_midi_events,
    render_frame,
    write_goals_proto,
    write_video,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render dense Impromptu hand-state playback in RoboPianist.")
    parser.add_argument("--trajectory-npz", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--environment-name", default="RoboPianist-debug-TwinkleTwinkleLittleStar-v0")
    parser.add_argument("--control-timestep", type=float, default=0.05)
    parser.add_argument("--interpolation-substeps", type=int, default=None)
    parser.add_argument("--fps", type=int, default=200)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser


def _load_dense_payload(path: Path, control_timestep: float, interpolation_substeps: int | None) -> tuple[np.ndarray, np.ndarray, float, int]:
    data = np.load(path, allow_pickle=False)
    if "planned_hand_joints_dense" not in data:
        raise ValueError(f"{path} does not contain planned_hand_joints_dense")
    dense = np.asarray(data["planned_hand_joints_dense"], dtype=np.float32)
    target_keys = np.asarray(data["target_keys"], dtype=np.float32) if "target_keys" in data else np.zeros((dense.shape[0], 88), dtype=np.float32)
    if interpolation_substeps is None:
        if target_keys.shape[0] > 0 and dense.shape[0] % target_keys.shape[0] == 0:
            substeps = max(int(dense.shape[0] // target_keys.shape[0]), 1)
        else:
            substeps = 1
    else:
        substeps = max(int(interpolation_substeps), 1)
    if target_keys.shape[0] * substeps == dense.shape[0]:
        target_keys_dense = np.repeat(target_keys[:, :88], substeps, axis=0).astype(np.float32)
    elif target_keys.shape[0] == dense.shape[0]:
        target_keys_dense = target_keys[:, :88].astype(np.float32)
    else:
        raise ValueError(f"Cannot align target_keys shape {target_keys.shape} with dense hand states {dense.shape}")
    dense_dt = float(control_timestep) / float(substeps)
    return dense, target_keys_dense, dense_dt, substeps


def render_dense_playback(
    *,
    trajectory_npz: str | Path,
    output_dir: str | Path | None,
    environment_name: str,
    control_timestep: float,
    interpolation_substeps: int | None,
    fps: int,
    width: int,
    height: int,
    seed: int,
    threshold: float,
) -> dict[str, Any]:
    npz_path = Path(trajectory_npz).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve() if output_dir else npz_path.parent / "render_200fps"
    out_dir.mkdir(parents=True, exist_ok=True)
    hand_states, target_keys_dense, dense_dt, substeps = _load_dense_payload(npz_path, control_timestep, interpolation_substeps)
    midi_proto = write_goals_proto(target_keys_dense[:, :88], out_dir / "impromptu_dense_target_goals.proto", dt=dense_dt, title="Impromptu dense playback render")
    env_name, env, load_info = _load_env(
        environment_names=candidate_environment_names(environment_name),
        midi_proto_path=midi_proto,
        control_timestep=dense_dt,
        seed=int(seed),
        reduced_action_space=True,
        extra_task_kwargs={
            "disable_forearm_reward": True,
            "disable_fingering_reward": True,
            "disable_colorization": True,
            "disable_hand_collisions": False,
            "wrong_press_termination": False,
        },
        suite_load_kwargs=None,
        prefer_canonical_midi=False,
    )
    played_roll: list[np.ndarray] = []
    frames: list[np.ndarray] = []
    render_error = None
    terminated = False
    restored_count = 0
    try:
        env.reset()
        task, physics, piano = _locate_task_physics_piano(env)
        action_spec = env.action_spec()
        zero_action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        for hand_state in hand_states:
            restored_count = _set_reduced_hand_qpos(task, physics, hand_state)
            if hasattr(physics, "forward"):
                physics.forward()
            timestep = env.step(zero_action)
            update_key_state = getattr(piano, "_update_key_state", None)
            if callable(update_key_state):
                update_key_state(physics)
            update_key_color = getattr(piano, "_update_key_color", None)
            if callable(update_key_color):
                update_key_color(physics)
            activation = _capture_piano_activation(env)
            if activation is None:
                activation = np.asarray(getattr(piano, "activation"), dtype=np.float32).reshape(-1)[:88]
            played_roll.append(np.asarray(activation[:88], dtype=np.float32))
            if render_error is None:
                try:
                    frames.append(render_frame(env, height=int(height), width=int(width)))
                except Exception as exc:
                    render_error = str(exc)
            if timestep.last():
                terminated = True
                break
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    played = np.stack(played_roll, axis=0) if played_roll else np.zeros((0, 88), dtype=np.float32)
    steps = min(int(played.shape[0]), int(target_keys_dense.shape[0]))
    rollout_path = out_dir / "dense_playback_rollout.npz"
    atomic_save_npz(rollout_path, target_keys=target_keys_dense[:steps], played_keys=played[:steps])
    score = score_rollout(
        target_keys=target_keys_dense[:steps],
        played_keys=played[:steps],
        dt=dense_dt,
        threshold=float(threshold),
        timing_tolerance_s=0.15,
    )
    events = piano_roll_to_midi_events(played[:steps], dt=dense_dt, threshold=float(threshold))
    video_path = None
    video_format = None
    video_audio_warning = None
    if render_error is None and frames:
        video_path, video_format, video_audio_warning = write_video(frames, out_dir / "rollout_video.mp4", fps=int(fps), audio_events=events)
    summary: dict[str, Any] = {
        "trajectory_npz": str(npz_path),
        "run_dir": str(out_dir),
        "rollout_npz": str(rollout_path),
        "midi_proto_path": str(midi_proto),
        "environment_name": env_name,
        "load_info": load_info,
        "dense_control_timestep": dense_dt,
        "interpolation_substeps": int(substeps),
        "fps": int(fps),
        "video_path": str(video_path) if video_path is not None else None,
        "video_format": video_format,
        "video_audio_warning": video_audio_warning,
        "render_error": render_error,
        "rendered_frames": int(len(frames)),
        "terminated": bool(terminated),
        "restored_hand_joint_count": int(restored_count),
        "score": score,
    }
    atomic_save_json(out_dir / "render_summary.json", summary)
    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = render_dense_playback(
        trajectory_npz=args.trajectory_npz,
        output_dir=args.output_dir,
        environment_name=str(args.environment_name),
        control_timestep=float(args.control_timestep),
        interpolation_substeps=args.interpolation_substeps,
        fps=int(args.fps),
        width=int(args.width),
        height=int(args.height),
        seed=int(args.seed),
        threshold=float(args.threshold),
    )
    print(f"Wrote Impromptu dense playback render: {summary.get('video_path')}")
    print(
        "matched="
        f"{summary['score']['matched_press_events']}/{summary['score']['target_press_events']} "
        f"missed={summary['score']['missed_key_presses']} "
        f"mispresses={summary['score']['mispresses']}"
    )


if __name__ == "__main__":
    main()
