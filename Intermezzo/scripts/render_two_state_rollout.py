#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np

INTERMEZZO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = INTERMEZZO_ROOT.parent
for path in (INTERMEZZO_ROOT / "src", REPO_ROOT / "partita" / "src", REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from intermezzo.evaluation import compute_interpolation_scale_metrics, compute_two_state_metrics, save_metrics  # noqa: E402
from intermezzo.key_geometry import load_key_geometry  # noqa: E402
from intermezzo.kinematics import RoboPianistHandKinematics  # noqa: E402
from partita.evaluation.rollout import (  # noqa: E402
    _clone_midi_events,
    _default_soundfont_path,
    _load_env,
    _locate_task_physics_piano,
    _note_midi_events,
    _set_reduced_hand_qpos,
    _write_waveform,
    candidate_environment_names,
    piano_roll_to_midi_events,
    render_frame,
    write_goals_proto,
    write_video,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a two-state Intermezzo trajectory in RoboPianist.")
    parser.add_argument("trajectory_npz", nargs="?", default=None, help="Path to two_state_trajectory.npz")
    parser.add_argument("--trajectory-npz", dest="trajectory_npz_flag", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--song-name", default="RoboPianist-debug-TwinkleTwinkleLittleStar-v0")
    parser.add_argument("--control-timestep", type=float, default=0.05)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--render-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser


def render_two_state_rollout(
    *,
    trajectory_path: str | Path,
    output_dir: str | Path | None = None,
    song_name: str = "RoboPianist-debug-TwinkleTwinkleLittleStar-v0",
    control_timestep: float = 0.05,
    fps: int = 20,
    width: int = 640,
    height: int = 480,
    render_every: int = 1,
    seed: int = 0,
    threshold: float = 0.5,
) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    npz_path = Path(trajectory_path).expanduser().resolve()
    data = np.load(npz_path, allow_pickle=False)
    planned = np.asarray(data["planned_hand_joints"], dtype=np.float32)
    target_keys = np.asarray(data["target_keys"], dtype=np.float32) if "target_keys" in data else np.zeros((planned.shape[0], 88), dtype=np.float32)
    waypoint_frames = np.asarray(data["waypoint_frames"], dtype=np.int64) if "waypoint_frames" in data else np.zeros((0,), dtype=np.int64)
    keyset_b = np.asarray(data["keyset_b"], dtype=np.float32) if "keyset_b" in data else target_keys[-1]
    endpoint_hand_joints = np.asarray(data["endpoint_hand_joints"], dtype=np.float32) if "endpoint_hand_joints" in data else planned[[0, -1]]
    out_dir = Path(output_dir).expanduser().resolve() if output_dir is not None else npz_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    midi_proto_path = write_goals_proto(target_keys[:, :88], out_dir / "two_state_target_goals.proto", dt=float(control_timestep), title="Intermezzo two-state render target")
    env_name, env, load_info = _load_env(
        environment_names=candidate_environment_names(song_name),
        midi_proto_path=midi_proto_path,
        control_timestep=float(control_timestep),
        seed=int(seed),
        reduced_action_space=True,
    )

    frames: list[np.ndarray] = []
    pressed_roll: list[np.ndarray] = []
    restored_hand_joint_count = 0
    render_error: str | None = None
    try:
        env.reset()
        task, physics, piano = _locate_task_physics_piano(env)
        kinematics = RoboPianistHandKinematics(task, physics)
        for step_index, hand_state in enumerate(planned):
            restored_hand_joint_count = _set_reduced_hand_qpos(task, physics, hand_state)
            if hasattr(physics, "forward"):
                physics.forward()
            piano._update_key_state(physics)
            piano._update_key_color(physics)
            pressed_roll.append(np.asarray(piano.activation, dtype=np.float32).reshape(-1)[:88])
            if step_index % max(int(render_every), 1) == 0 and render_error is None:
                try:
                    frames.append(render_frame(env, height=int(height), width=int(width)))
                except Exception as exc:
                    render_error = str(exc)

        pressed = np.stack(pressed_roll, axis=0) if pressed_roll else np.zeros((0, 88), dtype=np.float32)
        events = piano_roll_to_midi_events(pressed, dt=float(control_timestep), threshold=float(threshold))
        csv_path = _write_key_events_csv(out_dir / "robot_key_events.csv", pressed, dt=float(control_timestep), threshold=float(threshold))
        wav_path, audio_warning = _write_robot_pressed_audio(out_dir / "robot_pressed_audio.wav", events, duration_s=float(planned.shape[0] * control_timestep))
        video_path = None
        video_format = None
        video_audio_warning = None
        if render_error is None:
            video_path, video_format, video_audio_warning = write_video(
                frames,
                out_dir / "rollout_video.mp4",
                fps=max(int(fps / max(int(render_every), 1)), 1),
                audio_events=events,
            )
        key_xy = load_key_geometry(allow_approximate=False)
        metrics = compute_two_state_metrics(
            planned_hand_joints=planned,
            endpoint_hand_joints=endpoint_hand_joints,
            final_target_keys=keyset_b,
            key_xy=key_xy,
            kinematics=kinematics,
            robot_pressed_keys=pressed,
            control_timestep=float(control_timestep),
            threshold=float(threshold),
        )
        metrics.update(
            {
                "environment_name": env_name,
                "trajectory_npz": str(npz_path),
                "video_path": str(video_path) if video_path is not None else None,
                "video_format": video_format,
                "robot_pressed_audio_wav": str(wav_path) if wav_path is not None else None,
                "robot_key_events_csv": str(csv_path),
                "render_error": render_error,
                "audio_warning": audio_warning,
                "video_audio_warning": video_audio_warning,
                "audio_source": "robot_physical_key_activation_only",
                "restored_hand_joint_count": int(restored_hand_joint_count),
                "rendered_frames": int(len(frames)),
                "load_info": load_info,
                **compute_interpolation_scale_metrics(
                    planned_timestep_count=int(planned.shape[0]),
                    target_state_count=int(waypoint_frames.size),
                ),
            }
        )
        save_metrics(out_dir / "rollout_metrics.json", metrics)
        return metrics
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def _write_key_events_csv(path: Path, piano_roll: np.ndarray, *, dt: float, threshold: float) -> Path:
    active = np.asarray(piano_roll, dtype=np.float32)[:, :88] > float(threshold)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["time_s", "step", "event", "key_index", "midi_note"])
        writer.writeheader()
        for key in range(active.shape[1]):
            was_active = False
            for step, is_active in enumerate(active[:, key]):
                if bool(is_active) and not was_active:
                    writer.writerow({"time_s": float(step * dt), "step": int(step), "event": "note_on", "key_index": int(key), "midi_note": int(21 + key)})
                elif was_active and not bool(is_active):
                    writer.writerow({"time_s": float(step * dt), "step": int(step), "event": "note_off", "key_index": int(key), "midi_note": int(21 + key)})
                was_active = bool(is_active)
            if was_active:
                writer.writerow({"time_s": float(active.shape[0] * dt), "step": int(active.shape[0]), "event": "note_off", "key_index": int(key), "midi_note": int(21 + key)})
    return path


def _write_robot_pressed_audio(path: Path, audio_events: list[Any], *, duration_s: float) -> tuple[Path | None, str | None]:
    note_events = _note_midi_events(audio_events)
    sample_rate = 44100
    silence = np.zeros((max(int(duration_s * sample_rate), 1),), dtype=np.int16)
    if not note_events:
        _write_waveform(path, silence, sample_rate=sample_rate)
        return path, "Audio WAV is silent because the robot did not physically press any keys."
    soundfont_path = _default_soundfont_path()
    if soundfont_path is None:
        _write_waveform(path, silence, sample_rate=sample_rate)
        return path, "Audio WAV is silent because no valid soundfont file was found."
    try:
        from robopianist.music import synthesizer
    except Exception as exc:
        _write_waveform(path, silence, sample_rate=sample_rate)
        return path, f"Audio WAV is silent because RoboPianist synthesizer import failed: {exc}"
    synth = synthesizer.Synthesizer(soundfont_path=soundfont_path)
    try:
        waveform = synth.get_samples(_clone_midi_events(note_events))
    finally:
        synth.stop()
    _write_waveform(path, waveform)
    return path, None

def main() -> None:
    args = build_parser().parse_args()
    trajectory = args.trajectory_npz_flag or args.trajectory_npz
    if trajectory is None:
        raise SystemExit("Provide a two_state_trajectory.npz path.")
    summary = render_two_state_rollout(
        trajectory_path=trajectory,
        output_dir=args.output_dir,
        song_name=str(args.song_name),
        control_timestep=float(args.control_timestep),
        fps=int(args.fps),
        width=int(args.width),
        height=int(args.height),
        render_every=int(args.render_every),
        seed=int(args.seed),
        threshold=float(args.threshold),
    )
    print(f"Wrote Intermezzo two-state render: {summary.get('video_path')}")


if __name__ == "__main__":
    main()
