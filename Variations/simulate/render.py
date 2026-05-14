from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from partita.evaluation.rollout import (
    _load_env,
    _locate_task_physics_piano,
    _set_reduced_hand_qpos,
    candidate_environment_names,
    piano_roll_to_midi_events,
    render_frame,
    write_goals_proto,
    write_video,
)
from partita.evaluation.metrics import key_metrics
from partita.utils.io import ensure_dir, save_json


def rollout_variations_maestro_prediction(
    *,
    target_keys: np.ndarray,
    hand_joints: np.ndarray,
    song_name: str,
    output_dir: str | Path,
    label: str = "variations_maestro_sim",
    control_timestep: float = 0.05,
    fps: int = 20,
    width: int = 640,
    height: int = 480,
    max_steps: int | None = None,
    render_every: int = 1,
    seed: int = 0,
    threshold: float = 0.5,
    reduced_action_space: bool = True,
    prefer_canonical_midi: bool = False,
    extra_task_kwargs: dict[str, Any] | None = None,
    suite_load_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Frame-by-frame RoboPianist visualization: predicted hand qpos + physical piano state.

    This is not an online policy rollout because the Variations models emit hand poses,
    not actions. It intentionally does not copy MIDI-derived target keys into the piano
    state. Any captured key activation must come from the simulator's piano state after
    the predicted hand pose is restored and physics is forwarded.
    """
    os.environ.setdefault("MUJOCO_GL", "egl")
    output_dir = ensure_dir(output_dir)
    target_keys = np.asarray(target_keys, dtype=np.float32)
    hand_joints = np.asarray(hand_joints, dtype=np.float32)
    if target_keys.ndim != 2 or target_keys.shape[1] < 88:
        raise ValueError(f"Expected target_keys [T, 88+], got {target_keys.shape}")
    if hand_joints.ndim != 2:
        raise ValueError(f"Expected hand_joints [T, joints], got {hand_joints.shape}")

    goals = target_keys[:, :88]
    steps = min(int(hand_joints.shape[0]), int(goals.shape[0]))
    if max_steps is not None:
        steps = min(steps, int(max_steps))

    midi_proto_path = write_goals_proto(
        goals[:steps],
        output_dir / f"{label}_target_goals.proto",
        dt=control_timestep,
        title=f"Variations MAESTRO sim {song_name}",
    )
    env_name, env, load_info = _load_env(
        environment_names=candidate_environment_names(song_name),
        midi_proto_path=midi_proto_path,
        control_timestep=control_timestep,
        seed=seed,
        reduced_action_space=reduced_action_space,
        extra_task_kwargs=extra_task_kwargs,
        suite_load_kwargs=suite_load_kwargs,
        prefer_canonical_midi=prefer_canonical_midi,
    )

    frames: list[np.ndarray] = []
    played_roll: list[np.ndarray] = []
    render_error: str | None = None
    restored_hand_joint_count = 0

    try:
        env.reset()
        task, physics, piano = _locate_task_physics_piano(env)
        for step_index in range(steps):
            restored_hand_joint_count = _set_reduced_hand_qpos(task, physics, hand_joints[step_index])
            if hasattr(physics, "forward"):
                physics.forward()
            piano._update_key_state(physics)
            piano._update_key_color(physics)
            played_roll.append(np.asarray(piano.activation, dtype=np.float32).reshape(-1)[:88])
            if step_index % max(int(render_every), 1) == 0 and render_error is None:
                try:
                    frames.append(render_frame(env, height=height, width=width))
                except Exception as exc:
                    render_error = str(exc)

        video_path = None
        video_format = None
        audio_warning = None
        played_for_audio = np.stack(played_roll, axis=0) if played_roll else np.zeros((0, 88), dtype=np.float32)
        try:
            audio_events = piano_roll_to_midi_events(played_for_audio, dt=control_timestep, threshold=threshold)
        except ModuleNotFoundError as exc:
            audio_events = []
            audio_warning = f"Audio disabled because a playback dependency is unavailable: {exc}"
        if render_error is None:
            video_path, video_format, write_audio_warning = write_video(
                frames,
                output_dir / f"{label}_playback.mp4",
                fps=max(int(fps / max(render_every, 1)), 1),
                audio_events=audio_events,
            )
            audio_warning = audio_warning or write_audio_warning

        played = np.stack(played_roll, axis=0) if played_roll else np.zeros((0, 88), dtype=np.float32)
        goal_steps = min(int(goals.shape[0]), int(played.shape[0]))
        goal_keys = min(int(goals.shape[-1]), int(played.shape[-1]), 88)
        against_goals = key_metrics(goals[:goal_steps, :goal_keys], played[:goal_steps, :goal_keys], threshold=threshold)

        result: dict[str, Any] = {
            "label": label,
            "song_name": song_name,
            "environment_name": env_name,
            "playback_mode": "variations_predicted_hands_physical_piano_state",
            "audio_source": "physical_piano_activation_after_pose_injection",
            "midi_key_state_injection": False,
            "evaluation_warning": (
                "This playback does not step a policy action and should not be used as "
                "an online playing evaluation. It only visualizes predicted hand poses "
                "and records physical piano activation after pose injection."
            ),
            "midi_proto_path": str(midi_proto_path),
            "load_info": {k: (str(v) if isinstance(v, Path) else v) for k, v in load_info.items()},
            "target_keys_shape": list(goals.shape),
            "hand_joints_shape": list(hand_joints.shape),
            "steps_rendered": int(steps),
            "restored_hand_joint_count": int(restored_hand_joint_count),
            "rendered_frames": int(len(frames)),
            "render_every": int(render_every),
            "render_error": render_error,
            "video_path": str(video_path) if video_path is not None else None,
            "video_format": video_format,
            "audio_warning": audio_warning,
            "against_goals": {**against_goals, "scored_steps": int(goal_steps), "scored_keys": int(goal_keys)},
        }
        save_json(output_dir / f"{label}_playback.json", result)
        return result
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def write_simulation_bundle(
    output_dir: Path,
    *,
    target_keys: np.ndarray,
    hand_joints: np.ndarray,
    rollout_meta: dict[str, Any],
    run_meta: dict[str, Any],
) -> None:
    output_dir = ensure_dir(output_dir)
    save_json(output_dir / "simulation.json", {**run_meta, "rollout": rollout_meta})
    np.savez_compressed(
        output_dir / "inputs_and_predictions.npz",
        target_keys=np.asarray(target_keys, dtype=np.float32),
        hand_joints=np.asarray(hand_joints, dtype=np.float32),
    )


__all__ = ["rollout_variations_maestro_prediction", "write_simulation_bundle"]
