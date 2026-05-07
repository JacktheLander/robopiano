from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any

import numpy as np

from partita.evaluation.metrics import key_metrics
from partita.utils.io import ensure_dir, save_json


def canonicalize_rp1m_environment_name(song_name: str) -> str:
    return re.sub(r"(-v\d+)_\d+$", r"\1", str(song_name))


def candidate_environment_names(song_name: str) -> list[str]:
    canonical = canonicalize_rp1m_environment_name(song_name)
    candidates = [canonical]
    if "repertoire-150" in canonical:
        candidates.append(canonical.replace("repertoire-150", "etude-12"))
    seen = set()
    out = []
    for name in candidates:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def _load_music_pb2():
    # Avoid importing top-level note_seq here. On some WAVE nodes that path imports
    # fluidsynth and can hit a system libstdc++ mismatch before rollout starts.
    import importlib.util
    import site

    candidates = []
    for root in [*site.getsitepackages(), site.getusersitepackages()]:
        candidates.append(Path(root) / "note_seq" / "protobuf" / "music_pb2.py")
    for path in candidates:
        if path.exists():
            spec = importlib.util.spec_from_file_location("partita_note_seq_music_pb2", path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise RuntimeError("Could not find note_seq/protobuf/music_pb2.py for generated MIDI proto writing.")


def goals_to_note_sequence(goals: np.ndarray, *, dt: float = 0.05, title: str = "Partita target goals"):
    music_pb2 = _load_music_pb2()

    goal = np.asarray(goals) > 0.5
    if goal.ndim != 2:
        raise ValueError(f"Expected goals with shape [T, keys], got {goal.shape}")
    key_dim = min(goal.shape[1], 88)
    seq = music_pb2.NoteSequence()
    seq.sequence_metadata.title = title
    seq.sequence_metadata.artist = "partita"
    for key in range(key_dim):
        active = goal[:, key]
        start = None
        for t, is_active in enumerate(active):
            if is_active and start is None:
                start = t
            if start is not None and (not is_active or t == len(active) - 1):
                end_t = t + 1 if is_active and t == len(active) - 1 else t
                if end_t > start:
                    note = seq.notes.add()
                    note.pitch = int(21 + key)  # A0 is MIDI 21.
                    note.velocity = 80
                    note.start_time = float(start * dt)
                    note.end_time = float(max(end_t * dt, start * dt + dt))
                    note.part = 0  # Leave fingering unspecified; RoboPianist will use OT fingering reward.
                start = None
    seq.total_time = float(goal.shape[0] * dt)
    seq.tempos.add(qpm=60)
    return seq


def write_goals_proto(goals: np.ndarray, path: str | Path, *, dt: float, title: str) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    seq = goals_to_note_sequence(goals, dt=dt, title=title)
    with path.open("wb") as f:
        f.write(seq.SerializeToString())
    return path


def _iter_wrapped_envs(env: Any) -> list[Any]:
    current = env
    wrapped = []
    seen = set()
    while current is not None and id(current) not in seen:
        wrapped.append(current)
        seen.add(id(current))
        current = getattr(current, "_environment", None)
    return wrapped


def render_frame(env: Any, *, height: int, width: int) -> np.ndarray:
    errors = []
    for current in _iter_wrapped_envs(env):
        physics = getattr(current, "physics", None)
        if physics is not None and hasattr(physics, "render"):
            for kwargs in ({"height": height, "width": width}, {"height": height, "width": width, "camera_id": 0}):
                try:
                    frame = physics.render(**kwargs)
                    arr = np.asarray(frame, dtype=np.uint8)
                    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
                        return arr[..., :3]
                except Exception as exc:
                    errors.append(f"physics.render({kwargs}): {exc}")
    raise RuntimeError("Unable to render RoboPianist frame. Try MUJOCO_GL=egl or MUJOCO_GL=osmesa. " + " | ".join(errors[:4]))


def write_video(
    frames: list[np.ndarray],
    output_path: str | Path,
    *,
    fps: int,
    audio_events: list[Any] | None = None,
) -> tuple[Path, str, str | None]:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    if not frames:
        raise ValueError("No frames were captured.")
    audio_warning = None
    try:
        import imageio.v2 as imageio

        imageio.mimwrite(output_path, frames, fps=fps, codec="libx264", quality=7, macro_block_size=None)
        if audio_events:
            audio_warning = _attach_keypress_audio(output_path, audio_events)
        return output_path, "mp4", audio_warning
    except Exception as mp4_exc:
        gif_path = output_path.with_suffix(".gif")
        import imageio.v2 as imageio

        imageio.mimsave(gif_path, frames, fps=fps)
        return gif_path, f"gif fallback after MP4 failure: {mp4_exc}", None



def _scale_action_to_spec(action: np.ndarray, action_spec: Any, *, source: str) -> np.ndarray:
    values = np.asarray(action, dtype=np.float32).reshape(-1)
    minimum = np.asarray(action_spec.minimum, dtype=np.float32).reshape(-1)
    maximum = np.asarray(action_spec.maximum, dtype=np.float32).reshape(-1)
    control = np.zeros_like(minimum, dtype=np.float32)
    width = min(control.size, values.size)
    if width <= 0:
        return control
    clipped = np.clip(values[:width], -1.0, 1.0)
    if source == "normalized_minus_one_to_one":
        control[:width] = minimum[:width] + 0.5 * (clipped + 1.0) * (maximum[:width] - minimum[:width])
    elif source == "actuator_units":
        control[:width] = values[:width]
    else:
        raise ValueError(f"Unknown action source scale: {source}")
    return np.clip(control, minimum, maximum)

def _load_env(*, environment_names: list[str], midi_proto_path: Path, control_timestep: float, seed: int):
    import sys

    repo_root = Path(__file__).resolve().parents[4]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from robopianist import suite

    last_error = None
    for env_name in environment_names:
        try:
            env = suite.load(
                environment_name=env_name,
                midi_file=midi_proto_path,
                seed=seed,
                task_kwargs={
                    "control_timestep": float(control_timestep),
                    "n_steps_lookahead": 1,
                    "disable_colorization": False,
                    "disable_hand_collisions": False,
                    "reduced_action_space": True,
                },
            )
            return env_name, env
        except Exception as exc:  # pragma: no cover - depends on local RoboPianist assets.
            last_error = exc
    raise RuntimeError(f"Could not load RoboPianist environment from candidates {environment_names}: {last_error}")


def _collect_metrics(env: Any, *, terminated: bool) -> dict[str, float | str]:
    if terminated:
        for current in _iter_wrapped_envs(env):
            get_metrics = getattr(current, "get_musical_metrics", None)
            if callable(get_metrics):
                try:
                    return {k: float(v) for k, v in dict(get_metrics()).items()}
                except Exception as exc:
                    return {"metrics_error": str(exc)}
    return {"metrics_note": "Episode did not terminate before action sequence ended; final musical metrics unavailable."}


def _capture_piano_activation(env: Any) -> np.ndarray | None:
    for current in _iter_wrapped_envs(env):
        task = getattr(current, "task", None)
        piano = getattr(task, "piano", None)
        activation = getattr(piano, "activation", None)
        if activation is not None:
            return np.asarray(activation, dtype=np.float32).reshape(-1)
    return None


def _find_piano_midi_events(env: Any) -> list[Any]:
    for current in _iter_wrapped_envs(env):
        task = getattr(current, "task", None)
        piano = getattr(task, "piano", None)
        midi_module = getattr(piano, "midi_module", None)
        get_all = getattr(midi_module, "get_all_midi_messages", None)
        if callable(get_all):
            return list(get_all())
    return []


def _note_midi_events(events: list[Any]) -> list[Any]:
    return [event for event in events if type(event).__name__ in {"NoteOn", "NoteOff"}]


def _rollout_key_metrics(goals: np.ndarray, piano_roll: list[np.ndarray]) -> dict[str, float | str | int]:
    if not piano_roll:
        return {"rollout_key_metrics_note": "No piano activation frames were captured from RoboPianist rollout."}
    played = np.stack(piano_roll, axis=0)
    target = np.asarray(goals, dtype=np.float32)
    steps = min(int(target.shape[0]), int(played.shape[0]))
    keys = min(int(target.shape[-1]), int(played.shape[-1]), 88)
    metrics = key_metrics(target[:steps, :keys], played[:steps, :keys], threshold=0.5)
    return {
        "rollout_key_precision": metrics["key_precision"],
        "rollout_key_recall": metrics["key_recall"],
        "rollout_key_f1": metrics["key_f1"],
        "rollout_mispress_rate": metrics["mispress_rate"],
        "rollout_scored_steps": int(steps),
        "rollout_scored_keys": int(keys),
        "rollout_scoring_source": "robopianist_piano_activation_from_replayed_actions",
    }


def _locate_task_physics_piano(env: Any) -> tuple[Any, Any, Any]:
    for current in _iter_wrapped_envs(env):
        task = getattr(current, "task", None)
        physics = getattr(current, "physics", None)
        piano = getattr(task, "piano", None)
        if task is not None and physics is not None and piano is not None:
            return task, physics, piano
    raise RuntimeError("Could not locate RoboPianist task, physics, and piano handles.")


def _set_reduced_hand_qpos(task: Any, physics: Any, hand_qpos: np.ndarray) -> int:
    values = np.asarray(hand_qpos, dtype=np.float32).reshape(-1)
    joints = []
    for hand_name in ("right_hand", "left_hand"):
        hand = getattr(task, hand_name, None)
        hand_joints = getattr(hand, "joints", None)
        if hand_joints is not None:
            joints.extend(list(hand_joints))
    if values.size < len(joints):
        raise ValueError(f"RP1M hand_joints has {values.size} values but environment expects {len(joints)} joints.")
    for joint, value in zip(joints, values[: len(joints)]):
        physics.bind(joint).qpos = float(value)
    return len(joints)


def _set_piano_qpos_from_state(piano: Any, physics: Any, piano_state: np.ndarray, threshold: float) -> np.ndarray:
    state = np.asarray(piano_state, dtype=np.float32).reshape(-1)
    key_active = state[:88] > float(threshold)
    qpos_range = np.asarray(getattr(piano, "_qpos_range"), dtype=np.float64)
    inactive = np.maximum(qpos_range[:, 0], 0.0)
    active = qpos_range[:, 1]
    physics.bind(piano.joints).qpos = np.where(key_active, active, inactive)
    if state.size > 88:
        getattr(piano, "_sustain_state")[0] = float(state[88])
    else:
        getattr(piano, "_sustain_state")[0] = 0.0
    if hasattr(physics, "forward"):
        physics.forward()
    piano._update_key_state(physics)
    piano._update_key_color(physics)
    return np.asarray(piano.activation, dtype=np.float32).reshape(-1)[:88]


def piano_roll_to_midi_events(piano_roll: np.ndarray, *, dt: float, threshold: float) -> list[Any]:
    from robopianist.music import midi_file, midi_message

    active = np.asarray(piano_roll, dtype=np.float32)[:, :88] > float(threshold)
    events: list[Any] = []
    for key in range(active.shape[1]):
        was_active = False
        for step, is_active in enumerate(active[:, key]):
            time_value = float(step * dt)
            if is_active and not was_active:
                events.append(
                    midi_message.NoteOn(
                        note=midi_file.key_number_to_midi_number(key),
                        velocity=127,
                        time=time_value,
                    )
                )
            elif was_active and not is_active:
                events.append(
                    midi_message.NoteOff(
                        note=midi_file.key_number_to_midi_number(key),
                        time=time_value,
                    )
                )
            was_active = bool(is_active)
        if was_active:
            events.append(
                midi_message.NoteOff(
                    note=midi_file.key_number_to_midi_number(key),
                    time=float(active.shape[0] * dt),
                )
            )
    events.sort(key=lambda event: (float(getattr(event, "time", 0.0)), 0 if type(event).__name__ == "NoteOff" else 1))
    return events


def rollout_recorded_rp1m_episode_with_robopianist(
    *,
    hand_joints: np.ndarray,
    piano_states: np.ndarray,
    goals: np.ndarray,
    song_name: str,
    output_dir: str | Path,
    label: str = "rp1m_recorded_state",
    control_timestep: float = 0.05,
    fps: int = 20,
    width: int = 640,
    height: int = 480,
    max_steps: int | None = None,
    render_every: int = 1,
    seed: int = 0,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Render RP1M recorded hand/key states directly in RoboPianist.

    This validates dataset playback independently from open-loop action replay. Audio is
    generated only from transitions in recorded RP1M piano key states.
    """
    os.environ.setdefault("MUJOCO_GL", "egl")
    output_dir = ensure_dir(output_dir)
    hand_joints = np.asarray(hand_joints, dtype=np.float32)
    piano_states = np.asarray(piano_states, dtype=np.float32)
    goals = np.asarray(goals, dtype=np.float32)
    if hand_joints.ndim != 2:
        raise ValueError(f"Expected hand_joints [T, joints], got {hand_joints.shape}")
    if piano_states.ndim != 2:
        raise ValueError(f"Expected piano_states [T, keys], got {piano_states.shape}")
    steps = min(int(hand_joints.shape[0]), int(piano_states.shape[0]), int(goals.shape[0]))
    if max_steps is not None:
        steps = min(steps, int(max_steps))

    midi_proto_path = write_goals_proto(
        goals,
        output_dir / f"{label}_target_goals.proto",
        dt=control_timestep,
        title=f"Partita {label} {song_name}",
    )
    env_name, env = _load_env(
        environment_names=candidate_environment_names(song_name),
        midi_proto_path=midi_proto_path,
        control_timestep=control_timestep,
        seed=seed,
    )
    frames: list[np.ndarray] = []
    played_roll: list[np.ndarray] = []
    render_error = None
    restored_hand_joint_count = 0
    try:
        env.reset()
        task, physics, piano = _locate_task_physics_piano(env)
        for step_index in range(steps):
            restored_hand_joint_count = _set_reduced_hand_qpos(task, physics, hand_joints[step_index])
            activation = _set_piano_qpos_from_state(piano, physics, piano_states[step_index], threshold)
            played_roll.append(activation)
            if step_index % max(int(render_every), 1) == 0 and render_error is None:
                try:
                    frames.append(render_frame(env, height=height, width=width))
                except Exception as exc:
                    render_error = str(exc)
        video_path = None
        video_format = None
        audio_warning = None
        audio_events = piano_roll_to_midi_events(piano_states[:steps], dt=control_timestep, threshold=threshold)
        if render_error is None:
            video_path, video_format, audio_warning = write_video(
                frames,
                output_dir / f"{label}_playback.mp4",
                fps=max(int(fps / max(render_every, 1)), 1),
                audio_events=audio_events,
            )
        played = np.stack(played_roll, axis=0) if played_roll else np.zeros((0, 88), dtype=np.float32)
        goal_steps = min(int(goals.shape[0]), int(played.shape[0]))
        goal_keys = min(int(goals.shape[-1]), int(played.shape[-1]), 88)
        state_steps = min(int(piano_states.shape[0]), int(played.shape[0]))
        state_keys = min(int(piano_states.shape[-1]), int(played.shape[-1]), 88)
        against_goals = key_metrics(goals[:goal_steps, :goal_keys], played[:goal_steps, :goal_keys], threshold=threshold)
        against_states = key_metrics(piano_states[:state_steps, :state_keys], played[:state_steps, :state_keys], threshold=threshold)
        result = {
            "label": label,
            "song_name": song_name,
            "environment_name": env_name,
            "playback_mode": "recorded_rp1m_hand_joints_and_piano_states",
            "audio_source": "recorded_rp1m_piano_state_key_transitions",
            "midi_proto_path": str(midi_proto_path),
            "hand_joints_shape": list(hand_joints.shape),
            "piano_states_shape": list(piano_states.shape),
            "steps_rendered": int(steps),
            "restored_hand_joint_count": int(restored_hand_joint_count),
            "rendered_frames": int(len(frames)),
            "render_every": int(render_every),
            "render_error": render_error,
            "video_path": str(video_path) if video_path is not None else None,
            "video_format": video_format,
            "audio_warning": audio_warning,
            "audio_midi_note_event_count": int(len(_note_midi_events(audio_events))),
            "against_goals": {
                **against_goals,
                "scored_steps": int(goal_steps),
                "scored_keys": int(goal_keys),
            },
            "against_rp1m_piano_states": {
                **against_states,
                "scored_steps": int(state_steps),
                "scored_keys": int(state_keys),
            },
        }
        save_json(output_dir / f"{label}_playback.json", result)
        return result
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def _default_soundfont_path() -> Path | None:
    repo_root = Path(__file__).resolve().parents[4]
    candidates = [
        repo_root / "robopianist" / "soundfonts" / "SalamanderGrandPiano.sf2",
        repo_root / "robopianist" / "soundfonts" / "TimGM6mb.sf2",
        repo_root / "robopianist" / "third_party" / "soundfonts" / "TimGM6mb.sf2",
        repo_root / "third_party" / "soundfonts" / "TimGM6mb.sf2",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            try:
                with candidate.open("rb") as handle:
                    if handle.read(4) == b"RIFF":
                        return candidate
            except OSError:
                pass
    return None


def _clone_midi_events(events: list[Any]) -> list[Any]:
    cloned = []
    for event in events:
        time_value = float(getattr(event, "time"))
        note = getattr(event, "note", None)
        velocity = getattr(event, "velocity", None)
        if note is not None and velocity is not None:
            cloned.append(type(event)(note=int(note), velocity=int(velocity), time=time_value))
        elif note is not None:
            cloned.append(type(event)(note=int(note), time=time_value))
    return cloned


def _write_waveform(path: Path, waveform: np.ndarray, sample_rate: int = 44100) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(np.asarray(waveform, dtype=np.int16).tobytes())


def _attach_keypress_audio(video_path: Path, audio_events: list[Any]) -> str | None:
    note_events = _note_midi_events(audio_events)
    if not note_events:
        return "Audio mux skipped because rollout produced no piano key press MIDI events."
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        return "Audio mux skipped because ffmpeg was not found on PATH."
    soundfont_path = _default_soundfont_path()
    if soundfont_path is None:
        return "Audio mux skipped because no valid soundfont file was found."
    try:
        from robopianist.music import synthesizer
    except Exception as exc:
        return f"Audio mux skipped because RoboPianist synthesizer import failed: {exc}"

    temp_dir = Path(tempfile.mkdtemp(prefix=f"{video_path.stem}_audio_", dir=str(video_path.parent)))
    wav_path = temp_dir / f"{video_path.stem}.wav"
    temp_video = temp_dir / video_path.name
    try:
        synth = synthesizer.Synthesizer(soundfont_path=soundfont_path)
        try:
            waveform = synth.get_samples(_clone_midi_events(note_events))
        finally:
            synth.stop()
        _write_waveform(wav_path, waveform)
        shutil.copyfile(video_path, temp_video)
        subprocess.run(
            [
                ffmpeg_path,
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(temp_video),
                "-i",
                str(wav_path),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                str(video_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return None
    except Exception as exc:
        return f"Audio mux failed: {exc}"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def rollout_reconstructed_actions_with_robopianist(
    *,
    actions: np.ndarray,
    goals: np.ndarray,
    song_name: str,
    output_dir: str | Path,
    label: str,
    control_timestep: float = 0.05,
    fps: int = 20,
    width: int = 640,
    height: int = 480,
    max_steps: int | None = None,
    render_every: int = 1,
    seed: int = 0,
) -> dict[str, Any]:
    """
    Replay a Partita action sequence in RoboPianist / DM Control and render a video.

    The MIDI target is synthesized from the RP1M goal pianoroll because this checkout may
    not include the full PIG MIDI/proto asset library. robopianist_root is expected at:
    /WAVE/projects/ECEN-524-Wi26/robopiano/robopianist
    """
    os.environ.setdefault("MUJOCO_GL", "egl")
    output_dir = ensure_dir(output_dir)
    actions = np.asarray(actions, dtype=np.float32)
    goals = np.asarray(goals)
    if actions.ndim != 2:
        raise ValueError(f"Expected actions [T, action_dim], got {actions.shape}")
    midi_proto_path = write_goals_proto(
        goals,
        output_dir / f"{label}_target_goals.proto",
        dt=control_timestep,
        title=f"Partita {label} {song_name}",
    )
    env_name, env = _load_env(
        environment_names=candidate_environment_names(song_name),
        midi_proto_path=midi_proto_path,
        control_timestep=control_timestep,
        seed=seed,
    )
    frames: list[np.ndarray] = []
    piano_roll: list[np.ndarray] = []
    total_reward = 0.0
    actions_executed = 0
    render_error = None
    terminated = False
    try:
        timestep = env.reset()
        try:
            frames.append(render_frame(env, height=height, width=width))
        except Exception as exc:
            render_error = str(exc)
        action_spec = env.action_spec()
        action_dim = int(action_spec.shape[0])
        steps = actions if max_steps is None else actions[: int(max_steps)]
        for step_index, action in enumerate(steps):
            control = _scale_action_to_spec(action, action_spec, source="normalized_minus_one_to_one")
            timestep = env.step(control)
            total_reward += float(timestep.reward or 0.0)
            actions_executed += 1
            piano_activation = _capture_piano_activation(env)
            if piano_activation is not None:
                piano_roll.append(piano_activation)
            if render_error is None and (step_index + 1) % max(int(render_every), 1) == 0:
                try:
                    frames.append(render_frame(env, height=height, width=width))
                except Exception as exc:
                    render_error = str(exc)
            if timestep.last():
                terminated = True
                break
        video_path = None
        video_format = None
        audio_warning = None
        audio_events = _find_piano_midi_events(env)
        if render_error is None:
            video_path, video_format, audio_warning = write_video(
                frames,
                output_dir / f"{label}_rollout.mp4",
                fps=max(int(fps / max(render_every, 1)), 1),
                audio_events=audio_events,
            )
        result = {
            "label": label,
            "song_name": song_name,
            "environment_name": env_name,
            "midi_proto_path": str(midi_proto_path),
            "actions_shape": list(actions.shape),
            "action_dim_environment": action_dim,
            "action_source_scale": "normalized_minus_one_to_one",
            "actions_executed": int(actions_executed),
            "terminated": bool(terminated),
            "total_reward": float(total_reward),
            "rendered_frames": int(len(frames)),
            "render_error": render_error,
            "video_path": str(video_path) if video_path is not None else None,
            "video_format": video_format,
            "audio_warning": audio_warning,
            "audio_source": "robopianist_piano_midi_keypress_events",
            "audio_midi_note_event_count": int(len(_note_midi_events(audio_events))),
            **_rollout_key_metrics(goals, piano_roll),
            **_collect_metrics(env, terminated=terminated),
        }
        save_json(output_dir / f"{label}_rollout.json", result)
        return result
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
