from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

PARTITA_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PARTITA_ROOT.parent
SRC_ROOT = PARTITA_ROOT / "src"
for import_root in [REPO_ROOT, SRC_ROOT]:
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import numpy as np

from partita.evaluation.metrics import key_metrics
from partita.evaluation.rollout import (
    candidate_environment_names,
    render_frame,
    rollout_recorded_rp1m_episode_with_robopianist,
    write_goals_proto,
    write_video,
)
from partita.utils.config import experiment_name, load_config, output_root
from partita.utils.io import ensure_dir, load_json, save_json


def _iter_wrapped_envs(env: Any) -> list[Any]:
    current = env
    wrapped = []
    seen = set()
    while current is not None and id(current) not in seen:
        wrapped.append(current)
        seen.add(id(current))
        current = getattr(current, "_environment", None)
    return wrapped


def _load_env(*, environment_name: str, midi_proto_path: Path, control_timestep: float, seed: int, reduced_action_space: bool):
    from robopianist import suite

    return suite.load(
        environment_name=environment_name,
        midi_file=midi_proto_path,
        seed=seed,
        task_kwargs={
            "control_timestep": float(control_timestep),
            "n_steps_lookahead": 1,
            "disable_colorization": False,
            "disable_hand_collisions": False,
            "reduced_action_space": bool(reduced_action_space),
        },
    )


def _piano_activation(env: Any) -> np.ndarray | None:
    for current in _iter_wrapped_envs(env):
        piano = getattr(getattr(current, "task", None), "piano", None)
        activation = getattr(piano, "activation", None)
        if activation is not None:
            return np.asarray(activation, dtype=np.float32).reshape(-1)[:88]
    return None


def _hand_joints(env: Any) -> np.ndarray | None:
    for current in _iter_wrapped_envs(env):
        task = getattr(current, "task", None)
        physics = getattr(current, "physics", None)
        if task is None or physics is None:
            continue
        values = []
        for hand_name in ("right_hand", "left_hand"):
            hand = getattr(task, hand_name, None)
            joints = getattr(hand, "joints", None)
            if joints is None:
                continue
            for joint in joints:
                try:
                    values.append(float(physics.bind(joint).qpos))
                except Exception:
                    pass
        if values:
            return np.asarray(values, dtype=np.float32)
    return None


def _midi_events(env: Any) -> list[Any]:
    for current in _iter_wrapped_envs(env):
        piano = getattr(getattr(current, "task", None), "piano", None)
        midi_module = getattr(piano, "midi_module", None)
        get_all = getattr(midi_module, "get_all_midi_messages", None)
        if callable(get_all):
            return list(get_all())
    return []


def _note_event_count(events: list[Any]) -> int:
    return sum(type(event).__name__ in {"NoteOn", "NoteOff"} for event in events)


def _active_indices(frame: np.ndarray | None, threshold: float) -> list[int]:
    if frame is None:
        return []
    return [int(i) for i in np.flatnonzero(np.asarray(frame)[:88] > threshold)]


def _extract_events(roll: np.ndarray, threshold: float) -> list[tuple[int, int, int]]:
    active = np.asarray(roll)[:, :88] > threshold
    events: list[tuple[int, int, int]] = []
    for key in range(active.shape[1]):
        start = None
        for t, is_active in enumerate(active[:, key]):
            if is_active and start is None:
                start = t
            if start is not None and (not is_active or t == active.shape[0] - 1):
                end = t + 1 if is_active and t == active.shape[0] - 1 else t
                if end > start:
                    events.append((int(key), int(start), int(end)))
                start = None
    return events


def _roll_metrics(target: np.ndarray, played: np.ndarray, threshold: float) -> dict[str, Any]:
    steps = min(int(target.shape[0]), int(played.shape[0]))
    keys = min(int(target.shape[-1]), int(played.shape[-1]), 88)
    if steps == 0 or keys == 0:
        return {
            "key_precision": 0.0,
            "key_recall": 0.0,
            "key_f1": 0.0,
            "mispress_rate": 0.0,
            "scored_steps": int(steps),
            "scored_keys": int(keys),
            "target_event_count": 0,
            "played_event_count": 0,
            "target_unique_keys": [],
            "played_unique_keys": [],
            "missed_unique_keys": [],
            "extra_unique_keys": [],
            "overlap_unique_keys": [],
        }
    metrics = key_metrics(target[:steps, :keys], played[:steps, :keys], threshold=threshold)
    target_events = _extract_events(target[:steps, :keys], threshold)
    played_events = _extract_events(played[:steps, :keys], threshold)
    target_keys = sorted({key for key, _, _ in target_events})
    played_keys = sorted({key for key, _, _ in played_events})
    return {
        **metrics,
        "scored_steps": int(steps),
        "scored_keys": int(keys),
        "target_event_count": int(len(target_events)),
        "played_event_count": int(len(played_events)),
        "target_unique_keys": target_keys,
        "played_unique_keys": played_keys,
        "missed_unique_keys": sorted(set(target_keys) - set(played_keys)),
        "extra_unique_keys": sorted(set(played_keys) - set(target_keys)),
        "overlap_unique_keys": sorted(set(target_keys).intersection(played_keys)),
    }




def _map_action_order(action: np.ndarray, *, mapping: str) -> np.ndarray:
    values = np.asarray(action, dtype=np.float32).reshape(-1)
    if values.size != 39:
        return values
    if mapping == "as_is":
        return values
    right = values[:19]
    left = values[19:38]
    sustain = values[38:39]
    if mapping == "swap_hands":
        return np.concatenate([left, right, sustain]).astype(np.float32)
    if mapping == "zero_sustain":
        return np.concatenate([right, left, np.zeros_like(sustain)]).astype(np.float32)
    if mapping == "invert_sustain":
        return np.concatenate([right, left, -sustain]).astype(np.float32)
    if mapping == "swap_hands_zero_sustain":
        return np.concatenate([left, right, np.zeros_like(sustain)]).astype(np.float32)
    raise ValueError(f"Unknown action mapping: {mapping}")

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

def _safe_restore_initial_hands(env: Any, initial_hand_joints: np.ndarray | None) -> str:
    if initial_hand_joints is None:
        return "unavailable"
    for current in _iter_wrapped_envs(env):
        task = getattr(current, "task", None)
        physics = getattr(current, "physics", None)
        if task is None or physics is None:
            continue
        joints = []
        for hand_name in ("right_hand", "left_hand"):
            hand = getattr(task, hand_name, None)
            hand_joints = getattr(hand, "joints", None)
            if hand_joints is not None:
                joints.extend(list(hand_joints))
        if not joints:
            continue
        values = np.asarray(initial_hand_joints, dtype=np.float32).reshape(-1)
        if values.size < len(joints):
            return f"incompatible: rp1m has {values.size} hand joint values, env has {len(joints)}"
        try:
            for joint, value in zip(joints, values[: len(joints)]):
                physics.bind(joint).qpos = float(value)
            if hasattr(physics, "forward"):
                physics.forward()
            return f"restored {len(joints)} hand joints"
        except Exception as exc:
            return f"failed: {exc}"
    return "env_handles_unavailable"


def _write_frame_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fields = ["mode", "step", "reward", "target_keys", "rp1m_piano_keys", "sim_keys", "action_l2", "action_max_abs"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _run_mode(
    *,
    mode: str,
    env_name: str,
    task_variant: str,
    reduced_action_space: bool,
    midi_proto_path: Path,
    actions: np.ndarray,
    goals: np.ndarray,
    rp1m_piano: np.ndarray | None,
    hand_joints: np.ndarray | None,
    output_dir: Path,
    threshold: float,
    control_timestep: float,
    seed: int,
    render_video: bool,
    width: int,
    height: int,
    fps: int,
    max_steps: int | None,
    action_mapping: str = "as_is",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env = _load_env(environment_name=env_name, midi_proto_path=midi_proto_path, control_timestep=control_timestep, seed=seed, reduced_action_space=reduced_action_space)
    sim_frames: list[np.ndarray] = []
    frame_rows: list[dict[str, Any]] = []
    video_frames: list[np.ndarray] = []
    total_reward = 0.0
    render_error = None
    terminated = False
    restore_status = "not_requested"
    try:
        timestep = env.reset()
        action_spec = env.action_spec()
        action_dim = int(action_spec.shape[0])
        if mode == "rp1m_restore_hands":
            restore_status = _safe_restore_initial_hands(env, hand_joints[0] if hand_joints is not None else None)
        if render_video:
            try:
                video_frames.append(render_frame(env, height=height, width=width))
            except Exception as exc:
                render_error = str(exc)
        steps = actions if max_steps is None else actions[: int(max_steps)]
        for step, action in enumerate(steps):
            if mode == "zero":
                control = np.zeros((action_dim,), dtype=np.float32)
            else:
                mapped_action = _map_action_order(action, mapping=action_mapping)
                control = _scale_action_to_spec(mapped_action, action_spec, source="normalized_minus_one_to_one")
            timestep = env.step(control)
            total_reward += float(timestep.reward or 0.0)
            sim = _piano_activation(env)
            if sim is not None:
                sim_frames.append(sim)
            target_frame = goals[step] if step < goals.shape[0] else np.zeros((goals.shape[-1],), dtype=np.float32)
            rp1m_frame = rp1m_piano[step] if rp1m_piano is not None and step < rp1m_piano.shape[0] else None
            frame_rows.append(
                {
                    "mode": mode,
                    "step": int(step),
                    "reward": float(timestep.reward or 0.0),
                    "target_keys": json.dumps(_active_indices(target_frame, threshold)),
                    "rp1m_piano_keys": json.dumps(_active_indices(rp1m_frame, threshold)),
                    "sim_keys": json.dumps(_active_indices(sim, threshold)),
                    "action_l2": float(np.linalg.norm(control)),
                    "action_max_abs": float(np.max(np.abs(control))) if control.size else 0.0,
                }
            )
            if render_video and render_error is None:
                try:
                    video_frames.append(render_frame(env, height=height, width=width))
                except Exception as exc:
                    render_error = str(exc)
            if timestep.last():
                terminated = True
                break
        played = np.stack(sim_frames, axis=0) if sim_frames else np.zeros((0, 88), dtype=np.float32)
        events = _midi_events(env)
        video_path = None
        video_format = None
        audio_warning = None
        if render_video and render_error is None:
            video_path, video_format, audio_warning = write_video(video_frames, output_dir / f"{mode}_playback.mp4", fps=fps, audio_events=events)
        result = {
            "mode": mode,
            "task_variant": task_variant,
            "reduced_action_space": bool(reduced_action_space),
            "environment_name": env_name,
            "restore_status": restore_status,
            "action_dim_environment": int(action_dim),
            "action_dim_rp1m": int(actions.shape[-1]),
            "action_source_scale": "normalized_minus_one_to_one",
            "action_mapping": action_mapping,
            "actions_executed": int(len(frame_rows)),
            "terminated": bool(terminated),
            "total_reward": float(total_reward),
            "midi_note_event_count": int(_note_event_count(events)),
            "render_error": render_error,
            "video_path": str(video_path) if video_path is not None else None,
            "video_format": video_format,
            "audio_warning": audio_warning,
            "against_goals": _roll_metrics(goals, played, threshold),
        }
        if rp1m_piano is not None:
            result["against_rp1m_piano_states"] = _roll_metrics(rp1m_piano, played, threshold)
        return result, frame_rows
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def _goal_vs_piano_report(goals: np.ndarray, piano_states: np.ndarray | None, threshold: float) -> dict[str, Any]:
    if piano_states is None:
        return {"available": False, "note": "target trajectory does not include piano_states"}
    return {"available": True, **_roll_metrics(goals, piano_states, threshold)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug raw RP1M action playback before validating learned primitives.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--render-video", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    config = load_config(args.config)
    exp = experiment_name(config)
    root = output_root(config)
    data_dir = root / "data" / exp
    debug_dir = ensure_dir(root / "rp1m_playback_debug" / exp)
    selection = load_json(data_dir / "selection.json")
    target = np.load(data_dir / "target_trajectory.npz")
    actions = np.asarray(target["actions"], dtype=np.float32)
    goals = np.asarray(target["goals"], dtype=np.float32)
    piano_states = np.asarray(target["piano_states"], dtype=np.float32) if "piano_states" in target.files else None
    hand_joints = np.asarray(target["hand_joints"], dtype=np.float32) if "hand_joints" in target.files else None
    threshold = float(selection.get("key_threshold", config.get("selection", {}).get("key_threshold", 0.5)))
    control_timestep = float(config.get("control_timestep", 0.05))

    midi_proto_path = write_goals_proto(goals, debug_dir / "target_goals.proto", dt=control_timestep, title=f"Partita RP1M playback debug {selection['song_name']}")
    candidates = candidate_environment_names(selection["song_name"])
    recorded_state_result = None
    if piano_states is not None and hand_joints is not None:
        recorded_state_result = rollout_recorded_rp1m_episode_with_robopianist(
            hand_joints=hand_joints,
            piano_states=piano_states,
            goals=goals,
            song_name=selection["song_name"],
            output_dir=debug_dir,
            label="rp1m_recorded_state",
            control_timestep=control_timestep,
            fps=args.fps,
            width=args.width,
            height=args.height,
            max_steps=args.max_steps,
            render_every=1,
            seed=args.seed,
            threshold=threshold,
        )

    candidate_reports = []
    all_results = []
    all_frame_rows = []
    for env_name in candidates:
        for task_variant, reduced_action_space in (("full_action_space", False), ("reduced_action_space", True)):
            try:
                env = _load_env(
                    environment_name=env_name,
                    midi_proto_path=midi_proto_path,
                    control_timestep=control_timestep,
                    seed=args.seed,
                    reduced_action_space=reduced_action_space,
                )
                try:
                    initial_piano = _piano_activation(env)
                    initial_hand = _hand_joints(env)
                    candidate_reports.append(
                        {
                            "environment_name": env_name,
                            "task_variant": task_variant,
                            "reduced_action_space": bool(reduced_action_space),
                            "loads": True,
                            "action_dim": int(env.action_spec().shape[0]),
                            "initial_piano_keys": _active_indices(initial_piano, threshold),
                            "initial_hand_joint_dim": int(initial_hand.size) if initial_hand is not None else 0,
                        }
                    )
                finally:
                    close = getattr(env, "close", None)
                    if callable(close):
                        close()
            except Exception as exc:
                candidate_reports.append(
                    {
                        "environment_name": env_name,
                        "task_variant": task_variant,
                        "reduced_action_space": bool(reduced_action_space),
                        "loads": False,
                        "error": str(exc),
                    }
                )
                continue

            mapping_modes = [("zero", "as_is")]
            for mapping in ("as_is", "swap_hands", "zero_sustain", "invert_sustain", "swap_hands_zero_sustain"):
                mapping_modes.append((f"rp1m_reset_only__{mapping}", mapping))
            for mapping in ("as_is", "swap_hands"):
                mapping_modes.append((f"rp1m_restore_hands__{mapping}", mapping))
            for mode, action_mapping in mapping_modes:
                base_mode = "zero" if mode == "zero" else ("rp1m_restore_hands" if mode.startswith("rp1m_restore_hands") else "rp1m_reset_only")
                result, rows = _run_mode(
                    mode=base_mode,
                    env_name=env_name,
                    task_variant=task_variant,
                    reduced_action_space=reduced_action_space,
                    midi_proto_path=midi_proto_path,
                    actions=actions,
                    goals=goals,
                    rp1m_piano=piano_states,
                    hand_joints=hand_joints,
                    output_dir=debug_dir,
                    threshold=threshold,
                    control_timestep=control_timestep,
                    seed=args.seed,
                    render_video=bool(args.render_video),
                    width=args.width,
                    height=args.height,
                    fps=args.fps,
                    max_steps=args.max_steps,
                    action_mapping=action_mapping,
                )
                result["mode"] = mode
                all_results.append(result)
                all_frame_rows.extend(rows)

    summary = {
        "experiment_name": exp,
        "song_name": selection["song_name"],
        "target_trajectory_id": selection.get("target_trajectory_id"),
        "threshold": threshold,
        "control_timestep": control_timestep,
        "max_steps": args.max_steps,
        "target_shapes": {name: list(target[name].shape) for name in target.files if hasattr(target[name], "shape")},
        "goal_vs_rp1m_piano_states": _goal_vs_piano_report(goals, piano_states, threshold),
        "recorded_state_playback": recorded_state_result,
        "environment_candidates": candidate_reports,
        "results": all_results,
        "interpretation_hint": "Raw RP1M reset_only or restored playback should clearly beat zero actions before primitive reconstruction is meaningful.",
    }
    save_json(debug_dir / "rp1m_playback_debug.json", summary)
    _write_frame_rows(debug_dir / "rp1m_playback_frames.csv", all_frame_rows)
    print(f"Saved RP1M playback debug summary: {debug_dir / 'rp1m_playback_debug.json'}")
    if recorded_state_result is not None:
        goals_metrics = recorded_state_result["against_goals"]
        state_metrics = recorded_state_result["against_rp1m_piano_states"]
        print(
            "recorded_state_playback: "
            f"goals_f1={goals_metrics['key_f1']} states_f1={state_metrics['key_f1']} "
            f"video={recorded_state_result.get('video_path')} audio_warning={recorded_state_result.get('audio_warning')}"
        )
    for result in all_results:
        metrics = result["against_goals"]
        print(
            f"{result['environment_name']} {result['task_variant']} {result['mode']}: "
            f"f1={metrics['key_f1']} precision={metrics['key_precision']} recall={metrics['key_recall']} "
            f"mispress={metrics['mispress_rate']} events={result['midi_note_event_count']} "
            f"restore={result['restore_status']}"
        )


if __name__ == "__main__":
    main()
