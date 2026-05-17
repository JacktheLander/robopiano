from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

from intermezzo.io import atomic_save_json, atomic_save_npz, create_unique_run_dir, filesystem_slug
from intermezzo.keys import validate_target_keys
from intermezzo.midi import load_target_keys_from_midi
from intermezzo.planner import PlannerConfig, build_intermezzo_trajectory
from intermezzo.variations_bridge import load_variations_diffusion_predictor


DEFAULT_MAESTRO_ROOT = Path("/WAVE/datasets/ccoelho_lab-jlanders/maestro-v3.0.0/maestro-v3.0.0")
REQUESTED_FULL_DIFFUSION_CHECKPOINT = Path(
    "/WAVE/datasets/ccoelho_lab-jlanders/Variations/diffusion/full/checkpoints/best.pt"
)
DEFAULT_OUTPUT_ROOT = Path("/WAVE/datasets/ccoelho_lab-jlanders/Intermezzo/evaluation")


@dataclass(frozen=True)
class PressEvent:
    time_s: float
    frame: int
    key: int


@dataclass(frozen=True)
class RolloutConfig:
    control_timestep: float = 0.05
    threshold: float = 0.5
    timing_tolerance_s: float = 0.15
    seed: int = 0
    environment_name: str = "RoboPianist-debug-TwinkleTwinkleLittleStar-v0"
    reduced_action_space: bool = True
    physics_settle_steps: int = 1


def ensure_repo_paths() -> Path:
    sys.dont_write_bytecode = True
    repo = Path(__file__).resolve().parents[3]
    for path in (
        repo / "Intermezzo" / "src",
        repo / "Variations" / "src",
        repo / "Variations",
        repo / "partita" / "src",
        repo,
    ):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    return repo


def resolve_diffusion_checkpoint(requested: str | Path) -> tuple[Path, dict[str, Any]]:
    checkpoint = Path(requested).expanduser()
    meta: dict[str, Any] = {"requested_checkpoint": str(checkpoint)}
    if checkpoint.is_file():
        resolved = checkpoint.resolve()
        meta["resolved_checkpoint"] = str(resolved)
        meta["checkpoint_resolution"] = "requested_path"
        return resolved, meta

    search_root = Path("/WAVE/datasets/ccoelho_lab-jlanders/Variations")
    candidates = sorted(
        search_root.glob("*/variations/diffusion/checkpoints/best.pt"),
        key=lambda path: path.stat().st_mtime if path.exists() else 0.0,
        reverse=True,
    )
    if candidates:
        resolved = candidates[0].resolve()
        meta["resolved_checkpoint"] = str(resolved)
        meta["checkpoint_resolution"] = "fallback_latest_variations_diffusion_best_pt"
        meta["requested_checkpoint_missing"] = True
        return resolved, meta

    raise FileNotFoundError(f"Diffusion checkpoint not found: {checkpoint}")


def select_maestro_midi(
    *,
    midi_path: str | Path | None,
    maestro_root: str | Path,
    selection: str = "shortest",
    piece_index: int = 0,
) -> tuple[Path, dict[str, Any]]:
    if midi_path:
        path = Path(midi_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"MIDI file not found: {path}")
        return path, {"midi_selection": "explicit", "midi_path": str(path)}

    root = Path(maestro_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"MAESTRO root not found: {root}")
    metadata = root / "maestro-v3.0.0.csv"
    if selection == "shortest" and metadata.is_file():
        rows: list[dict[str, str]] = []
        with metadata.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                filename = row.get("midi_filename")
                duration = row.get("duration")
                if not filename or duration is None:
                    continue
                path = root / filename
                if path.is_file():
                    rows.append({**row, "_path": str(path)})
        if rows:
            rows.sort(key=lambda row: float(row["duration"]))
            row = rows[0]
            return Path(row["_path"]).resolve(), {
                "midi_selection": "shortest_from_maestro_metadata",
                "maestro_root": str(root),
                "maestro_duration_s": float(row["duration"]),
                "canonical_composer": row.get("canonical_composer"),
                "canonical_title": row.get("canonical_title"),
                "split": row.get("split"),
            }

    files = [path for path in root.rglob("*") if path.suffix.lower() in {".mid", ".midi"}]
    files.sort(key=lambda path: path.relative_to(root).as_posix())
    if not files:
        raise RuntimeError(f"No MIDI files found under {root}")
    index = int(piece_index)
    if index < 0 or index >= len(files):
        raise IndexError(f"piece_index {index} out of range for {len(files)} MIDI files")
    return files[index].resolve(), {
        "midi_selection": "sorted_piece_index",
        "piece_index": index,
        "maestro_root": str(root),
    }


def press_events_from_roll(roll: np.ndarray, *, dt: float, threshold: float) -> list[PressEvent]:
    active = validate_target_keys(roll) > float(threshold)
    previous = np.zeros((active.shape[1],), dtype=bool)
    events: list[PressEvent] = []
    for frame, row in enumerate(active):
        onsets = np.flatnonzero(np.logical_and(row, ~previous))
        for key in onsets:
            events.append(PressEvent(time_s=float(frame) * float(dt), frame=int(frame), key=int(key)))
        previous = row
    return events


def match_press_events(
    target_events: list[PressEvent],
    played_events: list[PressEvent],
    *,
    tolerance_s: float,
) -> tuple[list[dict[str, Any]], list[PressEvent], list[PressEvent]]:
    by_key: dict[int, list[int]] = {}
    for index, event in enumerate(played_events):
        by_key.setdefault(event.key, []).append(index)

    used_played: set[int] = set()
    matches: list[dict[str, Any]] = []
    missed: list[PressEvent] = []
    for target in target_events:
        candidate_index: int | None = None
        candidate_abs_error = float("inf")
        for played_index in by_key.get(target.key, []):
            if played_index in used_played:
                continue
            played = played_events[played_index]
            abs_error = abs(float(played.time_s) - float(target.time_s))
            if abs_error <= float(tolerance_s) and abs_error < candidate_abs_error:
                candidate_index = played_index
                candidate_abs_error = abs_error
        if candidate_index is None:
            missed.append(target)
            continue
        used_played.add(candidate_index)
        played = played_events[candidate_index]
        signed_error = float(played.time_s) - float(target.time_s)
        matches.append(
            {
                "key": int(target.key),
                "target_frame": int(target.frame),
                "played_frame": int(played.frame),
                "target_time_s": float(target.time_s),
                "played_time_s": float(played.time_s),
                "signed_error_s": signed_error,
                "abs_error_s": abs(signed_error),
            }
        )

    mispresses = [event for index, event in enumerate(played_events) if index not in used_played]
    return matches, missed, mispresses


def score_rollout(
    *,
    target_keys: np.ndarray,
    played_keys: np.ndarray,
    dt: float,
    threshold: float,
    timing_tolerance_s: float,
) -> dict[str, Any]:
    target = validate_target_keys(target_keys) > float(threshold)
    played = validate_target_keys(played_keys) > float(threshold)
    steps = min(target.shape[0], played.shape[0])
    target = target[:steps]
    played = played[:steps]

    tp = int(np.logical_and(target, played).sum())
    fp = int(np.logical_and(~target, played).sum())
    fn = int(np.logical_and(target, ~played).sum())
    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    f1 = float(2.0 * precision * recall / max(precision + recall, 1e-12))

    target_events = press_events_from_roll(target.astype(np.float32), dt=dt, threshold=0.5)
    played_events = press_events_from_roll(played.astype(np.float32), dt=dt, threshold=0.5)
    matches, missed, mispresses = match_press_events(
        target_events,
        played_events,
        tolerance_s=timing_tolerance_s,
    )
    abs_errors = np.asarray([row["abs_error_s"] for row in matches], dtype=np.float64)
    signed_errors = np.asarray([row["signed_error_s"] for row in matches], dtype=np.float64)

    return {
        "scored_steps": int(steps),
        "scored_keys": 88,
        "frame_true_positives": tp,
        "frame_false_positives": fp,
        "frame_false_negatives": fn,
        "frame_precision": precision,
        "frame_recall": recall,
        "frame_f1": f1,
        "target_press_events": int(len(target_events)),
        "played_press_events": int(len(played_events)),
        "matched_press_events": int(len(matches)),
        "missed_key_presses": int(len(missed)),
        "mispresses": int(len(mispresses)),
        "timing_tolerance_s": float(timing_tolerance_s),
        "timing_abs_error_mean_s": float(abs_errors.mean()) if abs_errors.size else None,
        "timing_abs_error_median_s": float(np.median(abs_errors)) if abs_errors.size else None,
        "timing_abs_error_p95_s": float(np.percentile(abs_errors, 95)) if abs_errors.size else None,
        "timing_signed_error_mean_s": float(signed_errors.mean()) if signed_errors.size else None,
        "timing_signed_error_median_s": float(np.median(signed_errors)) if signed_errors.size else None,
        "matches": matches[:200],
        "missed_events": [asdict(event) for event in missed[:200]],
        "mispress_events": [asdict(event) for event in mispresses[:200]],
    }


def _actuator_target_indices(task: Any) -> list[tuple[int, ...]]:
    joint_handles: list[Any] = []
    actuator_handles: list[Any] = []
    for hand_name in ("right_hand", "left_hand"):
        hand = getattr(task, hand_name)
        joint_handles.extend(list(getattr(hand, "joints")))
        actuator_handles.extend(list(getattr(hand, "actuators")))

    joint_by_name: dict[str, int] = {}
    for index, joint in enumerate(joint_handles):
        name = str(getattr(joint, "name", ""))
        full = str(getattr(joint, "full_identifier", name))
        joint_by_name[name] = index
        joint_by_name[full] = index

    indices: list[tuple[int, ...]] = []
    for actuator in actuator_handles:
        joint_ref = getattr(actuator, "joint", None)
        candidates = [str(joint_ref), str(getattr(joint_ref, "name", "")), str(getattr(joint_ref, "full_identifier", ""))]
        actuator_name = str(getattr(actuator, "name", ""))
        if actuator_name.startswith("A_"):
            candidates.append(actuator_name[2:])
        if "_A_" in actuator_name:
            prefix, raw = actuator_name.split("_A_", 1)
            candidates.append(f"{prefix}_{raw}")
            candidates.append(raw)
        else:
            candidates.append(actuator_name)
        found = None
        for candidate in candidates:
            if candidate in joint_by_name:
                found = joint_by_name[candidate]
                break
        if found is not None:
            indices.append((found,))
            continue

        tendon_match = None
        if "_A_" in actuator_name and actuator_name.endswith("J0"):
            prefix, raw = actuator_name.split("_A_", 1)
            joint_prefix = f"{prefix}_{raw[:-1]}"
            j1 = joint_by_name.get(f"{joint_prefix}1")
            j2 = joint_by_name.get(f"{joint_prefix}2")
            if j1 is not None and j2 is not None:
                tendon_match = (j2, j1)
        indices.append(tendon_match or ())
    return indices


def _targets_to_action(hand_targets: np.ndarray, action_spec: Any, actuator_indices: list[tuple[int, ...]]) -> np.ndarray:
    target = np.asarray(hand_targets, dtype=np.float32).reshape(-1)
    action = np.zeros(tuple(action_spec.shape), dtype=action_spec.dtype)
    hand_action_dim = min(len(actuator_indices), action.size - 1)
    for out_index in range(hand_action_dim):
        target_indices = actuator_indices[out_index]
        if target_indices and all(target_index < target.size for target_index in target_indices):
            action[out_index] = float(np.sum(target[list(target_indices)]))
    action[-1] = 0.0
    return np.clip(action, action_spec.minimum, action_spec.maximum).astype(action_spec.dtype, copy=False)


def _refresh_piano_activation(piano: Any, physics: Any) -> None:
    update_key_state = getattr(piano, "_update_key_state", None)
    if callable(update_key_state):
        update_key_state(physics)
    update_key_color = getattr(piano, "_update_key_color", None)
    if callable(update_key_color):
        update_key_color(physics)


def rollout_hand_targets_headless(
    *,
    hand_targets: np.ndarray,
    target_keys: np.ndarray,
    output_dir: str | Path,
    label: str,
    config: RolloutConfig,
) -> dict[str, Any]:
    ensure_repo_paths()
    from partita.evaluation.rollout import (
        _capture_piano_activation,
        _load_env,
        _locate_task_physics_piano,
        _set_reduced_hand_qpos,
        candidate_environment_names,
        write_goals_proto,
    )

    os.environ.setdefault("MUJOCO_GL", "egl")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    targets = np.asarray(hand_targets, dtype=np.float32)
    keys = validate_target_keys(target_keys)
    if targets.ndim != 2:
        raise ValueError(f"hand_targets must be [T, joints], got {targets.shape}")
    steps = min(int(targets.shape[0]), int(keys.shape[0]))

    midi_proto = write_goals_proto(
        keys[:steps],
        output / f"{label}_target_goals.proto",
        dt=float(config.control_timestep),
        title=f"Intermezzo online eval {label}",
    )
    env_name, env, load_info = _load_env(
        environment_names=candidate_environment_names(config.environment_name),
        midi_proto_path=midi_proto,
        control_timestep=float(config.control_timestep),
        seed=int(config.seed),
        reduced_action_space=bool(config.reduced_action_space),
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

    piano_roll: list[np.ndarray] = []
    try:
        env.reset()
        task, physics, piano = _locate_task_physics_piano(env)
        action_spec = env.action_spec()
        zero_action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        restored_hand_joint_count = 0
        pose_frames_applied = 0
        physics_steps_applied = 0
        terminated = False
        for step in range(steps):
            restored_hand_joint_count = _set_reduced_hand_qpos(task, physics, targets[step])
            if hasattr(physics, "forward"):
                physics.forward()
            for _ in range(max(int(config.physics_settle_steps), 0)):
                timestep = env.step(zero_action)
                physics_steps_applied += 1
                if timestep.last():
                    terminated = True
                    break
            _refresh_piano_activation(piano, physics)
            pose_frames_applied += 1
            activation = _capture_piano_activation(env)
            if activation is not None:
                piano_roll.append(np.asarray(activation[:88], dtype=np.float32))
            if terminated:
                break
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    played = np.stack(piano_roll, axis=0) if piano_roll else np.zeros((0, 88), dtype=np.float32)
    score = score_rollout(
        target_keys=keys[: played.shape[0]],
        played_keys=played,
        dt=float(config.control_timestep),
        threshold=float(config.threshold),
        timing_tolerance_s=float(config.timing_tolerance_s),
    )
    atomic_save_npz(output / f"{label}_online_rollout.npz", target_keys=keys[: played.shape[0]], played_keys=played)
    result = {
        "label": label,
        "environment_name": env_name,
        "midi_proto_path": str(midi_proto),
        "load_info": load_info,
        "hand_targets_shape": list(targets.shape),
        "target_keys_shape": list(keys.shape),
        "played_keys_shape": list(played.shape),
        "control_mode": "direct_hand_qpos_pose_injection",
        "action_dim": None,
        "mapped_actuators": 0,
        "unmapped_actuators": 0,
        "actions_executed": 0,
        "pose_frames_applied": int(pose_frames_applied),
        "physics_steps_applied": int(physics_steps_applied),
        "restored_hand_joint_count": int(restored_hand_joint_count),
        "terminated": bool(terminated),
        "rollout_config": asdict(config),
        "score": score,
    }
    atomic_save_json(output / f"{label}_online_rollout.json", result)
    return result


def evaluate_variations_vs_intermezzo(
    *,
    midi_path: str | Path | None = None,
    maestro_root: str | Path = DEFAULT_MAESTRO_ROOT,
    midi_selection: str = "shortest",
    piece_index: int = 0,
    checkpoint: str | Path = REQUESTED_FULL_DIFFUSION_CHECKPOINT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    control_timestep: float = 0.05,
    max_steps: int | None = None,
    max_duration_s: float | None = None,
    batch_size: int = 256,
    diffusion_steps: int | None = None,
    device: str = "auto",
    seed: int = 0,
    threshold: float = 0.5,
    timing_tolerance_s: float = 0.15,
) -> dict[str, Any]:
    ensure_repo_paths()
    midi, midi_meta = select_maestro_midi(
        midi_path=midi_path,
        maestro_root=maestro_root,
        selection=midi_selection,
        piece_index=piece_index,
    )
    target_keys, quantization_meta = load_target_keys_from_midi(
        midi,
        control_timestep=float(control_timestep),
        max_steps=max_steps,
        max_duration_s=max_duration_s,
    )
    checkpoint_path, checkpoint_meta = resolve_diffusion_checkpoint(checkpoint)
    predictor = load_variations_diffusion_predictor(
        checkpoint_path,
        device=str(device),
        diffusion_steps=diffusion_steps,
    )

    run_name = (
        f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%SZ')}_"
        f"{filesystem_slug(Path(midi).stem)}"
    )
    run_dir = create_unique_run_dir(output_root, run_name=run_name)

    variations_hand = predictor.predict(target_keys, batch_size=int(batch_size))
    planner_config = PlannerConfig(control_timestep=float(control_timestep), threshold=float(threshold))
    planned = build_intermezzo_trajectory(
        target_keys,
        predictor=lambda keys: predictor.predict(keys, batch_size=int(batch_size)),
        config=planner_config,
        batch_size=int(batch_size),
    )

    rollout_config = RolloutConfig(
        control_timestep=float(control_timestep),
        threshold=float(threshold),
        timing_tolerance_s=float(timing_tolerance_s),
        seed=int(seed),
    )
    variations = rollout_hand_targets_headless(
        hand_targets=variations_hand,
        target_keys=target_keys,
        output_dir=run_dir,
        label="variations_direct",
        config=rollout_config,
    )
    intermezzo = rollout_hand_targets_headless(
        hand_targets=planned.planned_hand_joints,
        target_keys=target_keys,
        output_dir=run_dir,
        label="intermezzo_planner",
        config=rollout_config,
    )

    atomic_save_npz(
        run_dir / "model_hand_targets.npz",
        target_keys=target_keys,
        variations_hand_joints=variations_hand,
        intermezzo_planned_hand_joints=planned.planned_hand_joints,
        intermezzo_planned_hand_joints_dense=planned.planned_hand_joints_dense,
        intermezzo_planned_hand_velocities_dense=planned.planned_hand_velocities_dense,
        intermezzo_waypoint_frames=planned.waypoint_frames,
        intermezzo_waypoint_hand_joints=planned.waypoint_hand_joints,
    )
    summary = {
        "run_dir": str(run_dir),
        "midi_path": str(midi),
        "midi": midi_meta,
        "midi_quantization": quantization_meta,
        **checkpoint_meta,
        "control_timestep": float(control_timestep),
        "max_steps": max_steps,
        "max_duration_s": max_duration_s,
        "batch_size": int(batch_size),
        "device": str(predictor.device),
        "diffusion_steps": int(predictor.diffusion_steps),
        "target_keys_shape": list(target_keys.shape),
        "intermezzo_metadata": planned.metadata,
        "models": {
            "variations_direct": variations,
            "intermezzo_planner": intermezzo,
        },
    }
    atomic_save_json(run_dir / "summary.json", summary)
    return summary
