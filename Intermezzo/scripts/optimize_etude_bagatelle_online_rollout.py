from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime, timezone
import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO = Path("/WAVE/projects/ECEN-524-Wi26/robopiano")
for path in [REPO / "Bagatelle/src", REPO / "Intermezzo/src", REPO / "partita/src", REPO / "Variations/src", REPO / "Variations", REPO]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from bagatelle.config import BagatelleConfig  # noqa: E402
from bagatelle.planner import plan_target_keys  # noqa: E402
from intermezzo.io import atomic_save_json, atomic_save_npz, create_unique_run_dir  # noqa: E402
from intermezzo.midi import load_target_keys_from_midi  # noqa: E402
from intermezzo.online_eval import RolloutConfig, score_rollout  # noqa: E402
from intermezzo.planner import PlannerConfig, plan_between_waypoints  # noqa: E402
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


DEFAULT_MAESTRO_ROOT = Path("/WAVE/datasets/ccoelho_lab-jlanders/maestro-v3.0.0/maestro-v3.0.0")
DEFAULT_OUTPUT_ROOT = Path("/WAVE/datasets/ccoelho_lab-jlanders/Intermezzo/optimization")


def start_config() -> PlannerConfig:
    return PlannerConfig(
        control_timestep=0.05,
        threshold=0.5,
        interpolation_substeps=10,
        press_approach_s=0.05,
        press_hold_s=0.03,
        press_release_s=0.05,
        press_envelope_power=1.5,
        press_depth=0.005,
        clearance_height=0.03,
        lift_fraction=0.20,
        descent_fraction=0.35,
        enable_key_magnetism=True,
        magnet_radius=0.12,
        magnet_sigma=0.06,
        magnet_gain=1.5,
        magnet_max_xy_step=0.01,
        magnet_start_fraction=0.40,
        magnet_power=1.0,
        ik_damping=1e-2,
        ik_max_delta_q=0.04,
        ik_iterations_per_frame=2,
        preserve_waypoint_endpoints=True,
        magnet_only_final_keyset=True,
    )


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")

    midi_path, midi_meta = select_etude(args.maestro_root, args.midi_path)
    run_name = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%SZ')}_etude_bagatelle_intermezzo_200hz_force_opt"
    run_dir = create_unique_run_dir(Path(args.output_root).expanduser(), run_name=run_name)

    target_keys, quant_meta = load_target_keys_from_midi(
        midi_path,
        control_timestep=0.05,
        max_duration_s=args.max_duration_s,
        max_steps=args.max_steps,
    )
    bagatelle = plan_target_keys(
        target_keys,
        config=BagatelleConfig(
            control_timestep=0.05,
            threshold=0.5,
            seed=0,
            environment_name="RoboPianist-debug-TwinkleTwinkleLittleStar-v0",
            ik_fingertip_weight=5.0,
            ik_smoothness_weight=0.01,
            ik_neutral_weight=0.002,
            ik_max_nfev=240,
            residual_success_threshold=0.02,
            key_press_depth=0.02,
            clearance_height=0.055,
        ),
    )
    bagatelle_npz = run_dir / "bagatelle_target_hand_states.npz"
    atomic_save_npz(bagatelle_npz, **bagatelle.npz_payload())

    shared_npz = run_dir / "shared_inputs.npz"
    atomic_save_npz(
        shared_npz,
        target_keys=target_keys,
        waypoint_frames=bagatelle.waypoint_frames,
        waypoint_target_keys=bagatelle.waypoint_target_keys,
        waypoint_hand_joints=bagatelle.waypoint_hand_joints,
    )

    best: dict[str, Any] | None = None
    current = start_config()
    history: list[dict[str, Any]] = []
    seen: set[str] = set()
    stale_rounds = 0

    for round_index in range(int(args.rounds)):
        candidates = propose_candidates(current, best, round_index)
        unique = []
        for config in candidates:
            key = config_key(config)
            if key not in seen:
                seen.add(key)
                unique.append(config)
        if not unique:
            break

        candidate_specs = []
        for cand_index, config in enumerate(unique):
            label = f"round{round_index:02d}_cand{cand_index:02d}"
            candidate_specs.append((str(shared_npz), asdict(config), str(run_dir / "candidates"), label))

        print(f"round={round_index} evaluating={len(candidate_specs)} jobs={args.jobs}", flush=True)
        results = evaluate_parallel(candidate_specs, max_workers=int(args.jobs))
        results.sort(key=lambda row: (-row["score"]["frame_f1"], -row["score"]["frame_recall"], row["score"]["frame_false_positives"]))
        history.extend(results)

        round_best = results[0]
        improved = best is None or round_best["score"]["frame_f1"] > best["score"]["frame_f1"] + 1e-9
        if improved:
            best = round_best
            current = config_from_dict(round_best["planner_config"])
            stale_rounds = 0
        else:
            stale_rounds += 1
            current = adjust_by_rules(config_from_dict(best["planner_config"] if best else asdict(current)), best["score"] if best else round_best["score"])

        print(
            "best "
            f"f1={best['score']['frame_f1']:.6f} "
            f"precision={best['score']['frame_precision']:.6f} "
            f"recall={best['score']['frame_recall']:.6f} "
            f"matched={best['score']['matched_press_events']}/{best['score']['target_press_events']} "
            f"candidate={best['label']}",
            flush=True,
        )
        atomic_save_json(run_dir / "optimization_history.json", {"history": history, "best": best})
        if stale_rounds >= int(args.patience):
            break

    if best is None:
        raise RuntimeError("No optimization candidates completed.")

    final = render_best(shared_npz, config_from_dict(best["planner_config"]), run_dir / "best_render")
    summary = {
        "run_dir": str(run_dir),
        "midi_path": str(midi_path),
        "midi": midi_meta,
        "midi_quantization": quant_meta,
        "max_duration_s": args.max_duration_s,
        "bagatelle_npz": str(bagatelle_npz),
        "shared_inputs_npz": str(shared_npz),
        "bagatelle_metadata": bagatelle.metadata,
        "best_candidate": best,
        "best_render": final,
        "history_count": len(history),
        "jobs": int(args.jobs),
        "rounds_requested": int(args.rounds),
    }
    atomic_save_json(run_dir / "summary.json", summary)
    print(json.dumps({"run_dir": str(run_dir), "best_f1": best["score"]["frame_f1"], "best_render": final["video_path"]}, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--midi-path", default=None)
    parser.add_argument("--maestro-root", default=str(DEFAULT_MAESTRO_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--max-duration-s", type=float, default=15.0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--jobs", type=int, default=2)
    return parser.parse_args()


def select_etude(maestro_root: str | Path, midi_path: str | None) -> tuple[Path, dict[str, Any]]:
    if midi_path:
        path = Path(midi_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path, {"selection": "explicit"}
    root = Path(maestro_root).expanduser().resolve()
    metadata = root / "maestro-v3.0.0.csv"
    rows: list[dict[str, str]] = []
    with metadata.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            title = (row.get("canonical_title") or "").lower()
            filename = row.get("midi_filename") or ""
            path = root / filename
            if ("etude" in title or "étude" in title) and path.is_file():
                rows.append({**row, "_path": str(path)})
    if not rows:
        raise RuntimeError(f"No etude entries found in {metadata}")
    rows.sort(key=lambda row: float(row["duration"]))
    row = rows[0]
    return Path(row["_path"]).resolve(), {
        "selection": "shortest_etude_from_maestro_metadata",
        "duration_s": float(row["duration"]),
        "canonical_composer": row.get("canonical_composer"),
        "canonical_title": row.get("canonical_title"),
        "midi_filename": row.get("midi_filename"),
        "split": row.get("split"),
    }


def evaluate_parallel(specs: list[tuple[str, dict[str, Any], str, str]], *, max_workers: int) -> list[dict[str, Any]]:
    if max_workers <= 1:
        return [evaluate_candidate(*spec) for spec in specs]
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(evaluate_candidate, *spec) for spec in specs]
        for future in as_completed(futures):
            results.append(future.result())
    return results


def evaluate_candidate(shared_npz: str, config_dict: dict[str, Any], output_root: str, label: str) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    data = np.load(shared_npz, allow_pickle=False)
    config = config_from_dict(config_dict)
    target_keys = np.asarray(data["target_keys"], dtype=np.float32)
    planned, _vel, _seg, _san, dense, _dense_vel, _dense_seg = plan_between_waypoints(
        total_steps=int(target_keys.shape[0]),
        waypoint_frames=np.asarray(data["waypoint_frames"], dtype=np.int64),
        waypoint_target_keys=np.asarray(data["waypoint_target_keys"], dtype=np.float32),
        waypoint_hand_joints=np.asarray(data["waypoint_hand_joints"], dtype=np.float32),
        config=config,
        return_dense=True,
    )
    del planned
    dense_dt = config.control_timestep / config.interpolation_substeps
    target_keys_dense = np.repeat(target_keys, int(config.interpolation_substeps), axis=0).astype(np.float32)
    out = Path(output_root) / label
    out.mkdir(parents=True, exist_ok=True)
    played, runtime = rollout_dense(dense, target_keys_dense, out, label, dense_dt, render=False)
    steps = min(int(played.shape[0]), int(target_keys_dense.shape[0]))
    score = score_rollout(
        target_keys=target_keys_dense[:steps],
        played_keys=played[:steps],
        dt=dense_dt,
        threshold=config.threshold,
        timing_tolerance_s=0.15,
    )
    result = {
        "label": label,
        "output_dir": str(out),
        "planner_config": asdict(config),
        "dense_control_timestep": dense_dt,
        "target_keys_dense_shape": list(target_keys_dense.shape),
        "played_keys_shape": list(played.shape),
        "played_key_nonzero_count": int(np.count_nonzero(played > config.threshold)),
        "score": score,
        "runtime": runtime,
    }
    atomic_save_npz(out / "rollout.npz", target_keys=target_keys_dense[:steps], played_keys=played[:steps])
    atomic_save_json(out / "result.json", result)
    return result


def rollout_dense(hand_states: np.ndarray, target_keys_dense: np.ndarray, output_dir: Path, label: str, dt: float, *, render: bool) -> tuple[np.ndarray, dict[str, Any]]:
    midi_proto = write_goals_proto(target_keys_dense[:, :88], output_dir / f"{label}_target_goals.proto", dt=dt, title=label)
    env_name, env, load_info = _load_env(
        environment_names=candidate_environment_names("RoboPianist-debug-TwinkleTwinkleLittleStar-v0"),
        midi_proto_path=midi_proto,
        control_timestep=dt,
        seed=0,
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
            if render and render_error is None:
                try:
                    frames.append(render_frame(env, height=480, width=640))
                except Exception as exc:  # noqa: BLE001
                    render_error = str(exc)
            if timestep.last():
                terminated = True
                break
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
    played = np.stack(played_roll, axis=0) if played_roll else np.zeros((0, 88), dtype=np.float32)
    runtime = {
        "environment_name": env_name,
        "load_info": load_info,
        "midi_proto_path": str(midi_proto),
        "terminated": bool(terminated),
        "restored_hand_joint_count": int(restored_count),
        "rendered_frames": int(len(frames)),
        "render_error": render_error,
    }
    if render and render_error is None:
        events = piano_roll_to_midi_events(played, dt=dt, threshold=0.5)
        video_path, video_format, video_audio_warning = write_video(frames, output_dir / "rollout_video.mp4", fps=200, audio_events=events)
        runtime.update({"video_path": str(video_path), "video_format": video_format, "video_audio_warning": video_audio_warning})
    return played, runtime


def render_best(shared_npz: Path, config: PlannerConfig, output_dir: Path) -> dict[str, Any]:
    data = np.load(shared_npz, allow_pickle=False)
    target_keys = np.asarray(data["target_keys"], dtype=np.float32)
    planned, velocities, segment_ids, sanitized, dense, dense_velocities, dense_segment_ids = plan_between_waypoints(
        total_steps=int(target_keys.shape[0]),
        waypoint_frames=np.asarray(data["waypoint_frames"], dtype=np.int64),
        waypoint_target_keys=np.asarray(data["waypoint_target_keys"], dtype=np.float32),
        waypoint_hand_joints=np.asarray(data["waypoint_hand_joints"], dtype=np.float32),
        config=config,
        return_dense=True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    dense_dt = config.control_timestep / config.interpolation_substeps
    target_keys_dense = np.repeat(target_keys, int(config.interpolation_substeps), axis=0).astype(np.float32)
    trajectory_path = output_dir / "trajectory.npz"
    atomic_save_npz(
        trajectory_path,
        target_keys=target_keys,
        target_keys_dense=target_keys_dense,
        waypoint_frames=np.asarray(data["waypoint_frames"], dtype=np.int64),
        waypoint_target_keys=np.asarray(data["waypoint_target_keys"], dtype=np.float32),
        waypoint_hand_joints=sanitized,
        planned_hand_joints=planned,
        planned_hand_velocities=velocities,
        planned_hand_joints_dense=dense,
        planned_hand_velocities_dense=dense_velocities,
        segment_ids=segment_ids,
        segment_ids_dense=dense_segment_ids,
    )
    played, runtime = rollout_dense(dense, target_keys_dense, output_dir, "best_200hz", dense_dt, render=True)
    score = score_rollout(target_keys=target_keys_dense[: played.shape[0]], played_keys=played, dt=dense_dt, threshold=config.threshold, timing_tolerance_s=0.15)
    rollout_path = output_dir / "rollout.npz"
    atomic_save_npz(rollout_path, target_keys=target_keys_dense[: played.shape[0]], played_keys=played)
    summary = {"trajectory_npz": str(trajectory_path), "rollout_npz": str(rollout_path), "planner_config": asdict(config), "score": score, **runtime}
    atomic_save_json(output_dir / "render_summary.json", summary)
    return summary


def propose_candidates(current: PlannerConfig, best: dict[str, Any] | None, round_index: int) -> list[PlannerConfig]:
    if round_index == 0 or best is None:
        base = current
    else:
        base = adjust_by_rules(current, best["score"])
    candidates = [base]
    for press_scale in [0.75, 1.25]:
        candidates.append(replace(base, press_depth=clip(base.press_depth * press_scale, 0.001, 0.03)))
    for magnet_scale in [0.75, 1.25]:
        candidates.append(scale_magnet(base, magnet_scale))
    candidates.append(replace(base, press_depth=clip(base.press_depth * 1.2, 0.001, 0.03), clearance_height=clip(base.clearance_height * 0.9, 0.01, 0.08)))
    candidates.append(scale_magnet(replace(base, press_depth=clip(base.press_depth * 0.85, 0.001, 0.03)), 0.85))
    return candidates


def adjust_by_rules(config: PlannerConfig, score: dict[str, Any]) -> PlannerConfig:
    precision = float(score.get("frame_precision") or 0.0)
    recall = float(score.get("frame_recall") or 0.0)
    target_nonzero = max(int(score.get("frame_true_positives", 0)) + int(score.get("frame_false_negatives", 0)), 1)
    played_nonzero = int(score.get("frame_true_positives", 0)) + int(score.get("frame_false_positives", 0))
    played_ratio = played_nonzero / target_nonzero
    out = config
    if recall < precision * 0.85:
        out = scale_magnet(out, 0.85)
        out = replace(out, press_depth=clip(out.press_depth * 0.85, 0.001, 0.03))
    if precision < recall * 0.85:
        out = scale_magnet(out, 1.2)
    if played_ratio < 0.75:
        out = replace(out, press_depth=clip(out.press_depth * 1.25, 0.001, 0.03))
    return out


def scale_magnet(config: PlannerConfig, scale: float) -> PlannerConfig:
    return replace(
        config,
        magnet_gain=clip(config.magnet_gain * scale, 0.2, 6.0),
        magnet_radius=clip(config.magnet_radius * (0.92 + 0.08 * scale), 0.04, 0.20),
        magnet_sigma=clip(config.magnet_sigma * (0.92 + 0.08 * scale), 0.02, 0.12),
        magnet_max_xy_step=clip(config.magnet_max_xy_step * scale, 0.002, 0.06),
    )


def config_from_dict(values: dict[str, Any]) -> PlannerConfig:
    valid = {field: values[field] for field in PlannerConfig.__dataclass_fields__ if field in values}
    return PlannerConfig(**valid)


def config_key(config: PlannerConfig) -> str:
    return json.dumps({k: round(v, 6) if isinstance(v, float) else v for k, v in asdict(config).items()}, sort_keys=True)


def clip(value: float, lo: float, hi: float) -> float:
    return float(min(max(float(value), lo), hi))


if __name__ == "__main__":
    main()
