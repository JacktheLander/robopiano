from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence
import csv
import json
import logging
import math
import os
import tempfile
import time

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "tin-mpl-cache"))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOGGER = logging.getLogger(__name__)
DEFAULT_DATASET_ROOT = Path("/WAVE/datasets/ccoelho_lab-jlanders/MAESTRO")
DEFAULT_OUTPUT_ROOT = Path("/WAVE/datasets/ccoelho_lab-jlanders/tin_eval")
_JSONL_NAME = "piece_metrics.jsonl"
_CSV_NAME = "piece_metrics.csv"
_SUMMARY_NAME = "summary.json"
_HISTOGRAM_NAME = "f1_histogram.png"
_CONFIG_NAME = "config_snapshot.json"


@dataclass(frozen=True)
class MaestroEvalConfig:
    dataset_root: Path = DEFAULT_DATASET_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    max_steps_per_song: int = 0
    final_eval_episodes: int = 1
    limit_songs: int | None = None
    resume: bool = True
    log_level: str = "INFO"
    seed: int = 42
    warmstart_steps: int = 5_000
    log_interval: int = 1_000
    eval_interval: int = 10_000
    batch_size: int = 256
    discount: float = 0.99
    tqdm_bar: bool = False
    replay_capacity: int = 1_000_000
    environment_name: str = "RoboPianist-debug-TwinkleTwinkleRousseau-v0"
    n_steps_lookahead: int = 10
    trim_silence: bool = False
    gravity_compensation: bool = False
    reduced_action_space: bool = False
    control_timestep: float = 0.05
    stretch_factor: float = 1.0
    shift_factor: int = 0
    wrong_press_termination: bool = False
    disable_fingering_reward: bool = False
    disable_forearm_reward: bool = False
    disable_colorization: bool = False
    disable_hand_collisions: bool = False
    disable_key_proximity_reward: bool = False
    disable_smooth_motion_reward: bool = False
    disable_anticipation_reward: bool = False
    primitive_fingertip_collisions: bool = False
    frame_stack: int = 1
    clip: bool = True
    action_reward_observation: bool = False
    device: str = "auto"
    agent_backend: str = "auto"
    utd_ratio: int = 20
    n_step_return: int = 3
    droq_hidden_dim: int = 256
    droq_dropout: float = 0.01
    droq_tau: float = 0.005
    droq_lr: float = 3e-4
    droq_min_alpha: float = 0.05
    droq_grad_clip: float = 1.0
    normalize_observations: bool = True
    normalize_rewards: bool = True
    normalizer_warmup_steps: int = 50
    observation_normalizer_clip: float = 5.0
    reward_normalizer_clip: float = 10.0
    compile_models: bool = True
    bc_checkpoint: Path | None = None
    use_mjx: bool = False
    n_mjx_envs: int = 4
    mjx_prefer_warp: bool = False
    run_intermediate_evals: bool = False


def discover_midi_files(dataset_root: Path) -> list[Path]:
    resolved_root = dataset_root.expanduser().resolve()
    if not resolved_root.exists():
        raise FileNotFoundError(f"MAESTRO dataset root does not exist: {resolved_root}")
    midi_files = [
        path
        for path in resolved_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".mid", ".midi"}
    ]
    midi_files.sort(key=lambda path: piece_id_from_path(path, resolved_root))
    return midi_files


def piece_id_from_path(midi_path: Path, dataset_root: Path) -> str:
    return midi_path.resolve().relative_to(dataset_root.resolve()).as_posix()


def append_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    serialized = json.dumps(_json_ready(row), sort_keys=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(serialized)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_jsonl_rows(path: Path, logger: logging.Logger | None = None) -> list[dict[str, Any]]:
    logger = logger or LOGGER
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("Skipping invalid JSONL line %s in %s", line_number, path)
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def materialize_outputs(rows: Sequence[dict[str, Any]], output_root: Path) -> dict[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / _CSV_NAME
    summary_path = output_root / _SUMMARY_NAME
    histogram_path = output_root / _HISTOGRAM_NAME
    _write_rows_csv(rows, csv_path)
    _write_json_atomic(build_summary(rows), summary_path)
    _render_histogram(rows, histogram_path)
    return {"csv": csv_path, "summary": summary_path, "histogram": histogram_path}


def build_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row.get("status") == "completed"]
    failed = [row for row in rows if row.get("status") != "completed"]
    f1_scores = [float(row["f1"]) for row in completed if _is_finite_number(row.get("f1"))]
    summary = {
        "generated_at": _utc_now(),
        "total_rows": int(len(rows)),
        "completed_rows": int(len(completed)),
        "failed_rows": int(len(failed)),
        "f1_count": int(len(f1_scores)),
    }
    if f1_scores:
        summary.update(
            {
                "f1_mean": float(np.mean(f1_scores)),
                "f1_median": float(np.median(f1_scores)),
                "f1_std": float(np.std(f1_scores)),
                "f1_min": float(np.min(f1_scores)),
                "f1_max": float(np.max(f1_scores)),
            }
        )
    return summary


def evaluate_piece_batch(
    *,
    dataset_root: Path,
    piece_paths: Sequence[Path],
    output_root: Path,
    runner: Callable[[Path, str], dict[str, Any]],
    resume: bool = True,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    logger = logger or LOGGER
    output_root.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_root / _JSONL_NAME
    rows = load_jsonl_rows(jsonl_path, logger=logger)
    completed_piece_ids = {
        str(row["piece_id"])
        for row in rows
        if isinstance(row, dict) and row.get("piece_id")
    }
    skipped = 0
    processed = 0

    for piece_path in piece_paths:
        piece_id = piece_id_from_path(piece_path, dataset_root)
        if resume and piece_id in completed_piece_ids:
            skipped += 1
            continue
        started_at = _utc_now()
        started_monotonic = time.time()
        try:
            payload = dict(runner(piece_path, piece_id))
            status = str(payload.pop("status", "completed"))
            error = payload.pop("error", None)
        except Exception as exc:  # pragma: no cover - covered through tests with stub runner
            payload = {}
            status = "error"
            error = str(exc)
            logger.exception("Failed to evaluate `%s`", piece_id)
        row = {
            "piece_id": piece_id,
            "midi_path": str(piece_path.resolve()),
            "status": status,
            "error": error,
            "started_at": started_at,
            "completed_at": _utc_now(),
            "elapsed_s": round(time.time() - started_monotonic, 6),
            **payload,
        }
        append_jsonl_row(jsonl_path, row)
        rows.append(row)
        completed_piece_ids.add(piece_id)
        processed += 1
        materialize_outputs(rows, output_root)

    outputs = materialize_outputs(rows, output_root)
    summary = build_summary(rows)
    summary["processed_now"] = int(processed)
    summary["skipped_existing"] = int(skipped)
    _write_json_atomic(summary, output_root / _SUMMARY_NAME)
    return {
        "jsonl": jsonl_path,
        **outputs,
        "processed_now": processed,
        "skipped_existing": skipped,
        "total_rows": len(rows),
    }


def write_config_snapshot(config: MaestroEvalConfig, output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    payload = _json_ready(
        asdict(config)
        | {
            "dataset_root": str(Path(config.dataset_root).expanduser().resolve()),
            "output_root": str(Path(config.output_root).expanduser().resolve()),
        }
    )
    path = output_root / _CONFIG_NAME
    _write_json_atomic(payload, path)
    return path


def evaluate_maestro_corpus(config: MaestroEvalConfig, logger: logging.Logger | None = None) -> dict[str, Any]:
    logger = logger or LOGGER
    if config.max_steps_per_song <= 0:
        raise ValueError("max_steps_per_song must be a positive integer.")
    dataset_root = Path(config.dataset_root).expanduser().resolve()
    output_root = Path(config.output_root).expanduser().resolve()
    write_config_snapshot(config, output_root)
    piece_paths = discover_midi_files(dataset_root)
    if config.limit_songs is not None:
        piece_paths = piece_paths[: config.limit_songs]
    logger.info("Discovered %s MIDI files under %s", len(piece_paths), dataset_root)

    def runner(piece_path: Path, piece_id: str) -> dict[str, Any]:
        del piece_id
        return _run_piece_online(piece_path=piece_path, config=config)

    payload = evaluate_piece_batch(
        dataset_root=dataset_root,
        piece_paths=piece_paths,
        output_root=output_root,
        runner=runner,
        resume=config.resume,
        logger=logger,
    )
    payload["dataset_root"] = str(dataset_root)
    payload["output_root"] = str(output_root)
    payload["num_discovered_pieces"] = int(len(piece_paths))
    return payload


def _run_piece_online(*, piece_path: Path, config: MaestroEvalConfig) -> dict[str, Any]:
    try:
        from tin.online_rl import (
            TrainArgs,
            get_env,
            get_train_env,
            initialize_agent_and_replay,
            run_eval_episodes,
            safe_close,
            train_online,
        )
    except ImportError:  # pragma: no cover
        from online_rl import (
            TrainArgs,
            get_env,
            get_train_env,
            initialize_agent_and_replay,
            run_eval_episodes,
            safe_close,
            train_online,
        )

    train_args = TrainArgs(
        seed=config.seed,
        max_steps=config.max_steps_per_song,
        warmstart_steps=config.warmstart_steps,
        log_interval=config.log_interval,
        eval_interval=config.eval_interval if config.run_intermediate_evals else 0,
        eval_episodes=config.final_eval_episodes,
        batch_size=config.batch_size,
        discount=config.discount,
        tqdm_bar=config.tqdm_bar,
        replay_capacity=config.replay_capacity,
        environment_name=config.environment_name,
        n_steps_lookahead=config.n_steps_lookahead,
        trim_silence=config.trim_silence,
        gravity_compensation=config.gravity_compensation,
        reduced_action_space=config.reduced_action_space,
        control_timestep=config.control_timestep,
        stretch_factor=config.stretch_factor,
        shift_factor=config.shift_factor,
        wrong_press_termination=config.wrong_press_termination,
        disable_fingering_reward=config.disable_fingering_reward,
        disable_forearm_reward=config.disable_forearm_reward,
        disable_colorization=config.disable_colorization,
        disable_hand_collisions=config.disable_hand_collisions,
        disable_key_proximity_reward=config.disable_key_proximity_reward,
        disable_smooth_motion_reward=config.disable_smooth_motion_reward,
        disable_anticipation_reward=config.disable_anticipation_reward,
        primitive_fingertip_collisions=config.primitive_fingertip_collisions,
        frame_stack=config.frame_stack,
        clip=config.clip,
        action_reward_observation=config.action_reward_observation,
        device=config.device,
        agent_backend=config.agent_backend,
        utd_ratio=config.utd_ratio,
        n_step_return=config.n_step_return,
        droq_hidden_dim=config.droq_hidden_dim,
        droq_dropout=config.droq_dropout,
        droq_tau=config.droq_tau,
        droq_lr=config.droq_lr,
        droq_min_alpha=config.droq_min_alpha,
        droq_grad_clip=config.droq_grad_clip,
        normalize_observations=config.normalize_observations,
        normalize_rewards=config.normalize_rewards,
        normalizer_warmup_steps=config.normalizer_warmup_steps,
        observation_normalizer_clip=config.observation_normalizer_clip,
        reward_normalizer_clip=config.reward_normalizer_clip,
        compile_models=config.compile_models,
        bc_checkpoint=config.bc_checkpoint,
        use_mjx=config.use_mjx,
        n_mjx_envs=config.n_mjx_envs,
        mjx_prefer_warp=config.mjx_prefer_warp,
    )
    env = None
    eval_env = None
    try:
        env = get_train_env(train_args, midi_file=piece_path)
        if config.run_intermediate_evals:
            eval_env = get_env(train_args, midi_file=piece_path, enable_midi_metrics=True)
        spec, agent, replay_buffer, device_info = initialize_agent_and_replay(train_args, env)
        agent, train_summary = train_online(
            args=train_args,
            env=env,
            spec=spec,
            agent=agent,
            replay_buffer=replay_buffer,
            eval_env=eval_env,
        )
        if eval_env is None:
            eval_env = get_env(train_args, midi_file=piece_path, enable_midi_metrics=True)
        eval_payload = run_eval_episodes(agent, eval_env, max(config.final_eval_episodes, 1))
        music = dict(eval_payload.get("music", {}))
        statistics = dict(eval_payload.get("statistics", {}))
        return {
            "status": "completed",
            "train_steps": int(config.max_steps_per_song),
            "precision": _optional_float(music.get("precision")),
            "recall": _optional_float(music.get("recall")),
            "f1": _optional_float(music.get("f1")),
            "sustain_precision": _optional_float(music.get("sustain_precision")),
            "sustain_recall": _optional_float(music.get("sustain_recall")),
            "sustain_f1": _optional_float(music.get("sustain_f1")),
            "episode_return": _optional_float(statistics.get("episode_return")),
            "episode_length": _optional_float(statistics.get("episode_length")),
            "train_elapsed_s": _optional_float(train_summary.get("elapsed_s")),
            "train_fps": _optional_float(train_summary.get("fps")),
            "requested_device": device_info.get("requested_device"),
            "resolved_device": device_info.get("resolved_device"),
            "agent_device_applied": bool(device_info.get("agent_device_applied")),
            "agent_device_fields": device_info.get("agent_device_fields"),
        }
    finally:
        safe_close(eval_env)
        safe_close(env)


def _write_rows_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    fieldnames = _fieldnames(rows)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=path.parent) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _csv_ready(row.get(name)) for name in fieldnames})
        temp_path = Path(handle.name)
    temp_path.replace(path)


def _render_histogram(rows: Sequence[dict[str, Any]], path: Path) -> None:
    scores = [float(row["f1"]) for row in rows if row.get("status") == "completed" and _is_finite_number(row.get("f1"))]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    if scores:
        axis.hist(scores, bins=20, range=(0.0, 1.0), color="#0b6e4f", edgecolor="#083d2b")
    else:
        axis.text(0.5, 0.5, "No completed pieces yet", ha="center", va="center", transform=axis.transAxes)
    axis.set_title("TIN MAESTRO F1 Histogram")
    axis.set_xlabel("F1 score")
    axis.set_ylabel("Piece count")
    axis.set_xlim(0.0, 1.0)
    figure.tight_layout()
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=path.parent, suffix=".png") as handle:
        temp_path = Path(handle.name)
    figure.savefig(temp_path, dpi=150)
    plt.close(figure)
    temp_path.replace(path)


def _fieldnames(rows: Sequence[dict[str, Any]]) -> list[str]:
    preferred = [
        "piece_id",
        "midi_path",
        "status",
        "error",
        "started_at",
        "completed_at",
        "elapsed_s",
        "train_steps",
        "train_elapsed_s",
        "train_fps",
        "precision",
        "recall",
        "f1",
        "sustain_precision",
        "sustain_recall",
        "sustain_f1",
        "episode_return",
        "episode_length",
        "requested_device",
        "resolved_device",
        "agent_device_applied",
        "agent_device_fields",
    ]
    extras = sorted({key for row in rows for key in row.keys() if key not in preferred})
    return preferred + extras


def _write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    serialized = json.dumps(_json_ready(payload), indent=2, sort_keys=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent) as handle:
        handle.write(serialized)
        temp_path = Path(handle.name)
    temp_path.replace(path)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _csv_ready(value: Any) -> Any:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return value


def _is_finite_number(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _optional_float(value: Any) -> float | None:
    if not _is_finite_number(value):
        return None
    return float(value)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
