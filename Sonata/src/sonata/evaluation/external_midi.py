from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from sonata.data.loading import load_manifest
from sonata.data.score import dumps_score_context, load_note_events, score_context_from_roll
from sonata.data.schema import ScoreEvent
from sonata.diffusion.dataset import load_diffusion_inputs, resample_sequence
from sonata.evaluation.offline import resample_prediction
from sonata.evaluation.rollout import (
    _build_video_caption,
    _collect_musical_metrics,
    _find_piano_midi_events,
    _relative_output_path,
    _render_frame,
    _safe_close,
    _safe_filename,
    _summarize_rollout_results,
    _write_rollout_video,
)
from sonata.evaluation.task_config import build_rollout_task_kwargs, validate_rollout_action_dim
from sonata.models.pipeline import Sonata3Pipeline
from sonata.transformer.decode import decode_factored_outputs
from sonata.transformer.dataset import build_goal_context, build_history_context, planner_collate_fn
from sonata.utils.io import write_json, write_table
from sonata.utils.robopianist import ensure_local_robopianist_on_path, format_robopianist_import_error
from sonata.utils.wandb_eval import log_prefixed_metrics, log_rollout_table, log_rollout_video

LOGGER = logging.getLogger(__name__)
DEFAULT_EXTERNAL_ENVIRONMENT = "RoboPianist-debug-TwinkleTwinkleRousseau-v0"


@dataclass(slots=True)
class ExternalReferenceStats:
    motion_energy_median: float
    start_state_norm_median: float
    end_state_norm_median: float
    seed_primitive_index: int
    seed_family_index: int
    seed_duration_bucket: int
    seed_dynamics_bucket: int
    seed_primitive_id: str


def evaluate_external_midi_benchmark(
    *,
    primitive_root: Path,
    diffusion_checkpoint: Path,
    output_root: Path,
    benchmark_manifest: str | Path | None = None,
    benchmark_root: str | Path | None = None,
    benchmark_split: str = "test",
    variant: str | None = None,
    limit_episodes: int | None = None,
    device: str = "cpu",
    environment_name: str = DEFAULT_EXTERNAL_ENVIRONMENT,
    render_video: bool = False,
    video_fps: int = 20,
    video_height: int = 480,
    video_width: int = 640,
    max_render_episodes: int | None = None,
    wandb_run: Any | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    logger = logger or LOGGER
    output_root.mkdir(parents=True, exist_ok=True)
    ensure_local_robopianist_on_path()
    try:
        from robopianist import suite
        from robopianist.wrappers.evaluation import MidiEvaluationWrapper
    except Exception as exc:  # pragma: no cover
        payload = {"available": False, "error": format_robopianist_import_error(exc)}
        write_json(payload, output_root / "external_midi_rollout.json")
        log_prefixed_metrics(wandb_run, payload, prefix="external_midi", summary=True)
        return payload

    manifest_base = resolve_external_manifest_base(
        benchmark_manifest=benchmark_manifest,
        benchmark_root=benchmark_root,
    )
    manifest_df = load_manifest(manifest_base)
    if benchmark_split:
        manifest_df = manifest_df[manifest_df["split"].astype(str) == benchmark_split].copy()
    manifest_df = manifest_df.sort_values(["song_id", "episode_id"]).reset_index(drop=True)
    if limit_episodes is not None:
        manifest_df = manifest_df.head(int(limit_episodes)).copy()
    if manifest_df.empty:
        raise FileNotFoundError(f"No benchmark episodes found in {manifest_base} for split={benchmark_split!r}")

    pipeline = Sonata3Pipeline(
        primitive_root=primitive_root,
        diffusion_checkpoint=diffusion_checkpoint,
        device=device,
    )
    action_horizon = int(pipeline.config["action_horizon"])
    state_context_steps = int(pipeline.config["state_context_steps"])
    context_length = int(pipeline.config["context_length"])
    token_df, metadata, prior_lookup = load_diffusion_inputs(
        primitive_root,
        action_horizon=action_horizon,
        state_context_steps=state_context_steps,
        family_mapping_mode=str(pipeline.config.get("family_mapping_mode", "heuristic_stats")),
        continuous_param_names=pipeline.config.get("continuous_param_names"),
    )
    reference = compute_external_reference_stats(token_df)
    expected_hand_joint_dim = (
        int(metadata.state_dim // state_context_steps)
        if int(state_context_steps) > 0 and int(metadata.state_dim) % int(state_context_steps) == 0
        else None
    )
    rollout_task_kwargs = build_rollout_task_kwargs(
        control_timestep=float(manifest_df.iloc[0].control_timestep),
        expected_action_dim=int(metadata.action_dim),
    )

    videos_root = output_root / "videos"
    frames_tmp_root = output_root / "frames_tmp"
    results: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    rendered_episodes = 0
    for episode_index, row in enumerate(manifest_df.itertuples(index=False)):
        note_path = Path(str(row.note_path)).resolve()
        if not note_path.exists():
            results.append(
                {
                    "episode_index": episode_index,
                    "song_id": str(row.song_id),
                    "episode_id": str(row.episode_id),
                    "split": str(row.split),
                    "note_path": str(note_path),
                    "error": "Missing note_path for external MIDI benchmark episode.",
                }
            )
            continue
        try:
            events = load_note_events(
                note_path,
                control_timestep=float(row.control_timestep),
                chord_tolerance_steps=1,
                song_id=str(row.song_id),
                episode_id=str(row.episode_id),
            )
        except Exception as exc:
            results.append(
                {
                    "episode_index": episode_index,
                    "song_id": str(row.song_id),
                    "episode_id": str(row.episode_id),
                    "split": str(row.split),
                    "note_path": str(note_path),
                    "error": f"Failed to parse note events: {exc}",
                }
            )
            continue
        if not events:
            results.append(
                {
                    "episode_index": episode_index,
                    "song_id": str(row.song_id),
                    "episode_id": str(row.episode_id),
                    "split": str(row.split),
                    "note_path": str(note_path),
                    "error": "No piano-range note events found for external MIDI benchmark episode.",
                }
            )
            continue
        env = None
        should_render = render_video and (max_render_episodes is None or rendered_episodes < max_render_episodes)
        try:
            env = MidiEvaluationWrapper(
                suite.load(
                    environment_name=environment_name,
                    midi_file=note_path,
                    seed=0,
                    task_kwargs=rollout_task_kwargs | {"control_timestep": float(row.control_timestep)},
                ),
                deque_size=1,
            )
            timestep = env.reset()
            action_dim = int(env.action_spec().shape[0])
            validate_rollout_action_dim(
                actual_action_dim=action_dim,
                expected_action_dim=int(metadata.action_dim),
                environment_name=environment_name,
            )
            base_rows = build_external_segment_rows(
                events=events,
                song_id=str(row.song_id),
                episode_id=str(row.episode_id),
                control_timestep=float(row.control_timestep),
                split=str(row.split),
                reference=reference,
                num_steps=max(int(row.num_steps), max(int(event.end_step) for event in events)),
            )
            total_reward = 0.0
            actions_executed = 0
            frames: list[np.ndarray] = []
            render_error: str | None = None
            if should_render:
                try:
                    frames.append(_render_frame(env, height=video_height, width=video_width))
                except Exception as exc:
                    render_error = str(exc)
                    logger.warning("Initial render failed for external MIDI episode `%s`: %s", row.episode_id, exc)
            hand_history = [extract_hand_joints(timestep.observation, expected_dim=expected_hand_joint_dim)]
            predicted_history: list[dict[str, Any]] = []
            segments_executed = 0
            for segment_index, template_row in enumerate(base_rows):
                if timestep.last():
                    break
                current_joint = hand_history[-1]
                current_row = dict(template_row)
                current_row["start_state_norm"] = float(np.linalg.norm(current_joint))
                planner_sample = build_external_planner_sample(
                    history_rows=predicted_history,
                    current_row=current_row,
                    metadata=metadata,
                    reference=reference,
                    context_length=context_length,
                )
                planner_batch = planner_collate_fn([planner_sample], metadata=metadata)
                planner_batch = {
                    key: value.to(pipeline.device) if hasattr(value, "to") else value
                    for key, value in planner_batch.items()
                }
                predicted_tokens = predict_external_tokens(
                    pipeline=pipeline,
                    planner_batch=planner_batch,
                    metadata=metadata,
                    reference=reference,
                )
                primitive_id = str(metadata.primitive_ids[predicted_tokens["primitive_index"]])
                state_context = build_state_context(
                    hand_history=hand_history,
                    state_context_steps=state_context_steps,
                )
                diffusion_batch = {
                    **planner_batch,
                    "primitive_index": torch.tensor(
                        [predicted_tokens["primitive_index"]],
                        dtype=torch.long,
                        device=pipeline.device,
                    ),
                    "duration_bucket": torch.tensor(
                        [predicted_tokens["duration_bucket"]],
                        dtype=torch.long,
                        device=pipeline.device,
                    ),
                    "dynamics_bucket": torch.tensor(
                        [predicted_tokens["dynamics_bucket"]],
                        dtype=torch.long,
                        device=pipeline.device,
                    ),
                    "state_context": torch.from_numpy(state_context[None, :]).to(pipeline.device),
                    "action_target": torch.zeros(
                        (1, action_horizon, metadata.action_dim),
                        dtype=torch.float32,
                        device=pipeline.device,
                    ),
                    "gmr_prior": torch.from_numpy(prior_lookup[primitive_id][None, :, :]).to(pipeline.device),
                }
                predicted_chunk = pipeline.predict_batch(diffusion_batch, variant=variant)[0].detach().cpu().numpy()
                duration_steps = max(int(current_row["duration_steps"]), 1)
                segment_actions = resample_prediction(predicted_chunk, duration_steps)
                segment_joint_trace = [current_joint]
                for action in segment_actions:
                    control = np.zeros((action_dim,), dtype=np.float32)
                    usable_dim = min(control.shape[0], action.shape[0])
                    control[:usable_dim] = action[:usable_dim]
                    timestep = env.step(control)
                    total_reward += float(timestep.reward or 0.0)
                    actions_executed += 1
                    latest_joint = extract_hand_joints(timestep.observation, expected_dim=expected_hand_joint_dim)
                    hand_history.append(latest_joint)
                    segment_joint_trace.append(latest_joint)
                    if should_render and render_error is None:
                        try:
                            frames.append(_render_frame(env, height=video_height, width=video_width))
                        except Exception as exc:
                            render_error = str(exc)
                            logger.warning("Render failed for external MIDI episode `%s`: %s", row.episode_id, exc)
                    if timestep.last():
                        break
                current_row["primitive_index"] = int(predicted_tokens["primitive_index"])
                current_row["primitive_id"] = primitive_id
                current_row["primitive_family_index"] = int(predicted_tokens["family_index"])
                current_row["primitive_family"] = metadata.primitive_family_names[int(predicted_tokens["family_index"])]
                current_row["duration_bucket"] = int(predicted_tokens["duration_bucket"])
                current_row["dynamics_bucket"] = int(predicted_tokens["dynamics_bucket"])
                current_row["predicted_family_index"] = int(predicted_tokens["family_index"])
                current_row["predicted_family"] = metadata.primitive_family_names[int(predicted_tokens["family_index"])]
                current_row["predicted_primitive_index"] = int(predicted_tokens["primitive_index"])
                current_row["predicted_primitive_id"] = primitive_id
                current_row["predicted_duration_bucket"] = int(predicted_tokens["duration_bucket"])
                current_row["predicted_dynamics_bucket"] = int(predicted_tokens["dynamics_bucket"])
                current_row["raw_predicted_primitive_index"] = int(predicted_tokens["raw_primitive_index"])
                current_row["raw_predicted_primitive_id"] = str(metadata.primitive_ids[int(predicted_tokens["raw_primitive_index"])])
                current_row["decode_valid_under_family_mask"] = bool(predicted_tokens["decode_valid_under_family_mask"])
                current_row["end_state_norm"] = float(np.linalg.norm(hand_history[-1]))
                current_row["motion_energy"] = compute_motion_energy(
                    np.stack(segment_joint_trace, axis=0),
                    control_timestep=float(row.control_timestep),
                    fallback=reference.motion_energy_median,
                )
                current_row["note_path"] = str(note_path)
                current_row["planned_horizon"] = int(predicted_chunk.shape[0])
                current_row["executed_steps"] = int(max(len(segment_joint_trace) - 1, 0))
                current_row["episode_index"] = episode_index
                current_row["segment_index"] = segment_index
                predicted_history.append(current_row)
                segment_rows.append(current_row)
                segments_executed += 1
                if timestep.last():
                    break
            video_path: Path | None = None
            video_format: str | None = None
            video_warning: str | None = None
            if should_render:
                rendered_episodes += 1
                if render_error is None:
                    try:
                        audio_events = _find_piano_midi_events(env)
                        video_path, video_format, video_warning = _write_rollout_video(
                            frames=frames,
                            output_path=videos_root / f"{_safe_filename(str(row.song_id))}_{_safe_filename(str(row.episode_id))}.mp4",
                            fps=video_fps,
                            temp_root=frames_tmp_root,
                            audio_source="robot_midi" if audio_events else "none",
                            robot_midi_events=audio_events,
                            logger=logger,
                        )
                    except Exception as exc:
                        render_error = str(exc)
                        logger.warning("Video export failed for external MIDI episode `%s`: %s", row.episode_id, exc)
            metrics, metrics_note = _collect_musical_metrics(env, episode_finished=bool(timestep.last()))
            result = {
                "episode_index": episode_index,
                "song_id": str(row.song_id),
                "episode_id": str(row.episode_id),
                "split": str(row.split),
                "environment_name": environment_name,
                "note_path": str(note_path),
                "benchmark_name": "paper_aligned_external_test",
                "paper_aligned_note": (
                    "Unseen external MIDI rollout benchmark; protocol-aligned with RoboPianist online evaluation but not "
                    "numerically interchangeable with RP1M paper tables."
                ),
                "reward": float(total_reward),
                "segments_planned": int(len(base_rows)),
                "segments_executed": int(segments_executed),
                "actions_executed": int(actions_executed),
                "terminated": bool(timestep.last()),
                "render_attempted": bool(should_render),
                "rendered_frames": int(len(frames)) if should_render else 0,
                "render_error": render_error,
                "video_path": _relative_output_path(video_path, output_root),
                "video_format": video_format,
                "video_warning": video_warning,
                "metrics_note": metrics_note,
                **metrics,
            }
            results.append(result)
            if video_path is not None:
                log_rollout_video(
                    wandb_run,
                    key=f"external_midi/videos/{_safe_filename(str(row.song_id))}_{_safe_filename(str(row.episode_id))}",
                    video_path=video_path,
                    caption=_build_video_caption(result),
                    fps=video_fps,
                    logger=logger,
                )
        except Exception as exc:  # pragma: no cover
            results.append(
                {
                    "episode_index": episode_index,
                    "song_id": str(row.song_id),
                    "episode_id": str(row.episode_id),
                    "split": str(row.split),
                    "environment_name": environment_name,
                    "note_path": str(note_path),
                    "error": str(exc),
                }
            )
        finally:
            _safe_close(env)

    summary = _summarize_rollout_results(results)
    summary.update(
        {
            "benchmark_name": "paper_aligned_external_test",
            "benchmark_split": benchmark_split,
            "manifest_base": str(Path(manifest_base).resolve()),
            "paper_aligned_note": (
                "Unseen external MIDI rollout benchmark; protocol-aligned with RoboPianist online evaluation but not "
                "numerically interchangeable with RP1M paper tables."
            ),
            "offline_action_metrics_included": False,
        }
    )
    payload = {
        "available": True,
        "benchmark_name": "paper_aligned_external_test",
        "summary": summary,
        "episodes": results,
        "segments_path": _relative_output_path((output_root / "external_midi_segments.csv"), output_root),
        "videos_dir": _relative_output_path(videos_root if videos_root.exists() else None, output_root),
    }
    write_json(payload, output_root / "external_midi_rollout.json")
    results_df = pd.DataFrame(results)
    write_table(results_df, output_root / "external_midi_rollout")
    if segment_rows:
        write_table(pd.DataFrame(segment_rows), output_root / "external_midi_segments")
    log_prefixed_metrics(wandb_run, summary, prefix="external_midi/summary", summary=True)
    log_rollout_table(wandb_run, key="external_midi/episodes_table", dataframe=results_df, logger=logger)
    if segment_rows:
        log_rollout_table(
            wandb_run,
            key="external_midi/segments_table",
            dataframe=pd.DataFrame(segment_rows),
            logger=logger,
        )
    return payload


def resolve_external_manifest_base(
    *,
    benchmark_manifest: str | Path | None = None,
    benchmark_root: str | Path | None = None,
) -> Path:
    if benchmark_manifest is not None:
        return _table_base_path(Path(benchmark_manifest).resolve())
    if benchmark_root is None:
        raise ValueError("Either benchmark_manifest or benchmark_root is required for external MIDI evaluation.")
    root = Path(benchmark_root).resolve()
    if root.is_file():
        return _table_base_path(root)
    for candidate in ("external_midi_manifest", "dataset_manifest"):
        base = root / candidate
        if base.with_suffix(".parquet").exists() or base.with_suffix(".csv").exists():
            return base
    raise FileNotFoundError(f"Unable to locate an external MIDI manifest under {root}")


def compute_external_reference_stats(token_df: pd.DataFrame) -> ExternalReferenceStats:
    if "split" in token_df.columns:
        train_df = token_df[token_df["split"].astype(str) == "train"].copy()
        source = train_df if not train_df.empty else token_df.copy()
    else:
        source = token_df.copy()
    if source.empty:
        return ExternalReferenceStats(
            motion_energy_median=0.0,
            start_state_norm_median=0.0,
            end_state_norm_median=0.0,
            seed_primitive_index=0,
            seed_family_index=0,
            seed_duration_bucket=0,
            seed_dynamics_bucket=0,
            seed_primitive_id="primitive_000",
        )
    seed_row = source.iloc[0]
    if "primitive_index" in source.columns and not source["primitive_index"].mode(dropna=True).empty:
        seed_row = source[source["primitive_index"] == int(source["primitive_index"].mode(dropna=True).iloc[0])].iloc[0]
    return ExternalReferenceStats(
        motion_energy_median=_median_or_default(source, "motion_energy", 0.0),
        start_state_norm_median=_median_or_default(source, "start_state_norm", 0.0),
        end_state_norm_median=_median_or_default(source, "end_state_norm", 0.0),
        seed_primitive_index=int(seed_row.get("primitive_index", 0)),
        seed_family_index=int(seed_row.get("primitive_family_index", 0)),
        seed_duration_bucket=int(seed_row.get("duration_bucket", 0)),
        seed_dynamics_bucket=int(seed_row.get("dynamics_bucket", 0)),
        seed_primitive_id=str(seed_row.get("primitive_id", "primitive_000")),
    )


def build_external_segment_rows(
    *,
    events: list[ScoreEvent],
    song_id: str,
    episode_id: str,
    control_timestep: float,
    split: str,
    reference: ExternalReferenceStats,
    num_steps: int,
) -> list[dict[str, Any]]:
    piano_roll = build_event_piano_roll(events=events, num_steps=max(int(num_steps), 1))
    rows: list[dict[str, Any]] = []
    for event in events:
        rows.append(
            {
                "song_id": song_id,
                "episode_id": episode_id,
                "onset_step": int(event.onset_step),
                "end_step": int(event.end_step),
                "duration_steps": max(int(event.end_step) - int(event.onset_step), 1),
                "segment_source": "note_aligned_external",
                "score_event_id": str(event.event_id),
                "key_signature": "-".join(str(key) for key in event.key_numbers) if event.key_numbers else "none",
                "heuristic_family": heuristic_family_from_event(event),
                "motion_energy": float(reference.motion_energy_median),
                "chord_size": int(event.chord_size),
                "key_center": float(event.key_center),
                "start_state_norm": float(reference.start_state_norm_median),
                "end_state_norm": float(reference.end_state_norm_median),
                "score_context_json": dumps_score_context(score_context_from_roll(piano_roll, int(event.onset_step))),
                "control_timestep": float(control_timestep),
                "split": split,
                "note_path": "",
            }
        )
    return rows


def build_event_piano_roll(*, events: list[ScoreEvent], num_steps: int) -> np.ndarray:
    roll = np.zeros((max(int(num_steps), 1), 88), dtype=np.float32)
    for event in events:
        start = max(int(event.onset_step), 0)
        end = min(max(int(event.end_step), start + 1), roll.shape[0])
        for key in event.key_numbers:
            if 0 <= int(key) < roll.shape[1]:
                roll[start:end, int(key)] = 1.0
    return roll


def heuristic_family_from_event(event: ScoreEvent) -> str:
    if int(event.chord_size) <= 1:
        return "single"
    if int(event.chord_size) == 2:
        return "stacked"
    return "chordal"


def build_external_planner_sample(
    *,
    history_rows: list[dict[str, Any]],
    current_row: dict[str, Any],
    metadata: Any,
    reference: ExternalReferenceStats,
    context_length: int,
) -> dict[str, Any]:
    usable_history = list(history_rows[-max(int(context_length), 1) :])
    if not usable_history:
        usable_history = [
            {
                "primitive_index": int(reference.seed_primitive_index),
                "primitive_family_index": int(reference.seed_family_index),
                "duration_bucket": int(reference.seed_duration_bucket),
                "dynamics_bucket": int(reference.seed_dynamics_bucket),
                "score_context_json": current_row["score_context_json"],
                "duration_steps": current_row["duration_steps"],
                "motion_energy": current_row.get("motion_energy", 0.0),
                "chord_size": current_row.get("chord_size", 0.0),
                "key_center": current_row.get("key_center", 0.0),
                "start_state_norm": current_row.get("start_state_norm", 0.0),
                "end_state_norm": current_row.get("start_state_norm", 0.0),
            }
        ]
    return {
        "primitive_history": np.asarray([int(row["primitive_index"]) for row in usable_history], dtype=np.int64),
        "family_history": np.asarray([int(row["primitive_family_index"]) for row in usable_history], dtype=np.int64),
        "duration_history": np.asarray([int(row["duration_bucket"]) for row in usable_history], dtype=np.int64),
        "dynamics_history": np.asarray([int(row["dynamics_bucket"]) for row in usable_history], dtype=np.int64),
        "history_context": np.stack(
            [build_history_context(pd.Series(row)) for row in usable_history],
            axis=0,
        ).astype(np.float32),
        "planner_context": build_goal_context(pd.Series(current_row)).astype(np.float32),
        "target_primitive": int(usable_history[-1]["primitive_index"]),
        "target_family": int(usable_history[-1]["primitive_family_index"]),
        "target_duration": int(usable_history[-1]["duration_bucket"]),
        "target_dynamics": int(usable_history[-1]["dynamics_bucket"]),
        "target_params": np.zeros((metadata.continuous_param_dim,), dtype=np.float32),
    }


def predict_external_tokens(
    *,
    pipeline: Sonata3Pipeline,
    planner_batch: dict[str, torch.Tensor],
    metadata: Any,
    reference: ExternalReferenceStats,
) -> dict[str, int]:
    if pipeline.planner is None:
        return {
            "family_index": int(reference.seed_family_index),
            "primitive_index": int(reference.seed_primitive_index),
            "duration_bucket": int(reference.seed_duration_bucket),
            "dynamics_bucket": int(reference.seed_dynamics_bucket),
            "raw_primitive_index": int(reference.seed_primitive_index),
            "decode_valid_under_family_mask": True,
        }
    with torch.no_grad():
        outputs = pipeline.planner(planner_batch)
    decoded = decode_factored_outputs(
        outputs,
        family_mask=pipeline.planner.family_primitive_mask,
        temperature=1.0,
    )
    family_index = int(decoded["predicted_family"].item())
    primitive_index = int(decoded["predicted_primitive"].item())
    duration_bucket = int(decoded["predicted_duration"].item())
    dynamics_bucket = int(decoded["predicted_dynamics"].item())
    raw_primitive_index = int(decoded["raw_predicted_primitive"].item())
    decode_valid_under_family_mask = bool(decoded["raw_decode_valid_under_family_mask"].item())
    family_index = int(np.clip(family_index, 0, metadata.num_families - 1))
    primitive_index = int(np.clip(primitive_index, 0, metadata.num_primitives - 1))
    duration_bucket = int(np.clip(duration_bucket, 0, metadata.num_duration_buckets - 1))
    dynamics_bucket = int(np.clip(dynamics_bucket, 0, metadata.num_dynamics_buckets - 1))
    raw_primitive_index = int(np.clip(raw_primitive_index, 0, metadata.num_primitives - 1))
    return {
        "family_index": family_index,
        "primitive_index": primitive_index,
        "duration_bucket": duration_bucket,
        "dynamics_bucket": dynamics_bucket,
        "raw_primitive_index": raw_primitive_index,
        "decode_valid_under_family_mask": decode_valid_under_family_mask,
    }


def extract_hand_joints(observation: dict[str, Any], expected_dim: int | None = None) -> np.ndarray:
    if not isinstance(observation, dict):
        raise TypeError("RoboPianist observation must be a mapping to extract hand joints.")
    keys = [
        key
        for key in observation.keys()
        if str(key).endswith("/joints_pos")
    ]
    if not keys:
        raise KeyError("No `*/joints_pos` observations were available to build state context.")
    preferred_order = ["rh_shadow_hand/joints_pos", "lh_shadow_hand/joints_pos"]
    ordered = [key for key in preferred_order if key in keys] + sorted(key for key in keys if key not in preferred_order)
    joint_vectors = [np.asarray(observation[key], dtype=np.float32).reshape(-1) for key in ordered]
    concatenated = np.concatenate(
        joint_vectors,
        axis=0,
    ).astype(np.float32)
    if expected_dim is None or concatenated.shape[0] == int(expected_dim):
        return concatenated
    if int(expected_dim) <= 0:
        raise ValueError(f"expected_dim must be positive when provided, got {expected_dim}.")
    # Some RoboPianist forks expose extra hand qpos entries. Trim symmetrically per hand when possible.
    if len(joint_vectors) == 2 and int(expected_dim) % 2 == 0:
        per_hand = int(expected_dim) // 2
        if all(vector.shape[0] >= per_hand for vector in joint_vectors):
            return np.concatenate([vector[:per_hand] for vector in joint_vectors], axis=0).astype(np.float32)
    if concatenated.shape[0] < int(expected_dim):
        raise ValueError(
            f"Observed hand joint vector has dim {concatenated.shape[0]}, smaller than expected {expected_dim}."
        )
    return concatenated[: int(expected_dim)].astype(np.float32)




def build_state_context(*, hand_history: list[np.ndarray], state_context_steps: int) -> np.ndarray:
    if not hand_history:
        raise ValueError("hand_history must contain at least one hand joint frame.")
    stacked = np.stack(hand_history[-max(int(state_context_steps), 1) :], axis=0).astype(np.float32)
    resampled = resample_sequence(stacked, max(int(state_context_steps), 1))
    return resampled.reshape(-1).astype(np.float32)


def compute_motion_energy(joints: np.ndarray, *, control_timestep: float, fallback: float) -> float:
    array = np.asarray(joints, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] <= 1:
        return float(fallback)
    velocity = np.gradient(array, max(float(control_timestep), 1e-6), axis=0).astype(np.float32)
    return float(np.linalg.norm(velocity, axis=1).mean())


def _median_or_default(frame: pd.DataFrame, column: str, default: float) -> float:
    if column not in frame.columns:
        return float(default)
    series = pd.to_numeric(frame[column], errors="coerce").dropna()
    if series.empty:
        return float(default)
    return float(series.median())


def _table_base_path(path: Path) -> Path:
    if path.suffix.lower() in {".csv", ".parquet"}:
        return path.with_suffix("")
    return path
