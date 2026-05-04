from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sonata.data.loading import build_manifest_lookup, load_stage1_source_manifest
from sonata.evaluation.action_diagnostics import (
    compute_action_magnitude_diagnostics,
    finalize_keypress_trace,
    init_keypress_trace,
    save_action_diagnostic_artifacts,
    try_read_piano_activation,
    try_read_target_notes_total,
    update_keypress_trace,
)
from sonata.evaluation.offline import stitch_segment_predictions
from sonata.evaluation.rollout_diagnostics import (
    collect_robot_midi_since_reset,
    cumulative_prf,
    flatten_fingertip_xyz,
    flatten_key_qpos_sample,
    format_int_list,
    get_piano,
    micro_precision_recall,
    mux_video_with_wav,
    overlay_debug_text_on_frame,
    pressed_key_indices,
    resolve_ffmpeg_executable,
    save_env_introspection,
    synthesize_reference_wav,
    target_key_indices_after_step,
)
from sonata.evaluation.task_config import build_rollout_task_kwargs, validate_rollout_action_dim
from sonata.training.mjx_rollout import MJXRolloutBackend, mjx_availability
from sonata.utils.io import write_json, write_table
from sonata.utils.robopianist import ensure_local_robopianist_on_path, format_robopianist_import_error
from sonata.utils.wandb_eval import log_prefixed_metrics, log_rollout_table, log_rollout_video

LOGGER = logging.getLogger(__name__)
_RENDER_BACKEND_HINT = (
    "Headless DM Control rendering may require MUJOCO_GL=egl or MUJOCO_GL=osmesa before launching the job."
)


def evaluate_dm_control_rollout(
    *,
    primitive_root: Path,
    predictions_by_episode: dict[str, list[dict[str, Any]]],
    output_root: Path,
    limit_episodes: int = 2,
    render_video: bool = False,
    video_fps: int = 20,
    video_height: int = 480,
    video_width: int = 640,
    max_render_episodes: int | None = None,
    environment_name_override: str | None = None,
    prefer_manifest_midi: bool = True,
    midi_root: Path | None = None,
    wandb_run: Any | None = None,
    logger: logging.Logger | None = None,
    action_source: str = "unknown",
    control_timestep: float = 0.05,
    rollout_mode: str = "policy",
    audio_source: str = "none",
    debug_overlay: bool = False,
) -> dict[str, Any]:
    logger = logger or LOGGER
    ensure_local_robopianist_on_path()
    try:
        from robopianist import suite
        from robopianist.wrappers.evaluation import MidiEvaluationWrapper
    except Exception as exc:  # pragma: no cover
        result = {
            "available": False,
            "error": format_robopianist_import_error(exc),
            "action_source": action_source,
            "audio_source": audio_source,
            "rollout_mode": rollout_mode,
        }
        write_json(result, output_root / "dm_control_rollout.json")
        log_prefixed_metrics(wandb_run, result, prefix="rollout/dm_control", summary=True)
        return result

    token_df = (
        pd.read_parquet(primitive_root / "tokens" / "primitive_tokens.parquet")
        if (primitive_root / "tokens" / "primitive_tokens.parquet").exists()
        else pd.read_csv(primitive_root / "tokens" / "primitive_tokens.csv")
    )
    source_manifest = load_stage1_source_manifest(primitive_root)
    action_series = pd.to_numeric(source_manifest.get("action_dim"), errors="coerce") if "action_dim" in source_manifest.columns else pd.Series(dtype=float)
    if not action_series.dropna().empty:
        expected_action_dim = int(action_series.dropna().iloc[0])
    elif predictions_by_episode:
        sample_chunks = next(iter(predictions_by_episode.values()))
        expected_action_dim = int(
            stitch_segment_predictions(token_df, sample_chunks or [], 1).shape[-1]
            if sample_chunks
            else 39
        )
    else:
        expected_action_dim = 39
    rollout_task_kwargs = build_rollout_task_kwargs(
        control_timestep=float(control_timestep), expected_action_dim=expected_action_dim
    )
    manifest_lookup = _load_rollout_manifest_lookup(primitive_root=primitive_root, logger=logger)
    output_root.mkdir(parents=True, exist_ok=True)
    videos_root = output_root / "videos"
    frames_tmp_root = output_root / "frames_tmp"
    results: list[dict[str, Any]] = []
    rendered_episodes = 0
    ffmpeg_exe = resolve_ffmpeg_executable()
    per_step_csv_path = output_root / "per_step_key_diagnostics.csv"
    env_intro_written = False
    episode_keys = sorted(predictions_by_episode.keys())[:limit_episodes]

    for episode_index, episode_id in enumerate(episode_keys):
        episode_rows = token_df[token_df["episode_id"] == episode_id].sort_values("onset_step")
        raw_song_id = str(episode_rows.iloc[0]["song_id"]) if not episode_rows.empty else str(episode_id)
        if episode_rows.empty:
            results.append({"episode_id": episode_id, "error": f"No token rows found for episode `{episode_id}`."})
            continue
        episode_predictions = list(predictions_by_episode.get(episode_id, []))
        max_step_tokens = int(episode_rows["end_step"].max())

        if rollout_mode == "zero_actions":
            stitched = np.zeros((max_step_tokens, expected_action_dim), dtype=np.float32)
            ah = min(16, max(1, max_step_tokens))
            episode_predictions = [
                {
                    "predicted": np.zeros((ah, expected_action_dim), dtype=np.float32),
                    "onset_step": 0,
                    "end_step": max_step_tokens,
                }
            ]
        else:
            if not episode_predictions:
                results.append(
                    {
                        "episode_id": episode_id,
                        "song_id": raw_song_id,
                        "error": "No diffusion predictions found for episode.",
                    }
                )
                continue
            stitched = stitch_segment_predictions(
                token_df=episode_rows,
                episode_predictions=episode_predictions,
                action_horizon=int(episode_predictions[0]["predicted"].shape[0]),
            )
        episode_diag_dir = output_root / "action_diagnostics" / _safe_filename(str(episode_id))
        pre_diag = compute_action_magnitude_diagnostics(
            stitched=stitched,
            action_source=action_source,
            episode_id=str(episode_id),
            song_id=str(episode_rows.iloc[0]["song_id"]),
            expected_action_dim=int(expected_action_dim),
            env_action_dim=None,
            control_timestep=float(control_timestep),
            episode_predictions=episode_predictions,
        )
        save_action_diagnostic_artifacts(
            output_rollout_dir=episode_diag_dir,
            diagnostics_row=pre_diag,
            stitched=stitched,
            logger=logger,
        )
        source = _resolve_rollout_source(
            song_id=raw_song_id,
            episode_id=str(episode_id),
            manifest_lookup=manifest_lookup,
            environment_name_override=environment_name_override,
            prefer_manifest_midi=prefer_manifest_midi,
            midi_root=midi_root,
            logger=logger,
        )
        should_render = render_video and (max_render_episodes is None or rendered_episodes < max_render_episodes)
        env = None
        try:
            env = MidiEvaluationWrapper(
                suite.load(
                    environment_name=source["environment_name"],
                    midi_file=source["midi_file"],
                    seed=0,
                    task_kwargs=rollout_task_kwargs,
                ),
                deque_size=1,
            )
            if not env_intro_written:
                intro_path = output_root / "env_introspection.txt"
                try:
                    save_env_introspection(env, intro_path)
                    env_intro_written = True
                except Exception as intro_exc:
                    logger.warning("env_introspection failed: %s", intro_exc)
            timestep = env.reset()
            total_reward = 0.0
            action_dim = int(env.action_spec().shape[0])
            validate_rollout_action_dim(
                actual_action_dim=action_dim,
                expected_action_dim=expected_action_dim,
                environment_name=source["environment_name"],
            )
            diag_payload = compute_action_magnitude_diagnostics(
                stitched=stitched,
                action_source=action_source,
                episode_id=str(episode_id),
                song_id=raw_song_id,
                expected_action_dim=int(expected_action_dim),
                env_action_dim=action_dim,
                control_timestep=float(rollout_task_kwargs.get("control_timestep", control_timestep)),
                episode_predictions=episode_predictions,
            )
            save_action_diagnostic_artifacts(
                output_rollout_dir=episode_diag_dir,
                diagnostics_row=diag_payload,
                stitched=stitched,
                logger=logger,
            )
            keypress_trace = init_keypress_trace()
            target_notes_total = try_read_target_notes_total(env)
            frames: list[np.ndarray] = []
            render_error: str | None = None
            per_step_rows: list[dict[str, Any]] = []
            cum_tp = cum_fp = cum_fn = 0

            if should_render:
                try:
                    init_fr = _render_frame(env, height=video_height, width=video_width)
                    if debug_overlay:
                        init_fr = overlay_debug_text_on_frame(
                            init_fr,
                            lines=[
                                "step=init",
                                f"mode={rollout_mode}",
                                f"audio={audio_source}",
                            ],
                        )
                    frames.append(init_fr)
                except Exception as exc:
                    render_error = str(exc)
                    logger.warning("Initial render failed for DM Control episode `%s`: %s", episode_id, exc)
            actions_executed = 0
            for step_idx, action in enumerate(stitched):
                control = np.zeros((action_dim,), dtype=np.float32)
                control[: min(action_dim, action.shape[0])] = action[: min(action_dim, action.shape[0])]
                timestep = env.step(control)
                step_reward = float(timestep.reward or 0.0)
                total_reward += step_reward
                actions_executed += 1

                piano = get_piano(env)
                targets = target_key_indices_after_step(env)
                pressed = pressed_key_indices(piano)
                t_set = set(targets)
                p_set = set(pressed)
                overlap = len(t_set & p_set)
                num_t = len(t_set)
                num_p = len(p_set)
                cum_tp += overlap
                cum_fp += max(num_p - overlap, 0)
                cum_fn += max(num_t - overlap, 0)
                p_step, r_step = micro_precision_recall(overlap, num_t, num_p)
                p_cum, r_cum, f1_cum = cumulative_prf(cum_tp, cum_fp, cum_fn)

                robot_step_n = 0
                if piano is not None:
                    mm = getattr(piano, "midi_module", None)
                    gl = getattr(mm, "get_latest_midi_messages", None)
                    if callable(gl):
                        try:
                            robot_step_n = len(list(gl()))
                        except Exception:
                            robot_step_n = 0

                per_step_rows.append(
                    {
                        "episode_id": str(episode_id),
                        "timestep_index": step_idx,
                        "reward": step_reward,
                        "action_max_abs": float(np.max(np.abs(control))),
                        "action_mean_abs": float(np.mean(np.abs(control))),
                        "target_key_indices": format_int_list(sorted(targets)),
                        "actual_pressed_key_indices": format_int_list(sorted(pressed)),
                        "num_target_keys": num_t,
                        "num_pressed_keys": num_p,
                        "overlap_count": overlap,
                        "precision_step": p_step,
                        "recall_step": r_step,
                        "precision_cumulative": p_cum,
                        "recall_cumulative": r_cum,
                        "f1_cumulative": f1_cum,
                        "robot_midi_events_this_step": robot_step_n,
                        "hand_fingertip_positions": flatten_fingertip_xyz(env) or "",
                        "key_qpos_sample": flatten_key_qpos_sample(piano) or "",
                    }
                )

                update_keypress_trace(
                    keypress_trace,
                    step_index=actions_executed - 1,
                    activation=try_read_piano_activation(env),
                )
                if should_render and render_error is None:
                    try:
                        fr = _render_frame(env, height=video_height, width=video_width)
                        if debug_overlay:
                            fr = overlay_debug_text_on_frame(
                                fr,
                                lines=[
                                    f"step={step_idx}",
                                    f"reward={step_reward:.4f}",
                                    f"act_max={float(np.max(np.abs(control))):.4f}",
                                    f"n_tgt={num_t} n_press={num_p} ov={overlap}",
                                    f"p_step={p_step:.3f} r_step={r_step:.3f}",
                                    f"p_cum={p_cum:.3f} r_cum={r_cum:.3f} f1={f1_cum:.3f}",
                                ],
                            )
                        frames.append(fr)
                    except Exception as exc:
                        render_error = str(exc)
                        logger.warning("Render failed for DM Control episode `%s`: %s", episode_id, exc)
                if timestep.last():
                    break

            if per_step_rows:
                pd.DataFrame(per_step_rows).to_csv(
                    per_step_csv_path,
                    mode="a",
                    header=not per_step_csv_path.exists(),
                    index=False,
                )

            metrics, metrics_note = _collect_musical_metrics(env, episode_finished=bool(timestep.last()))
            keypress_trace = finalize_keypress_trace(keypress_trace)
            robot_events_full = collect_robot_midi_since_reset(env)
            reference_midi_path = Path(source["midi_file"]) if source.get("midi_file") else None
            audio_used = audio_source
            video_path: Path | None = None
            video_format: str | None = None
            video_warning: str | None = None
            if should_render:
                rendered_episodes += 1
                if render_error is None:
                    try:
                        video_path, video_format, video_warning = _write_rollout_video(
                            frames=frames,
                            output_path=videos_root
                            / f"{_safe_filename(source['environment_name'])}_{_safe_filename(str(episode_id))}.mp4",
                            fps=video_fps,
                            temp_root=frames_tmp_root,
                            audio_source=str(audio_source),
                            reference_midi_path=reference_midi_path if reference_midi_path and reference_midi_path.exists() else None,
                            robot_midi_events=robot_events_full,
                            ffmpeg_exe=ffmpeg_exe,
                            logger=logger,
                        )
                        if audio_source == "reference_midi" and reference_midi_path is None:
                            audio_used = "none_missing_reference_path"
                        elif audio_source == "robot_midi" and not robot_events_full:
                            audio_used = "robot_midi_empty_no_mux"
                    except Exception as exc:
                        render_error = str(exc)
                        logger.warning("Video export failed for DM Control episode `%s`: %s", episode_id, exc)
            episode_result = {
                "episode_index": episode_index,
                "episode_id": episode_id,
                "song_id": raw_song_id,
                "original_song_id": source["original_song_id"],
                "environment_name": source["environment_name"],
                "midi_file": str(source["midi_file"]) if source["midi_file"] is not None else None,
                "used_manifest_midi": bool(source["used_manifest_midi"]),
                "midi_source": source.get("midi_source"),
                "environment_name_overridden": bool(source["environment_name_overridden"]),
                "resolved_midi_exists": bool(source["resolved_midi_exists"]),
                "song_id_normalized": bool(source["song_id_normalized"]),
                "reward": total_reward,
                "actions_planned": int(stitched.shape[0]),
                "actions_executed": actions_executed,
                "terminated": bool(timestep.last()),
                "render_attempted": bool(should_render),
                "rendered_frames": int(len(frames)) if should_render else 0,
                "render_error": render_error,
                "video_path": _relative_output_path(video_path, output_root),
                "video_format": video_format,
                "video_warning": video_warning,
                "metrics_note": metrics_note,
                "action_source": action_source,
                "rollout_mode": rollout_mode,
                "audio_source_requested": audio_source,
                "audio_source_used": audio_used,
                "robot_midi_message_count": len(robot_events_full),
                "per_step_key_diagnostics_csv": _relative_output_path(per_step_csv_path, output_root)
                if per_step_rows
                else None,
                "env_introspection_txt": _relative_output_path(output_root / "env_introspection.txt", output_root)
                if env_intro_written
                else None,
                "debug_overlay": bool(debug_overlay),
                "action_diagnostics_dir": _relative_output_path(episode_diag_dir, output_root),
                "target_notes_total": target_notes_total,
                "wrong_key_count": None,
                "missed_key_count": None,
                "suite_load_succeeded": True,
                **keypress_trace,
                **metrics,
            }
            results.append(episode_result)
            if video_path is not None:
                log_rollout_video(
                    wandb_run,
                    key=f"rollout/videos/{_safe_filename(source['environment_name'])}_{_safe_filename(str(episode_id))}",
                    video_path=video_path,
                    caption=_build_video_caption(episode_result),
                    fps=video_fps,
                    logger=logger,
                )
        except Exception as exc:  # pragma: no cover
            results.append(
                {
                    "episode_id": episode_id,
                    "song_id": raw_song_id,
                    "original_song_id": source["original_song_id"],
                    "environment_name": source["environment_name"],
                    "midi_file": str(source["midi_file"]) if source["midi_file"] is not None else None,
                    "used_manifest_midi": bool(source["used_manifest_midi"]),
                    "midi_source": source.get("midi_source"),
                    "environment_name_overridden": bool(source["environment_name_overridden"]),
                    "resolved_midi_exists": bool(source["resolved_midi_exists"]),
                    "song_id_normalized": bool(source["song_id_normalized"]),
                    "requested_environment_name": source["environment_name"],
                    "requested_midi_file": str(source["midi_file"]) if source["midi_file"] is not None else None,
                    "requested_midi_exists": bool(source["resolved_midi_exists"]),
                    "available_environment_hint": _available_environment_hint(str(exc)),
                    "suggestion": "Pass --environment-name with a valid generic RoboPianist environment and ensure the manifest MIDI path exists.",
                    "error": str(exc),
                    "suite_load_succeeded": False,
                    "action_source": action_source,
                    "rollout_mode": rollout_mode,
                    "audio_source_requested": audio_source,
                }
            )
        finally:
            _safe_close(env)
    summary = _summarize_rollout_results(results)
    payload = {
        "available": True,
        "videos_dir": _relative_output_path(videos_root if videos_root.exists() else None, output_root),
        "summary": summary,
        "episodes": results,
        "action_source": action_source,
        "rollout_mode": rollout_mode,
        "audio_source": audio_source,
    }
    write_json(payload, output_root / "dm_control_rollout.json")
    results_df = pd.DataFrame(results)
    write_table(results_df, output_root / "dm_control_rollout")
    _maybe_promote_action_diagnostics_flat(output_root=output_root, results=results, logger=logger)
    log_prefixed_metrics(wandb_run, summary, prefix="rollout/summary", summary=True)
    log_rollout_table(wandb_run, key="rollout/episodes_table", dataframe=results_df, logger=logger)
    return payload


def evaluate_mjx_physics(
    *,
    xml_path: Path,
    action_sequences: np.ndarray,
    output_root: Path,
) -> dict[str, Any]:
    availability = mjx_availability()
    if not availability.available:
        payload = {"available": False, "error": availability.message}
        write_json(payload, output_root / "mjx_rollout.json")
        return payload
    backend = MJXRolloutBackend(xml_path=xml_path, batch_size=int(action_sequences.shape[0]))
    backend.step(action_sequences[:, 0])
    payload = {
        "available": True,
        "batch_size": int(action_sequences.shape[0]),
        "ctrl_dim": int(action_sequences.shape[-1]),
        "qpos_shape": list(backend.qpos().shape),
        "qvel_shape": list(backend.qvel().shape),
    }
    write_json(payload, output_root / "mjx_rollout.json")
    return payload


def _load_rollout_manifest_lookup(
    *,
    primitive_root: Path,
    logger: logging.Logger,
) -> dict[tuple[str, str], dict[str, Any]]:
    try:
        manifest_df = load_stage1_source_manifest(primitive_root)
    except Exception as exc:
        logger.warning("Unable to load Stage 1 source manifest for rollout: %s", exc)
        return {}
    return build_manifest_lookup(manifest_df)


def _resolve_rollout_source(
    *,
    song_id: str,
    episode_id: str,
    manifest_lookup: dict[tuple[str, str], dict[str, Any]],
    environment_name_override: str | None,
    prefer_manifest_midi: bool,
    midi_root: Path | None,
    logger: logging.Logger,
) -> dict[str, Any]:
    manifest_row = manifest_lookup.get((song_id, episode_id), {})
    midi_file, midi_source = _manifest_note_path(manifest_row, midi_root=midi_root)
    if prefer_manifest_midi and environment_name_override:
        if midi_file is None:
            logger.warning(
                "No valid manifest MIDI path found for song `%s` episode `%s`; using generic environment `%s` without a MIDI override. "
                "Pass --midi-root if the manifest stores relative MIDI paths.",
                song_id,
                episode_id,
                environment_name_override,
            )
        return {
            "environment_name": str(environment_name_override),
            "midi_file": midi_file,
            "used_manifest_midi": midi_file is not None,
            "midi_source": midi_source,
            "environment_name_overridden": True,
            "original_song_id": str(song_id),
            "song_id_normalized": False,
            "resolved_midi_exists": bool(midi_file is not None and midi_file.exists()),
        }

    environment_name = _canonicalize_environment_name(song_id)
    song_id_normalized = environment_name != song_id
    if midi_file is None and song_id_normalized:
        logger.info(
            "Normalized rollout environment name from `%s` to `%s` for episode `%s`.",
            song_id,
            environment_name,
            episode_id,
        )
    return {
        "environment_name": environment_name,
        "midi_file": midi_file,
        "used_manifest_midi": False,
        "midi_source": midi_source,
        "environment_name_overridden": False,
        "original_song_id": str(song_id),
        "song_id_normalized": song_id_normalized,
        "resolved_midi_exists": bool(midi_file is not None and midi_file.exists()),
    }


def _manifest_note_path(manifest_row: dict[str, Any], *, midi_root: Path | None = None) -> tuple[Path | None, str | None]:
    if not manifest_row:
        return None, None
    for column in ("note_path", "midi_path", "midi_file", "source_midi", "path"):
        raw_value = manifest_row.get(column)
        text = _clean_manifest_path_value(raw_value)
        if text is None:
            continue
        path = _resolve_existing_midi_path(text, midi_root=midi_root)
        if path is not None:
            return path, column
    path = _resolve_maestro_midi_from_metadata(manifest_row, midi_root=midi_root)
    if path is not None:
        return path, "maestro_metadata"
    return None, None


def _clean_manifest_path_value(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return text


def _resolve_existing_midi_path(text: str, *, midi_root: Path | None) -> Path | None:
    raw_path = Path(text).expanduser()
    candidate_paths: list[Path] = [raw_path]
    if midi_root is not None:
        root = midi_root.expanduser().resolve()
        if not raw_path.is_absolute():
            candidate_paths.append(root / raw_path)
        candidate_paths.append(root / raw_path.name)
    for candidate in _with_midi_extension_alternatives(candidate_paths):
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        if resolved.exists() and resolved.is_file():
            return resolved
    return None


def _resolve_maestro_midi_from_metadata(manifest_row: dict[str, Any], *, midi_root: Path | None) -> Path | None:
    if midi_root is None:
        return None
    root = midi_root.expanduser().resolve()
    metadata_path = _find_maestro_metadata(root)
    if metadata_path is None:
        return None
    try:
        metadata = pd.read_csv(metadata_path)
    except Exception:
        return None
    if "midi_filename" not in metadata.columns or "canonical_title" not in metadata.columns:
        return None
    piece_key = _piece_key_from_song_id(str(manifest_row.get("song_id", "")))
    if not piece_key:
        return None
    normalized_key = _normalize_piece_text(piece_key)
    titles = metadata["canonical_title"].astype(str).map(_normalize_piece_text)
    matches = metadata[titles.str.contains(normalized_key, regex=False, na=False)].copy()
    if matches.empty:
        return None
    matches["_split_rank"] = matches.get("split", "").astype(str).map(
        {"validation": 0, "val": 0, "test": 1, "train": 2}
    ).fillna(3)
    matches = matches.sort_values(["_split_rank", "duration"] if "duration" in matches.columns else ["_split_rank"])
    for midi_filename in matches["midi_filename"].astype(str).tolist():
        path = _resolve_existing_midi_path(midi_filename, midi_root=root)
        if path is not None:
            return path
        nested_path = _resolve_existing_midi_path(midi_filename, midi_root=root / "maestro-v3.0.0")
        if nested_path is not None:
            return nested_path
    return None


def _find_maestro_metadata(root: Path) -> Path | None:
    candidates = [
        root / "maestro-v3.0.0.csv",
        root / "maestro-v3.0.0" / "maestro-v3.0.0.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _piece_key_from_song_id(song_id: str) -> str | None:
    match = re.search(r"RoboPianist-(?:repertoire-150|debug)-(.+?)-v\d+", song_id)
    if match:
        return match.group(1)
    return song_id or None


def _normalize_piece_text(value: str) -> str:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(value))
    text = re.sub(r"(No)(\d+)", r"\1 \2", text)
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _with_midi_extension_alternatives(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    expanded: list[Path] = []
    for path in paths:
        alternatives = [path]
        if path.suffix.lower() in {".mid", ".midi"}:
            alternatives.extend([path.with_suffix(".mid"), path.with_suffix(".midi")])
        elif path.suffix:
            alternatives.extend([path.with_suffix(".mid"), path.with_suffix(".midi")])
        else:
            alternatives.extend([path.with_suffix(".mid"), path.with_suffix(".midi")])
        for candidate in alternatives:
            key = str(candidate)
            if key not in seen:
                expanded.append(candidate)
                seen.add(key)
    return expanded


def _available_environment_hint(error_text: str) -> str | None:
    marker = "Available environments:"
    if marker not in error_text:
        return None
    return error_text[error_text.index(marker) :].strip()


def _canonicalize_environment_name(song_id: str) -> str:
    return re.sub(r"(-v\d+)_\d+$", r"\1", str(song_id))


def _collect_musical_metrics(env: Any, *, episode_finished: bool) -> tuple[dict[str, float], str | None]:
    if episode_finished:
        try:
            return dict(env.get_musical_metrics()), None
        except Exception as exc:  # pragma: no cover
            return {}, f"Failed to collect final musical metrics: {exc}"
    if hasattr(env, "_compute_key_press_metrics") and hasattr(env, "_compute_sustain_metrics"):
        try:
            key_press_metrics = env._compute_key_press_metrics()
            sustain_metrics = env._compute_sustain_metrics()
            return (
                {
                    "precision": float(key_press_metrics.precision),
                    "recall": float(key_press_metrics.recall),
                    "f1": float(key_press_metrics.f1),
                    "sustain_precision": float(sustain_metrics.precision),
                    "sustain_recall": float(sustain_metrics.recall),
                    "sustain_f1": float(sustain_metrics.f1),
                },
                "Computed musical metrics over the executed action prefix because the environment did not terminate.",
            )
        except Exception as exc:  # pragma: no cover
            return {}, f"Failed to compute partial musical metrics: {exc}"
    return {}, "Musical metrics unavailable because the environment did not terminate."


def _iter_wrapped_envs(env: Any) -> list[Any]:
    current = env
    wrapped: list[Any] = []
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        wrapped.append(current)
        seen.add(id(current))
        current = getattr(current, "_environment", None)
    return wrapped


def _render_frame(env: Any, *, height: int, width: int) -> np.ndarray:
    errors: list[str] = []
    for current in _iter_wrapped_envs(env):
        physics = getattr(current, "physics", None)
        if physics is not None and hasattr(physics, "render"):
            for kwargs in ({"height": height, "width": width}, {"height": height, "width": width, "camera_id": 0}):
                try:
                    frame = physics.render(**kwargs)
                    return _normalize_frame(frame)
                except Exception as exc:
                    errors.append(f"physics.render({kwargs}): {exc}")
        render = getattr(current, "render", None)
        if callable(render):
            for kwargs in ({"height": height, "width": width}, {"mode": "rgb_array", "height": height, "width": width}):
                try:
                    frame = render(**kwargs)
                    if frame is not None:
                        return _normalize_frame(frame)
                except Exception as exc:
                    errors.append(f"render({kwargs}): {exc}")
    raise RuntimeError(f"Unable to render DM Control frame. {_RENDER_BACKEND_HINT} Errors: {' | '.join(errors[:4])}")


def _normalize_frame(frame: Any) -> np.ndarray:
    array = np.asarray(frame, dtype=np.uint8)
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise ValueError(f"Expected an RGB/RGBA frame, got shape {array.shape}.")
    if array.shape[-1] == 4:
        array = array[..., :3]
    return array


def _write_rollout_video(
    *,
    frames: list[np.ndarray],
    output_path: Path,
    fps: int,
    temp_root: Path,
    audio_source: str = "none",
    reference_midi_path: Path | None = None,
    robot_midi_events: list[Any] | None = None,
    ffmpeg_exe: str | None = None,
    logger: logging.Logger | None = None,
) -> tuple[Path, str, str | None]:
    log = logger or LOGGER
    if not frames:
        raise ValueError("No frames were captured for this rollout.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ff = ffmpeg_exe or resolve_ffmpeg_executable()
    writer_errors: list[str] = []
    warning_parts: list[str] = []

    def mux_audio_phase(video_file: Path) -> None:
        nonlocal warning_parts
        src = str(audio_source or "none").strip().lower()
        if src == "none":
            return
        if ff is None:
            warning_parts.append("Audio skipped: ffmpeg not found (set SONATA_FFMPEG or install ffmpeg).")
            return
        if src == "reference_midi":
            if reference_midi_path is None or not reference_midi_path.is_file():
                warning_parts.append("reference_midi: missing MIDI path on disk.")
                return
            wav_path = temp_root / f"{output_path.stem}_reference.wav"
            if not synthesize_reference_wav(reference_midi_path, wav_path, logger=log):
                warning_parts.append("reference_midi: FluidSynth synthesize failed.")
                return
            warn = mux_video_with_wav(
                video_path=video_file, wav_path=wav_path, temp_root=temp_root, ffmpeg_exe=ff, logger=log
            )
            wav_path.unlink(missing_ok=True)
            if warn:
                warning_parts.append(warn)
            return
        if src == "robot_midi":
            events = robot_midi_events or []
            if not events:
                warning_parts.append("robot_midi: no MidiModule messages from simulated piano.")
                return
            w = _maybe_attach_rollout_audio(
                video_path=video_file,
                audio_events=events,
                temp_root=temp_root,
                ffmpeg_path=ff,
            )
            if w:
                warning_parts.append(w)
            return
        warning_parts.append(f"Unknown audio_source={audio_source!r}; leaving silent.")

    try:
        import imageio.v2 as imageio

        imageio.mimwrite(
            output_path,
            frames,
            fps=fps,
            codec="libx264",
            quality=7,
            macro_block_size=None,
        )
        mux_audio_phase(output_path)
        return output_path, "mp4", " ".join(warning_parts) if warning_parts else None
    except Exception as exc:
        if output_path.exists():
            output_path.unlink(missing_ok=True)
        writer_errors.append(f"imageio MP4 export failed: {exc}")
    try:
        if ff is None:
            raise RuntimeError("`ffmpeg` was not found on PATH.")
        video_path, video_format, warning = _write_video_with_ffmpeg(
            frames=frames,
            output_path=output_path,
            fps=fps,
            temp_root=temp_root,
            ffmpeg_path=ff,
        )
        if warning:
            warning_parts.append(warning)
        mux_audio_phase(video_path)
        return video_path, video_format, " ".join(warning_parts) if warning_parts else None
    except Exception as exc:
        writer_errors.append(f"ffmpeg MP4 export failed: {exc}")
    try:
        gif_path = _write_gif_fallback(frames=frames, output_path=output_path.with_suffix(".gif"), fps=fps)
        return (
            gif_path,
            "gif",
            "Fell back to GIF because MP4 encoding was unavailable. " + " ".join(writer_errors),
        )
    except Exception as exc:
        writer_errors.append(f"GIF fallback failed: {exc}")
    raise RuntimeError("Failed to export rollout video. " + " ".join(writer_errors))


def _write_video_with_ffmpeg(
    *,
    frames: list[np.ndarray],
    output_path: Path,
    fps: int,
    temp_root: Path,
    ffmpeg_path: str,
) -> tuple[Path, str, None]:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required for ffmpeg frame export fallback.") from exc
    temp_root.mkdir(parents=True, exist_ok=True)
    frame_dir = Path(tempfile.mkdtemp(prefix=f"{output_path.stem}_", dir=str(temp_root)))
    try:
        for index, frame in enumerate(frames):
            Image.fromarray(frame).save(frame_dir / f"{index:06d}.png")
        command = [
            ffmpeg_path,
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(max(int(fps), 1)),
            "-i",
            str(frame_dir / "%06d.png"),
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        if completed.stderr:
            LOGGER.debug("ffmpeg wrote rollout video with stderr output: %s", completed.stderr.strip())
        shutil.rmtree(frame_dir, ignore_errors=True)
        return output_path, "mp4", None
    except Exception:
        if output_path.exists():
            output_path.unlink(missing_ok=True)
        raise


def _find_piano_midi_events(env: Any) -> list[Any]:
    for current in _iter_wrapped_envs(env):
        task = getattr(current, "task", None)
        piano = getattr(task, "piano", None)
        midi_module = getattr(piano, "midi_module", None)
        get_all = getattr(midi_module, "get_all_midi_messages", None)
        if callable(get_all):
            events = list(get_all())
            if events:
                return events
    return []


def _maybe_attach_rollout_audio(
    *,
    video_path: Path,
    audio_events: list[Any],
    temp_root: Path,
    ffmpeg_path: str,
) -> str | None:
    if not audio_events:
        return None
    ffmpeg_exe = ffmpeg_path
    try:
        from robopianist.music import synthesizer
    except Exception as exc:
        return f"Audio mux skipped because RoboPianist synthesizer import failed: {exc}"

    soundfont_path = _default_soundfont_path()
    if soundfont_path is None:
        return "Audio mux skipped because no soundfont file was available."

    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f"{video_path.stem}_audio_", dir=str(temp_root)))
    wav_path = temp_dir / f"{video_path.stem}.wav"
    temp_video_path = temp_dir / video_path.name
    try:
        synth = synthesizer.Synthesizer(soundfont_path=soundfont_path)
        try:
            waveform = synth.get_samples(_clone_midi_events(audio_events))
        finally:
            synth.stop()
        _write_waveform(wav_path=wav_path, waveform=waveform)
        shutil.copyfile(video_path, temp_video_path)
        command = [
            ffmpeg_exe,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(temp_video_path),
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
        ]
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        if completed.stderr:
            LOGGER.debug("ffmpeg wrote rollout audio with stderr output: %s", completed.stderr.strip())
        return None
    except Exception as exc:
        return f"Audio mux failed: {exc}"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _default_soundfont_path() -> Path | None:
    repo_root = Path(__file__).resolve().parents[4]
    candidates = [
        repo_root / "robopianist" / "soundfonts" / "SalamanderGrandPiano.sf2",
        repo_root / "third_party" / "soundfonts" / "TimGM6mb.sf2",
    ]
    for candidate in candidates:
        if _is_valid_soundfont(candidate):
            return candidate
    return None


def _is_valid_soundfont(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    try:
        with path.open("rb") as handle:
            return handle.read(4) == b"RIFF"
    except OSError:
        return False


def _clone_midi_events(events: list[Any]) -> list[Any]:
    cloned: list[Any] = []
    for event in events:
        note = getattr(event, "note", None)
        velocity = getattr(event, "velocity", None)
        value = getattr(event, "value", None)
        control = getattr(event, "control", None)
        time_value = float(getattr(event, "time"))
        event_type = type(event).__name__
        if note is not None and velocity is not None:
            cloned.append(type(event)(note=int(note), velocity=int(velocity), time=time_value))
        elif note is not None:
            cloned.append(type(event)(note=int(note), time=time_value))
        elif control is not None and value is not None:
            try:
                cloned.append(type(event)(time=time_value))
            except TypeError:
                cloned.append(type(event)(control=int(control), value=int(value), time=time_value))
        else:
            raise ValueError(f"Unsupported MIDI event type for audio synthesis: {event_type}")
    return cloned


def _write_waveform(*, wav_path: Path, waveform: np.ndarray, sample_rate: int = 44100) -> None:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(wav_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(np.asarray(waveform, dtype=np.int16).tobytes())


def _write_gif_fallback(*, frames: list[np.ndarray], output_path: Path, fps: int) -> Path:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required for GIF fallback export.") from exc
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(frame) for frame in frames]
    if not images:
        raise ValueError("No frames available for GIF fallback.")
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=max(int(1000 / max(int(fps), 1)), 1),
        loop=0,
    )
    return output_path


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "episode"


def _relative_output_path(path: Path | None, output_root: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(output_root.resolve()))
    except Exception:
        return str(path.resolve())


def _maybe_promote_action_diagnostics_flat(
    *, output_root: Path, results: list[dict[str, Any]], logger: logging.Logger
) -> None:
    successes = [item for item in results if not item.get("error") and item.get("action_diagnostics_dir")]
    if len(successes) != 1:
        return
    rel = successes[0].get("action_diagnostics_dir")
    if not rel:
        return
    diag_dir = (output_root / str(rel)).resolve()
    if not diag_dir.is_dir():
        return
    for name in ("action_magnitude_diagnostics.csv", "action_magnitude_summary.json", "action_samples.npz"):
        src = diag_dir / name
        dest = output_root / name
        if src.exists():
            try:
                shutil.copy2(src, dest)
            except OSError as exc:
                logger.warning("Could not copy %s to rollout root: %s", name, exc)


def _build_video_caption(result: dict[str, Any]) -> str:
    parts = [
        f"song_id={result.get('song_id', 'unknown')}",
        f"episode_id={result.get('episode_id', 'unknown')}",
        f"reward={float(result.get('reward', 0.0)):.3f}",
    ]
    for key in ("precision", "recall", "f1", "sustain_f1"):
        if isinstance(result.get(key), (int, float)):
            parts.append(f"{key}={float(result[key]):.3f}")
    if result.get("metrics_note"):
        parts.append(str(result["metrics_note"]))
    return " | ".join(parts)


def _summarize_rollout_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "episode_count": len(results),
        "successful_episodes": sum(1 for item in results if not item.get("error")),
        "error_count": sum(1 for item in results if item.get("error")),
        "rendered_video_count": sum(1 for item in results if item.get("video_path")),
        "render_error_count": sum(1 for item in results if item.get("render_error")),
    }
    if not results:
        return summary
    numeric_df = pd.DataFrame(results).apply(pd.to_numeric, errors="coerce")
    for column in ("reward", "precision", "recall", "f1", "sustain_precision", "sustain_recall", "sustain_f1"):
        series = numeric_df[column].dropna() if column in numeric_df else pd.Series(dtype=float)
        if not series.empty:
            summary[f"mean_{column}"] = float(series.mean())
    return summary


def _safe_close(env: Any) -> None:
    if env is None:
        return
    close = getattr(env, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # pragma: no cover
            LOGGER.debug("Ignoring rollout environment close failure.", exc_info=True)
