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
from sonata.evaluation.offline import stitch_segment_predictions
from sonata.evaluation.causal_rollout_contract import CausalRolloutConfig, reset_or_validate_neutral_piano
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
    causal_eval: dict[str, Any] | None = None,
    wandb_run: Any | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    logger = logger or LOGGER
    causal_config = CausalRolloutConfig.from_mapping(causal_eval)
    ensure_local_robopianist_on_path()
    try:
        from robopianist import suite
        from robopianist.wrappers.evaluation import MidiEvaluationWrapper
    except Exception as exc:  # pragma: no cover
        result = {"available": False, "error": format_robopianist_import_error(exc)}
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
    expected_action_dim = int(action_series.dropna().iloc[0]) if not action_series.dropna().empty else int(stitch_segment_predictions(token_df, next(iter(predictions_by_episode.values()), []), 1).shape[-1])
    rollout_task_kwargs = build_rollout_task_kwargs(control_timestep=0.05, expected_action_dim=expected_action_dim)
    manifest_lookup = _load_rollout_manifest_lookup(primitive_root=primitive_root, logger=logger)
    output_root.mkdir(parents=True, exist_ok=True)
    videos_root = output_root / "videos"
    frames_tmp_root = output_root / "frames_tmp"
    results: list[dict[str, Any]] = []
    rendered_episodes = 0
    for episode_index, episode_id in enumerate(sorted(predictions_by_episode)[:limit_episodes]):
        episode_rows = token_df[token_df["episode_id"] == episode_id].sort_values("onset_step")
        if episode_rows.empty:
            results.append({"episode_id": episode_id, "error": f"No token rows found for episode `{episode_id}`."})
            continue
        episode_predictions = predictions_by_episode[episode_id]
        if not episode_predictions:
            results.append(
                {
                    "episode_id": episode_id,
                    "song_id": str(episode_rows.iloc[0]["song_id"]),
                    "error": "No diffusion predictions found for episode.",
                }
            )
            continue
        stitched = stitch_segment_predictions(
            token_df=episode_rows,
            episode_predictions=episode_predictions,
            action_horizon=int(episode_predictions[0]["predicted"].shape[0]),
        )
        raw_song_id = str(episode_rows.iloc[0]["song_id"])
        source = _resolve_rollout_source(
            song_id=raw_song_id,
            episode_id=str(episode_id),
            manifest_lookup=manifest_lookup,
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
            timestep = env.reset()
            neutral_check = reset_or_validate_neutral_piano(env, causal_config=causal_config)
            if causal_config.enabled and causal_config.require_neutral_piano_start and not neutral_check.passed:
                raise RuntimeError(
                    "Causal DM-Control rollout failed neutral piano start validation: "
                    f"{neutral_check.initial_active_key_indices}"
                )
            total_reward = 0.0
            action_dim = int(env.action_spec().shape[0])
            validate_rollout_action_dim(
                actual_action_dim=action_dim,
                expected_action_dim=expected_action_dim,
                environment_name=source["environment_name"],
            )
            frames: list[np.ndarray] = []
            render_error: str | None = None
            if should_render:
                try:
                    frames.append(_render_frame(env, height=video_height, width=video_width))
                except Exception as exc:
                    render_error = str(exc)
                    logger.warning("Initial render failed for DM Control episode `%s`: %s", episode_id, exc)
            actions_executed = 0
            for action in stitched:
                control = np.zeros((action_dim,), dtype=np.float32)
                control[: min(action_dim, action.shape[0])] = action[: min(action_dim, action.shape[0])]
                timestep = env.step(control)
                total_reward += float(timestep.reward or 0.0)
                actions_executed += 1
                if should_render and render_error is None:
                    try:
                        frames.append(_render_frame(env, height=video_height, width=video_width))
                    except Exception as exc:
                        render_error = str(exc)
                        logger.warning("Render failed for DM Control episode `%s`: %s", episode_id, exc)
                if timestep.last():
                    break
            metrics, metrics_note = _collect_musical_metrics(env, episode_finished=bool(timestep.last()))
            video_path: Path | None = None
            video_format: str | None = None
            video_warning: str | None = None
            if should_render:
                rendered_episodes += 1
                if render_error is None:
                    try:
                        audio_events = (
                            _find_piano_midi_events(env)
                            if str(causal_config.video_audio_source) == "robot_midi"
                            else []
                        )
                        video_path, video_format, video_warning = _write_rollout_video(
                            frames=frames,
                            output_path=videos_root
                            / f"{_safe_filename(source['environment_name'])}_{_safe_filename(str(episode_id))}.mp4",
                            fps=video_fps,
                            temp_root=frames_tmp_root,
                            audio_events=audio_events,
                        )
                    except Exception as exc:
                        render_error = str(exc)
                        logger.warning("Video export failed for DM Control episode `%s`: %s", episode_id, exc)
            episode_result = {
                "episode_index": episode_index,
                "episode_id": episode_id,
                "song_id": raw_song_id,
                "environment_name": source["environment_name"],
                "midi_file": str(source["midi_file"]) if source["midi_file"] is not None else None,
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
                "causal_eval_enabled": bool(causal_config.enabled),
                "causal_validated": bool(causal_config.enabled and neutral_check.passed),
                "causal_failure_reason": None if neutral_check.passed else neutral_check.failure_reason,
                "restore_mode": causal_config.restore_mode,
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
                    "environment_name": source["environment_name"],
                    "midi_file": str(source["midi_file"]) if source["midi_file"] is not None else None,
                    "song_id_normalized": bool(source["song_id_normalized"]),
                    "error": str(exc),
                }
            )
        finally:
            _safe_close(env)
    summary = _summarize_rollout_results(results)
    payload = {
        "available": True,
        "causal_eval": causal_config.to_dict(),
        "causal_note": "DM-Control rollout reports environment reward/MIDI wrapper metrics as legacy_state_metrics; causal success is not inferred from reference MIDI.",
        "videos_dir": _relative_output_path(videos_root if videos_root.exists() else None, output_root),
        "summary": summary,
        "episodes": results,
    }
    write_json(payload, output_root / "dm_control_rollout.json")
    results_df = pd.DataFrame(results)
    write_table(results_df, output_root / "dm_control_rollout")
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
    logger: logging.Logger,
) -> dict[str, Any]:
    manifest_row = manifest_lookup.get((song_id, episode_id), {})
    midi_file = _manifest_note_path(manifest_row)
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
        "song_id_normalized": song_id_normalized,
    }


def _manifest_note_path(manifest_row: dict[str, Any]) -> Path | None:
    note_path = manifest_row.get("note_path") if manifest_row else None
    if note_path is None:
        return None
    text = str(note_path).strip()
    if not text or text.lower() == "nan":
        return None
    path = Path(text).resolve()
    return path if path.exists() else None


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
    audio_events: list[Any] | None = None,
) -> tuple[Path, str, str | None]:
    if not frames:
        raise ValueError("No frames were captured for this rollout.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer_errors: list[str] = []
    warning_parts: list[str] = []
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
        if audio_events:
            warning = _maybe_attach_rollout_audio(
                video_path=output_path,
                audio_events=audio_events,
                temp_root=temp_root,
            )
            if warning:
                warning_parts.append(warning)
        return output_path, "mp4", " ".join(warning_parts) if warning_parts else None
    except Exception as exc:
        if output_path.exists():
            output_path.unlink(missing_ok=True)
        writer_errors.append(f"imageio MP4 export failed: {exc}")
    try:
        video_path, video_format, warning = _write_video_with_ffmpeg(
            frames=frames,
            output_path=output_path,
            fps=fps,
            temp_root=temp_root,
        )
        if warning:
            warning_parts.append(warning)
        if audio_events:
            warning = _maybe_attach_rollout_audio(
                video_path=video_path,
                audio_events=audio_events,
                temp_root=temp_root,
            )
            if warning:
                warning_parts.append(warning)
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
) -> tuple[Path, str, None]:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError("`ffmpeg` was not found on PATH.")
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
) -> str | None:
    if not audio_events:
        return None
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        return "Audio mux skipped because `ffmpeg` was not found on PATH."
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
            ffmpeg_path,
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
