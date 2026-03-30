from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sonata.evaluation.offline import stitch_segment_predictions
from sonata.training.mjx_rollout import MJXRolloutBackend, mjx_availability
from sonata.utils.io import write_json, write_table
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
    wandb_run: Any | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    logger = logger or LOGGER
    try:
        from robopianist import suite
        from robopianist.wrappers.evaluation import MidiEvaluationWrapper
    except Exception as exc:  # pragma: no cover
        result = {"available": False, "error": str(exc)}
        write_json(result, output_root / "dm_control_rollout.json")
        log_prefixed_metrics(wandb_run, result, prefix="rollout/dm_control", summary=True)
        return result

    token_df = (
        pd.read_parquet(primitive_root / "tokens" / "primitive_tokens.parquet")
        if (primitive_root / "tokens" / "primitive_tokens.parquet").exists()
        else pd.read_csv(primitive_root / "tokens" / "primitive_tokens.csv")
    )
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
        env_name = str(episode_rows.iloc[0]["song_id"])
        should_render = render_video and (max_render_episodes is None or rendered_episodes < max_render_episodes)
        env = None
        try:
            env = MidiEvaluationWrapper(
                suite.load(environment_name=env_name, seed=0, task_kwargs={"control_timestep": 0.05, "n_steps_lookahead": 1}),
                deque_size=1,
            )
            timestep = env.reset()
            total_reward = 0.0
            action_dim = int(env.action_spec().shape[0])
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
                        video_path, video_format, video_warning = _write_rollout_video(
                            frames=frames,
                            output_path=videos_root / f"{_safe_filename(env_name)}_{_safe_filename(str(episode_id))}.mp4",
                            fps=video_fps,
                            temp_root=frames_tmp_root,
                        )
                    except Exception as exc:
                        render_error = str(exc)
                        logger.warning("Video export failed for DM Control episode `%s`: %s", episode_id, exc)
            episode_result = {
                "episode_index": episode_index,
                "episode_id": episode_id,
                "song_id": env_name,
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
                **metrics,
            }
            results.append(episode_result)
            if video_path is not None:
                log_rollout_video(
                    wandb_run,
                    key=f"rollout/videos/{_safe_filename(env_name)}_{_safe_filename(str(episode_id))}",
                    video_path=video_path,
                    caption=_build_video_caption(episode_result),
                    fps=video_fps,
                    logger=logger,
                )
        except Exception as exc:  # pragma: no cover
            results.append({"episode_id": episode_id, "song_id": env_name, "error": str(exc)})
        finally:
            _safe_close(env)
    summary = _summarize_rollout_results(results)
    payload = {
        "available": True,
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
) -> tuple[Path, str, str | None]:
    if not frames:
        raise ValueError("No frames were captured for this rollout.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer_errors: list[str] = []
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
        return output_path, "mp4", None
    except Exception as exc:
        if output_path.exists():
            output_path.unlink(missing_ok=True)
        writer_errors.append(f"imageio MP4 export failed: {exc}")
    try:
        return _write_video_with_ffmpeg(frames=frames, output_path=output_path, fps=fps, temp_root=temp_root)
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
