from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sonata.data.loading import build_manifest_lookup, load_episode_record, load_stage1_source_manifest
from sonata.diffusion.dataset import load_diffusion_inputs, resample_sequence, slice_episode_array
from sonata.evaluation.rollout import evaluate_dm_control_rollout
from sonata.utils.io import write_json, write_table

LOGGER = logging.getLogger(__name__)


def evaluate_stage1_rollout(
    *,
    primitive_root: Path,
    output_root: Path,
    mode: str,
    backend: str,
    split: str = "val",
    episode_id: str | None = None,
    limit_episodes: int | None = None,
    render_video: bool = False,
    video_fps: int = 20,
    video_height: int = 480,
    video_width: int = 640,
    max_render_episodes: int | None = None,
    environment_name: str | None = None,
    midi_root: Path | None = None,
    prefer_midi_manifest: bool = True,
    control_timestep: float = 0.05,
    rollout_mode: str = "policy",
    video_audio_source: str = "none",
    debug_overlay: bool = False,
    wandb_run: Any | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    logger = logger or LOGGER
    primitive_root = primitive_root.resolve()
    output_root = output_root.resolve()
    _validate_inputs(primitive_root=primitive_root, mode=mode)
    output_root.mkdir(parents=True, exist_ok=True)

    token_df, metadata, prior_lookup = load_diffusion_inputs(
        primitive_root,
        action_horizon=16,
        state_context_steps=1,
        family_mapping_mode="heuristic_stats",
        continuous_param_names=None,
    )
    token_df = _filter_token_df(token_df, split=split, limit_episodes=limit_episodes, episode_id=episode_id)
    if token_df.empty:
        raise ValueError(f"No primitive token rows found for split={split!r} under {primitive_root}.")

    selected_episode_id, selected_song_id = _selection_ids(token_df)

    if mode == "oracle_dataset_actions":
        predictions_by_episode, prediction_records, metrics = _build_dataset_action_predictions(
            token_df=token_df,
            primitive_root=primitive_root,
            split=split,
            mode=mode,
            action_horizon=16,
        )
        action_source = "dataset_actions"
    elif mode == "oracle_gmr_primitives":
        predictions_by_episode, prediction_records = _build_oracle_gmr_predictions(
            token_df=token_df,
            metadata=metadata,
            prior_lookup=prior_lookup,
            split=split,
            mode=mode,
        )
        metrics = _oracle_gmr_metrics(predictions_by_episode=predictions_by_episode, prediction_records=prediction_records)
        action_source = "oracle_gmr_primitives"
    elif mode == "zero_actions":
        predictions_by_episode = {str(eid): [] for eid in sorted(token_df["episode_id"].astype(str).unique())}
        prediction_records = []
        metrics = {
            "mode": "zero_actions",
            "number_of_episodes": int(token_df["episode_id"].nunique()),
            "number_of_chunks": 0,
        }
        action_source = "zero_actions"
    else:
        raise ValueError(f"Unsupported mode={mode!r}.")

    prediction_df = pd.DataFrame(prediction_records)
    write_table(prediction_df, output_root / "stage1_rollout_predictions")
    write_table(pd.DataFrame([metrics]), output_root / "stage1_rollout_metrics")

    rollout_payload = None
    if backend == "dm_control":
        rollout_payload = evaluate_dm_control_rollout(
            primitive_root=primitive_root,
            predictions_by_episode=predictions_by_episode,
            output_root=output_root / "rollout",
            limit_episodes=limit_episodes if limit_episodes is not None else 2,
            render_video=render_video,
            video_fps=video_fps,
            video_height=video_height,
            video_width=video_width,
            max_render_episodes=max_render_episodes,
            environment_name_override=environment_name,
            prefer_manifest_midi=prefer_midi_manifest,
            midi_root=midi_root,
            logger=logger,
            wandb_run=wandb_run,
            action_source=action_source,
            control_timestep=float(control_timestep),
            rollout_mode="zero_actions" if mode == "zero_actions" else "policy",
            audio_source=str(video_audio_source),
            debug_overlay=bool(debug_overlay),
        )
    elif backend != "offline":
        raise ValueError(f"Unsupported backend={backend!r}.")

    summary = {
        **metrics,
        "selected_episode_id": selected_episode_id,
        "selected_song_id": selected_song_id,
        "split": split,
        "backend": backend,
        "mode": mode,
        "action_source": action_source,
        "primitive_root": str(primitive_root),
        "output_root": str(output_root),
        "predictions_csv": str((output_root / "stage1_rollout_predictions.csv").resolve()),
        "metrics_csv": str((output_root / "stage1_rollout_metrics.csv").resolve()),
        "control_timestep": float(control_timestep),
        "rollout_mode": "zero_actions" if mode == "zero_actions" else "policy",
        "audio_source_cli": str(video_audio_source),
        "debug_overlay": bool(debug_overlay),
    }
    if rollout_payload is not None:
        summary["rollout"] = rollout_payload
    write_json(summary, output_root / "stage1_rollout_summary.json")
    return {
        "summary": summary,
        "metrics": metrics,
        "predictions_by_episode": predictions_by_episode,
        "predictions": prediction_df,
        "rollout": rollout_payload,
    }


def _validate_inputs(*, primitive_root: Path, mode: str) -> None:
    if not primitive_root.exists():
        raise FileNotFoundError(f"primitive_root does not exist: {primitive_root}")
    token_base = primitive_root / "tokens" / "primitive_tokens"
    if not token_base.with_suffix(".parquet").exists() and not token_base.with_suffix(".csv").exists():
        raise FileNotFoundError(f"Missing Stage 1 primitive tokens at {token_base}.parquet or {token_base}.csv")
    if mode not in {"oracle_dataset_actions", "oracle_gmr_primitives", "zero_actions"}:
        raise ValueError(mode)


def _filter_token_df(
    token_df: pd.DataFrame,
    *,
    split: str,
    limit_episodes: int | None,
    episode_id: str | None,
) -> pd.DataFrame:
    if "split" not in token_df.columns:
        raise ValueError("Primitive token table is missing required column `split`.")
    selected = token_df[token_df["split"] == split].copy()
    if episode_id is not None:
        selected = selected[selected["episode_id"].astype(str) == str(episode_id)].copy()
        if selected.empty:
            raise ValueError(f"No rows for episode_id={episode_id!r} in split={split!r}.")
        return selected.sort_values(["episode_id", "onset_step", "end_step"]).reset_index(drop=True)
    if limit_episodes is not None:
        episode_ids = sorted(selected["episode_id"].astype(str).unique().tolist())[: int(limit_episodes)]
        selected = selected[selected["episode_id"].astype(str).isin(episode_ids)].copy()
    return selected.sort_values(["episode_id", "onset_step", "end_step"]).reset_index(drop=True)


def _selection_ids(token_df: pd.DataFrame) -> tuple[str, str]:
    eids = sorted(token_df["episode_id"].astype(str).unique().tolist())
    if not eids:
        raise ValueError("Empty token_df after filtering.")
    first = eids[0]
    song = str(token_df[token_df["episode_id"].astype(str) == first].iloc[0]["song_id"])
    return first, song


def _build_dataset_action_predictions(
    *,
    token_df: pd.DataFrame,
    primitive_root: Path,
    split: str,
    mode: str,
    action_horizon: int,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], dict[str, Any]]:
    manifest_lookup = build_manifest_lookup(load_stage1_source_manifest(primitive_root))
    predictions_by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    records: list[dict[str, Any]] = []

    for row in token_df.itertuples(index=False):
        episode_id = str(row.episode_id)
        song_id = str(row.song_id)
        key = (song_id, episode_id)
        if key not in manifest_lookup:
            raise KeyError(f"No manifest row for song_id={song_id!r} episode_id={episode_id!r}.")
        episode = load_episode_record(manifest_lookup[key])
        actions = episode.actions
        if actions is None:
            raise ValueError(f"Episode {episode_id} has no actions in dataset (cannot run oracle_dataset_actions).")
        onset = int(row.onset_step)
        end = int(row.end_step)
        segment = slice_episode_array(actions, onset, end)
        if segment is None or segment.shape[0] == 0:
            chunk = np.zeros((action_horizon, actions.shape[-1]), dtype=np.float32)
        else:
            chunk = resample_sequence(segment, action_horizon)
        chunk = chunk.astype(np.float32)
        predictions_by_episode[episode_id].append(
            {
                "predicted": chunk,
                "prior": chunk,
                "onset_step": onset,
                "end_step": end,
            }
        )
        records.append(_prediction_record(row=row, split=split, mode=mode))

    first_chunk = None
    for preds in predictions_by_episode.values():
        if preds:
            first_chunk = preds[0]["predicted"]
            break
    adim = int(np.asarray(first_chunk).shape[-1]) if first_chunk is not None else 0
    metrics = {
        "mode": mode,
        "number_of_episodes": int(len(predictions_by_episode)),
        "number_of_chunks": int(len(records)),
        "action_dim": adim,
    }
    return predictions_by_episode, records, metrics


def _build_oracle_gmr_predictions(
    *,
    token_df: pd.DataFrame,
    metadata: Any,
    prior_lookup: dict[str, np.ndarray],
    split: str,
    mode: str,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    predictions_by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    records: list[dict[str, Any]] = []
    for row in token_df.itertuples(index=False):
        primitive_id = str(row.primitive_id)
        action_chunk = _lookup_prior(prior_lookup, primitive_id)
        episode_id = str(row.episode_id)
        predictions_by_episode[episode_id].append(
            {
                "predicted": action_chunk,
                "prior": action_chunk,
                "onset_step": int(row.onset_step),
                "end_step": int(row.end_step),
            }
        )
        records.append(_prediction_record(row=row, split=split, mode=mode))
    return predictions_by_episode, records


def _prediction_record(*, row: Any, split: str, mode: str) -> dict[str, Any]:
    return {
        "episode_id": str(row.episode_id),
        "song_id": str(row.song_id) if getattr(row, "song_id", None) is not None else None,
        "onset_step": int(row.onset_step),
        "end_step": int(row.end_step),
        "primitive_id": str(getattr(row, "primitive_id", "")),
        "mode": mode,
        "split": split,
    }


def _lookup_prior(prior_lookup: dict[str, np.ndarray], primitive_id: str) -> np.ndarray:
    if primitive_id not in prior_lookup:
        raise KeyError(f"No GMR prior found for primitive_id={primitive_id!r}.")
    return np.asarray(prior_lookup[primitive_id], dtype=np.float32)


def _oracle_gmr_metrics(
    *,
    predictions_by_episode: dict[str, list[dict[str, Any]]],
    prediction_records: list[dict[str, Any]],
) -> dict[str, Any]:
    durations = [int(record["end_step"]) - int(record["onset_step"]) for record in prediction_records]
    action_dim = 0
    for predictions in predictions_by_episode.values():
        for item in predictions:
            predicted = item.get("predicted")
            if predicted is not None:
                action_dim = int(np.asarray(predicted).shape[-1])
                break
        if action_dim:
            break
    return {
        "mode": "oracle_gmr_primitives",
        "number_of_episodes": int(len(predictions_by_episode)),
        "number_of_chunks": int(len(prediction_records)),
        "action_dim": action_dim,
        "mean_chunk_duration": float(np.mean(durations)) if durations else 0.0,
    }
