from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from sonata.diffusion.dataset import build_prior_lookup, load_diffusion_inputs
from sonata.diffusion.trainer import maybe_load_planner
from sonata.evaluation.rollout import evaluate_dm_control_rollout
from sonata.transformer.dataset import PrimitiveSequenceDataset, family_mask_tensor, planner_collate_fn
from sonata.transformer.decode import decode_factored_outputs
from sonata.transformer.primitive_remap import build_remap_tensor
from sonata.utils.io import write_json, write_table
from sonata.utils.torch_utils import move_to_device

LOGGER = logging.getLogger(__name__)
DEFAULT_ACTION_HORIZON = 16
DEFAULT_STATE_CONTEXT_STEPS = 1


def evaluate_stage2_rollout(
    *,
    primitive_root: Path,
    output_root: Path,
    mode: str,
    backend: str,
    split: str = "val",
    planner_checkpoint: Path | None = None,
    device: str = "cpu",
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
    video_audio_source: str = "none",
    debug_overlay: bool = False,
    wandb_run: Any | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    logger = logger or LOGGER
    primitive_root = primitive_root.resolve()
    output_root = output_root.resolve()
    _validate_inputs(primitive_root=primitive_root, planner_checkpoint=planner_checkpoint, mode=mode)
    output_root.mkdir(parents=True, exist_ok=True)

    planner_payload = _load_checkpoint_metadata(planner_checkpoint, device=device) if mode == "stage2_gmr" else {}
    planner_config = _planner_config(planner_payload)
    action_horizon = int(planner_config.get("action_horizon", DEFAULT_ACTION_HORIZON))
    state_context_steps = int(planner_config.get("state_context_steps", DEFAULT_STATE_CONTEXT_STEPS))
    family_mapping_mode = str(planner_config.get("family_mapping_mode", "heuristic_stats"))
    continuous_param_names = planner_config.get("continuous_param_names")

    token_df, metadata, prior_lookup = load_diffusion_inputs(
        primitive_root,
        action_horizon=action_horizon,
        state_context_steps=state_context_steps,
        family_mapping_mode=family_mapping_mode,
        continuous_param_names=continuous_param_names,
    )
    token_df = _filter_token_df(token_df, split=split, limit_episodes=limit_episodes, episode_id=episode_id)
    if token_df.empty:
        raise ValueError(f"No primitive token rows found for split={split!r} under {primitive_root}.")
    selected_episode_id, selected_song_id = _selection_ids(token_df)

    if mode == "oracle_gmr":
        predictions_by_episode, prediction_records = _build_oracle_predictions(
            token_df=token_df,
            metadata=metadata,
            prior_lookup=prior_lookup,
            split=split,
            mode=mode,
        )
        metrics = _oracle_metrics(predictions_by_episode=predictions_by_episode, prediction_records=prediction_records)
    elif mode == "stage2_gmr":
        predictions_by_episode, prediction_records, metrics = _build_stage2_predictions(
            token_df=token_df,
            metadata=metadata,
            prior_lookup=prior_lookup,
            planner_checkpoint=planner_checkpoint,
            planner_config=planner_config,
            split=split,
            mode=mode,
            device=device,
            logger=logger,
        )
    else:
        raise ValueError(f"Unsupported mode={mode!r}.")

    prediction_df = pd.DataFrame(prediction_records)
    write_table(prediction_df, output_root / "stage2_rollout_predictions")
    write_table(pd.DataFrame([metrics]), output_root / "stage2_rollout_metrics")

    rollout_payload = None
    rollout_action_source = "oracle_gmr_primitives" if mode == "oracle_gmr" else "stage2_gmr"
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
            action_source=rollout_action_source,
            control_timestep=float(control_timestep),
            rollout_mode="policy",
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
        "primitive_root": str(primitive_root),
        "planner_checkpoint": str(planner_checkpoint.resolve()) if planner_checkpoint is not None else None,
        "output_root": str(output_root),
        "predictions_csv": str((output_root / "stage2_rollout_predictions.csv").resolve()),
        "metrics_csv": str((output_root / "stage2_rollout_metrics.csv").resolve()),
        "duration_timeline_policy": "Decoded predicted_duration is logged for diagnostics; rollout onset_step/end_step remain from the original token row.",
        "control_timestep": float(control_timestep),
        "video_audio_source": str(video_audio_source),
        "debug_overlay": bool(debug_overlay),
    }
    if rollout_payload is not None:
        summary["rollout"] = rollout_payload
    write_json(summary, output_root / "stage2_rollout_summary.json")
    return {
        "summary": summary,
        "metrics": metrics,
        "predictions_by_episode": predictions_by_episode,
        "predictions": prediction_df,
        "rollout": rollout_payload,
    }


def _validate_inputs(*, primitive_root: Path, planner_checkpoint: Path | None, mode: str) -> None:
    if not primitive_root.exists():
        raise FileNotFoundError(f"primitive_root does not exist: {primitive_root}")
    token_base = primitive_root / "tokens" / "primitive_tokens"
    if not token_base.with_suffix(".parquet").exists() and not token_base.with_suffix(".csv").exists():
        raise FileNotFoundError(f"Missing Stage 1 primitive tokens at {token_base}.parquet or {token_base}.csv")
    if mode == "stage2_gmr":
        if planner_checkpoint is None:
            raise ValueError("--planner-checkpoint is required for --mode stage2_gmr.")
        if not planner_checkpoint.exists():
            raise FileNotFoundError(f"planner checkpoint does not exist: {planner_checkpoint}")


def _load_checkpoint_metadata(path: Path | None, *, device: str) -> dict[str, Any]:
    if path is None:
        return {}
    from sonata.utils.checkpointing import load_checkpoint

    return load_checkpoint(path, map_location=torch.device(device))


def _planner_config(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("planner_config"):
        return dict(payload["planner_config"])
    return dict(payload.get("config", {}))


def _filter_token_df(
    token_df: pd.DataFrame, *, split: str, limit_episodes: int | None, episode_id: str | None
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


def _build_oracle_predictions(
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
        records.append(
            _prediction_record(
                row=row,
                metadata=metadata,
                split=split,
                mode=mode,
                target_primitive=int(row.primitive_index),
                predicted_primitive=None,
                target_family=int(row.primitive_family_index) if hasattr(row, "primitive_family_index") else None,
                predicted_family=None,
                target_duration=int(row.duration_bucket) if hasattr(row, "duration_bucket") else None,
                predicted_duration=None,
                target_dynamics=int(row.dynamics_bucket) if hasattr(row, "dynamics_bucket") else None,
                predicted_dynamics=None,
            )
        )
    return predictions_by_episode, records


def _build_stage2_predictions(
    *,
    token_df: pd.DataFrame,
    metadata: Any,
    prior_lookup: dict[str, np.ndarray],
    planner_checkpoint: Path | None,
    planner_config: dict[str, Any],
    split: str,
    mode: str,
    device: str,
    logger: logging.Logger,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], dict[str, Any]]:
    torch_device = torch.device(device)
    load_config = dict(planner_config)
    load_config["variant"] = "stage2_gmr"
    load_config["planner_checkpoint"] = str(planner_checkpoint)
    planner, _, loaded_planner_config, _ = maybe_load_planner(load_config, metadata, torch_device)
    if planner is None:
        raise ValueError(f"Unable to load Stage 2 planner checkpoint: {planner_checkpoint}")
    planner.eval()

    context_length = int(loaded_planner_config.get("context_length", planner_config.get("context_length", 0)))
    if context_length <= 0:
        raise ValueError("Planner checkpoint config is missing a positive context_length.")
    batch_size = int(loaded_planner_config.get("batch_size", planner_config.get("batch_size", 64)))
    eval_temperature = float(loaded_planner_config.get("eval_temperature", planner_config.get("eval_temperature", 1.0)))
    topk = int(loaded_planner_config.get("topk", planner_config.get("topk", 5)))

    dataset = PrimitiveSequenceDataset(token_df, metadata, context_length=context_length, split=split)
    row_records = _stage2_target_rows(token_df)
    if len(dataset) != len(row_records):
        raise RuntimeError(
            f"Planner dataset/sample metadata mismatch: dataset has {len(dataset)} samples but row metadata has {len(row_records)} rows."
        )
    logger.info("Evaluating Stage 2 planner over %d %s chunks from %d episode(s).", len(dataset), split, token_df["episode_id"].nunique())
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: planner_collate_fn(batch, metadata=metadata),
    )
    family_mask = family_mask_tensor(metadata, device=torch_device)
    remap_tensor = build_remap_tensor(
        metadata.num_primitives,
        metadata.primitive_remap_summary,
        metadata.primitive_ids,
        device=torch_device,
    )

    predictions_by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    records: list[dict[str, Any]] = []
    target_family: list[int] = []
    predicted_family: list[int] = []
    target_primitive: list[int] = []
    predicted_primitive: list[int] = []
    target_duration: list[int] = []
    predicted_duration: list[int] = []
    primitive_topk_hits: list[float] = []
    cursor = 0

    with torch.no_grad():
        for batch in loader:
            batch_size_actual = int(batch["target_primitive"].shape[0])
            rows = row_records[cursor : cursor + batch_size_actual]
            cursor += batch_size_actual
            batch = move_to_device(batch, str(torch_device))
            outputs = planner(batch)
            decoded = decode_factored_outputs(
                outputs,
                family_mask=family_mask,
                temperature=eval_temperature,
                remap_tensor=remap_tensor,
            )
            primitive_topk_hits.append(_topk_accuracy(decoded["masked_primitive_logits"], batch["target_primitive"], topk=topk))

            pred_family_values = decoded["predicted_family"].detach().cpu().numpy().astype(np.int64)
            pred_primitive_values = decoded["predicted_primitive"].detach().cpu().numpy().astype(np.int64)
            pred_duration_values = decoded["predicted_duration"].detach().cpu().numpy().astype(np.int64)
            pred_dynamics_values = decoded["predicted_dynamics"].detach().cpu().numpy().astype(np.int64)
            target_family_values = batch["target_family"].detach().cpu().numpy().astype(np.int64)
            target_primitive_values = batch["target_primitive"].detach().cpu().numpy().astype(np.int64)
            target_duration_values = batch["target_duration"].detach().cpu().numpy().astype(np.int64)

            target_family.extend(target_family_values.tolist())
            predicted_family.extend(pred_family_values.tolist())
            target_primitive.extend(target_primitive_values.tolist())
            predicted_primitive.extend(pred_primitive_values.tolist())
            target_duration.extend(target_duration_values.tolist())
            predicted_duration.extend(pred_duration_values.tolist())

            target_dynamics_values = batch["target_dynamics"].detach().cpu().numpy().astype(np.int64)
            for index, row in enumerate(rows):
                primitive_index = int(pred_primitive_values[index])
                primitive_id = _primitive_id(metadata, primitive_index)
                action_chunk = _lookup_prior(prior_lookup, primitive_id)
                episode_id = str(row["episode_id"])
                predictions_by_episode[episode_id].append(
                    {
                        "predicted": action_chunk,
                        "prior": action_chunk,
                        "onset_step": int(row["onset_step"]),
                        "end_step": int(row["end_step"]),
                        "predicted_duration": int(pred_duration_values[index]),
                    }
                )
                records.append(
                    _prediction_record(
                        row=row,
                        metadata=metadata,
                        split=split,
                        mode=mode,
                        target_primitive=int(target_primitive_values[index]),
                        predicted_primitive=primitive_index,
                        target_family=int(target_family_values[index]),
                        predicted_family=int(pred_family_values[index]),
                        target_duration=int(target_duration_values[index]),
                        predicted_duration=int(pred_duration_values[index]),
                        target_dynamics=int(target_dynamics_values[index]),
                        predicted_dynamics=int(pred_dynamics_values[index]),
                    )
                )

    metrics = _stage2_metrics(
        predictions_by_episode=predictions_by_episode,
        target_family=target_family,
        predicted_family=predicted_family,
        target_primitive=target_primitive,
        predicted_primitive=predicted_primitive,
        target_duration=target_duration,
        predicted_duration=predicted_duration,
        primitive_topk_hits=primitive_topk_hits,
        topk=topk,
    )
    metrics["mode"] = mode
    return predictions_by_episode, records, metrics


def _stage2_target_rows(token_df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouped = token_df.sort_values(["episode_id", "onset_step"]).groupby("episode_id", sort=True)
    for _, group in grouped:
        for target_index in range(1, len(group)):
            rows.append(group.iloc[target_index].to_dict())
    return rows


def _prediction_record(
    *,
    row: Any,
    metadata: Any,
    split: str,
    mode: str,
    target_primitive: int | None,
    predicted_primitive: int | None,
    target_family: int | None,
    predicted_family: int | None,
    target_duration: int | None,
    predicted_duration: int | None,
    target_dynamics: int | None,
    predicted_dynamics: int | None,
) -> dict[str, Any]:
    get = row.get if isinstance(row, dict) else lambda key, default=None: getattr(row, key, default)
    return {
        "episode_id": str(get("episode_id")),
        "song_id": str(get("song_id")) if get("song_id") is not None else None,
        "onset_step": int(get("onset_step")),
        "end_step": int(get("end_step")),
        "target_family": _family_name(metadata, target_family),
        "predicted_family": _family_name(metadata, predicted_family),
        "target_primitive": _primitive_id(metadata, target_primitive) if target_primitive is not None else None,
        "predicted_primitive": _primitive_id(metadata, predicted_primitive) if predicted_primitive is not None else None,
        "target_duration": target_duration,
        "predicted_duration": predicted_duration,
        "target_dynamics": target_dynamics,
        "predicted_dynamics": predicted_dynamics,
        "mode": mode,
        "split": split,
    }


def _lookup_prior(prior_lookup: dict[str, np.ndarray], primitive_id: str) -> np.ndarray:
    if primitive_id not in prior_lookup:
        raise KeyError(f"No GMR prior found for primitive_id={primitive_id!r}.")
    return np.asarray(prior_lookup[primitive_id], dtype=np.float32)


def _primitive_id(metadata: Any, primitive_index: int | None) -> str | None:
    if primitive_index is None:
        return None
    index = int(primitive_index)
    if index < 0 or index >= len(metadata.primitive_ids):
        return str(index)
    return str(metadata.primitive_ids[index])


def _family_name(metadata: Any, family_index: int | None) -> str | None:
    if family_index is None:
        return None
    index = int(family_index)
    if index < 0 or index >= len(metadata.primitive_family_names):
        return str(index)
    return str(metadata.primitive_family_names[index])


def _oracle_metrics(
    *,
    predictions_by_episode: dict[str, list[dict[str, Any]]],
    prediction_records: list[dict[str, Any]],
) -> dict[str, Any]:
    durations = [int(record["end_step"]) - int(record["onset_step"]) for record in prediction_records]
    action_dim = _action_dim(predictions_by_episode)
    return {
        "mode": "oracle_gmr",
        "number_of_episodes": int(len(predictions_by_episode)),
        "number_of_chunks": int(len(prediction_records)),
        "action_dim": action_dim,
        "mean_chunk_duration": float(np.mean(durations)) if durations else 0.0,
    }


def _stage2_metrics(
    *,
    predictions_by_episode: dict[str, list[dict[str, Any]]],
    target_family: list[int],
    predicted_family: list[int],
    target_primitive: list[int],
    predicted_primitive: list[int],
    target_duration: list[int],
    predicted_duration: list[int],
    primitive_topk_hits: list[float],
    topk: int,
) -> dict[str, Any]:
    target_family_arr = np.asarray(target_family, dtype=np.int64)
    predicted_family_arr = np.asarray(predicted_family, dtype=np.int64)
    target_primitive_arr = np.asarray(target_primitive, dtype=np.int64)
    predicted_primitive_arr = np.asarray(predicted_primitive, dtype=np.int64)
    target_duration_arr = np.asarray(target_duration, dtype=np.int64)
    predicted_duration_arr = np.asarray(predicted_duration, dtype=np.int64)
    primitive_f1 = _f1_scores(target_primitive_arr, predicted_primitive_arr)
    return {
        "primitive_accuracy": _accuracy(target_primitive_arr, predicted_primitive_arr),
        "family_accuracy": _accuracy(target_family_arr, predicted_family_arr),
        "duration_accuracy": _accuracy(target_duration_arr, predicted_duration_arr),
        "mean_abs_duration_error": float(np.mean(np.abs(predicted_duration_arr - target_duration_arr)))
        if target_duration_arr.size
        else 0.0,
        "primitive_weighted_f1": primitive_f1["weighted"],
        "primitive_macro_f1": primitive_f1["macro"],
        "primitive_topk_accuracy": float(np.mean(primitive_topk_hits)) if primitive_topk_hits else 0.0,
        "primitive_topk": int(topk),
        "number_of_episodes": int(len(predictions_by_episode)),
        "number_of_chunks": int(target_primitive_arr.size),
        "action_dim": _action_dim(predictions_by_episode),
    }


def _accuracy(targets: np.ndarray, predictions: np.ndarray) -> float:
    if targets.size == 0:
        return 0.0
    return float(np.mean(targets == predictions))


def _f1_scores(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    if targets.size == 0:
        return {"macro": 0.0, "weighted": 0.0}
    classes = np.unique(targets)
    f1_values: list[float] = []
    weights: list[int] = []
    for cls in classes:
        truth = targets == cls
        pred = predictions == cls
        tp = int(np.logical_and(truth, pred).sum())
        fp = int(np.logical_and(~truth, pred).sum())
        fn = int(np.logical_and(truth, ~pred).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
        f1_values.append(float(f1))
        weights.append(int(truth.sum()))
    weight_arr = np.asarray(weights, dtype=np.float64)
    f1_arr = np.asarray(f1_values, dtype=np.float64)
    return {
        "macro": float(f1_arr.mean()) if f1_arr.size else 0.0,
        "weighted": float(np.average(f1_arr, weights=weight_arr)) if weight_arr.sum() > 0 else 0.0,
    }


def _topk_accuracy(logits: torch.Tensor, target: torch.Tensor, *, topk: int) -> float:
    _, indices = logits.topk(min(int(topk), logits.shape[-1]), dim=-1)
    return float((indices == target.unsqueeze(-1)).any(dim=-1).float().mean().item())


def _action_dim(predictions_by_episode: dict[str, list[dict[str, Any]]]) -> int:
    for predictions in predictions_by_episode.values():
        for item in predictions:
            predicted = item.get("predicted")
            if predicted is not None:
                return int(np.asarray(predicted).shape[-1])
    return 0
