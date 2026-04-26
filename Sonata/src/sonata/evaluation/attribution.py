from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import logging
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from sonata.diffusion.dataset import DiffusionChunkDataset, diffusion_collate_fn, load_diffusion_inputs
from sonata.evaluation.rollout import evaluate_dm_control_rollout
from sonata.models.pipeline import Sonata3Pipeline
from sonata.utils.io import write_json, write_table
from sonata.utils.wandb_eval import log_prefixed_metrics, log_rollout_table

LOGGER = logging.getLogger(__name__)

UTILITY_METRICS = ("reward", "f1", "sustain_f1")
LOSS_METRICS = ("action_l1", "action_mse")
CONTROL_MODES = ("predicted_full", "oracle_full", "oracle_no_prior", "oracle_gmr_only")


@dataclass(frozen=True, slots=True)
class AttributionControl:
    name: str
    variant: str | None
    use_oracle_tokens: bool
    zero_prior: bool


CONTROL_CONFIGS = {
    "predicted_full": AttributionControl("predicted_full", None, use_oracle_tokens=False, zero_prior=False),
    "oracle_full": AttributionControl("oracle_full", None, use_oracle_tokens=True, zero_prior=False),
    "oracle_no_prior": AttributionControl("oracle_no_prior", "planner_no_prior", use_oracle_tokens=True, zero_prior=True),
    "oracle_gmr_only": AttributionControl("oracle_gmr_only", "gmr_only", use_oracle_tokens=True, zero_prior=False),
}


def evaluate_failure_attribution(
    *,
    primitive_root: Path,
    diffusion_checkpoint: Path,
    output_root: Path,
    device: str = "cpu",
    variant: str | None = None,
    rollout_sample_size: int = 32,
    bootstrap_samples: int = 1000,
    sample_seed: int = 7,
    wandb_run: Any | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    logger = logger or LOGGER
    output_root.mkdir(parents=True, exist_ok=True)
    pipeline = Sonata3Pipeline(primitive_root=primitive_root, diffusion_checkpoint=diffusion_checkpoint, device=device)
    action_horizon = int(pipeline.config["action_horizon"])
    state_context_steps = int(pipeline.config["state_context_steps"])
    token_df, metadata, prior_lookup = load_diffusion_inputs(
        primitive_root,
        action_horizon=action_horizon,
        state_context_steps=state_context_steps,
        family_mapping_mode=str(pipeline.config.get("family_mapping_mode", "heuristic_stats")),
        continuous_param_names=pipeline.config.get("continuous_param_names"),
    )
    dataset = DiffusionChunkDataset(
        token_df=token_df,
        metadata=metadata,
        primitive_root=primitive_root,
        split="val",
        context_length=int(pipeline.config["context_length"]),
        action_horizon=action_horizon,
        state_context_steps=state_context_steps,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(pipeline.config["batch_size"]),
        shuffle=False,
        num_workers=0,
        collate_fn=partial(diffusion_collate_fn, metadata=metadata, prior_lookup=prior_lookup),
    )
    sample_manifest = sample_rollout_manifest(dataset, sample_size=rollout_sample_size, seed=sample_seed)
    write_table(sample_manifest, output_root / "attribution_manifest")

    segment_rows, predictions_by_mode = collect_offline_attribution(
        pipeline=pipeline,
        loader=loader,
        metadata=metadata,
        prior_lookup=prior_lookup,
        variant=variant,
        rollout_episode_ids=set(sample_manifest["episode_id"].astype(str).tolist()),
    )
    segment_df = pd.DataFrame(segment_rows)
    write_table(segment_df, output_root / "attribution_segment_metrics")

    episode_rows: list[dict[str, Any]] = []
    for control_mode in CONTROL_MODES:
        rollout_output = evaluate_dm_control_rollout(
            primitive_root=primitive_root,
            predictions_by_episode=predictions_by_mode[control_mode],
            output_root=output_root / "rollout" / control_mode,
            limit_episodes=len(predictions_by_mode[control_mode]),
            render_video=False,
            wandb_run=None,
            logger=logger,
        )
        for row in rollout_output.get("episodes", []):
            episode_rows.append({"control_mode": control_mode, **row})
    episode_df = pd.DataFrame(episode_rows)
    write_table(episode_df, output_root / "attribution_episode_metrics")

    summary, bootstrap_df = summarize_attribution(segment_df=segment_df, episode_df=episode_df, bootstrap_samples=bootstrap_samples, seed=sample_seed)
    write_table(bootstrap_df, output_root / "attribution_bootstrap")
    write_json(summary, output_root / "attribution_summary.json")
    log_prefixed_metrics(wandb_run, summary, prefix="attribution", summary=True)
    log_rollout_table(wandb_run, key="attribution/episodes_table", dataframe=episode_df, logger=logger)
    log_rollout_table(wandb_run, key="attribution/segments_table", dataframe=segment_df, logger=logger)
    log_rollout_table(wandb_run, key="attribution/bootstrap_table", dataframe=bootstrap_df, logger=logger)
    return {
        "summary": summary,
        "segment_metrics": segment_df,
        "episode_metrics": episode_df,
        "bootstrap": bootstrap_df,
        "manifest": sample_manifest,
    }


def sample_rollout_manifest(dataset: DiffusionChunkDataset, *, sample_size: int, seed: int) -> pd.DataFrame:
    episode_ids = sorted({str(item["episode_id"]) for item in dataset.samples})
    if sample_size <= 0 or sample_size >= len(episode_ids):
        selected = episode_ids
    else:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(len(episode_ids), size=sample_size, replace=False))
        selected = [episode_ids[int(index)] for index in indices]
    return pd.DataFrame(
        {
            "episode_id": selected,
            "sample_seed": int(seed),
            "rollout_sample_size": int(len(selected)),
        }
    )


def collect_offline_attribution(
    *,
    pipeline: Sonata3Pipeline,
    loader: DataLoader,
    metadata: Any,
    prior_lookup: dict[str, np.ndarray],
    variant: str | None,
    rollout_episode_ids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, list[dict[str, Any]]]]]:
    segment_rows: list[dict[str, Any]] = []
    predictions_by_mode: dict[str, dict[str, list[dict[str, Any]]]] = {mode: defaultdict(list) for mode in CONTROL_MODES}
    with torch.no_grad():
        for batch in loader:
            moved = {key: value.to(pipeline.device) if hasattr(value, "to") else value for key, value in batch.items()}
            oracle_tokens = build_oracle_tokens(moved)
            predicted_tokens = predict_batch_tokens(pipeline, moved)
            per_mode_predictions: dict[str, torch.Tensor] = {}
            for control_mode in CONTROL_MODES:
                controlled_batch, control_oracle = build_controlled_batch(
                    batch=moved,
                    metadata=metadata,
                    prior_lookup=prior_lookup,
                    control_mode=control_mode,
                    oracle_tokens=oracle_tokens,
                )
                local_variant = variant if control_mode == "predicted_full" else CONTROL_CONFIGS[control_mode].variant
                per_mode_predictions[control_mode] = pipeline.predict_batch(
                    controlled_batch,
                    variant=local_variant,
                    oracle_tokens=control_oracle,
                )
            target = moved["action_target"]
            prior = moved["gmr_prior"]
            batch_size = target.shape[0]
            for index in range(batch_size):
                episode_id = str(batch["episode_id"][index])
                row = {
                    "episode_id": episode_id,
                    "onset_step": int(batch["onset_step"][index]),
                    "end_step": int(batch["end_step"][index]),
                    "target_primitive": int(moved["target_primitive"][index].item()),
                    "predicted_primitive": int(predicted_tokens["primitive_index"][index].item()),
                    "target_family": int(moved["target_family"][index].item()),
                    "predicted_family": int(predicted_tokens["family_index"][index].item()),
                    "target_duration": int(moved["target_duration"][index].item()),
                    "predicted_duration": int(predicted_tokens["duration_bucket"][index].item()),
                    "target_dynamics": int(moved["target_dynamics"][index].item()),
                    "predicted_dynamics": int(predicted_tokens["dynamics_bucket"][index].item()),
                    "primitive_correct": int(moved["target_primitive"][index].item() == predicted_tokens["primitive_index"][index].item()),
                    "family_correct": int(moved["target_family"][index].item() == predicted_tokens["family_index"][index].item()),
                    "duration_correct": int(moved["target_duration"][index].item() == predicted_tokens["duration_bucket"][index].item()),
                    "dynamics_correct": int(moved["target_dynamics"][index].item() == predicted_tokens["dynamics_bucket"][index].item()),
                    "prior_action_mse": float(torch.mean((prior[index] - target[index]) ** 2).item()),
                    "prior_action_l1": float(torch.mean(torch.abs(prior[index] - target[index])).item()),
                }
                for control_mode, prediction in per_mode_predictions.items():
                    pred = prediction[index]
                    row[f"{control_mode}/action_mse"] = float(torch.mean((pred - target[index]) ** 2).item())
                    row[f"{control_mode}/action_l1"] = float(torch.mean(torch.abs(pred - target[index])).item())
                    row[f"{control_mode}/smoothness"] = float(torch.mean((pred[1:] - pred[:-1]) ** 2).item())
                    if episode_id in rollout_episode_ids:
                        predictions_by_mode[control_mode][episode_id].append(
                            {
                                "predicted": pred.detach().cpu().numpy(),
                                "prior": prior[index].detach().cpu().numpy(),
                                "onset_step": int(batch["onset_step"][index]),
                                "end_step": int(batch["end_step"][index]),
                            }
                        )
                segment_rows.append(row)
    return segment_rows, predictions_by_mode


def build_oracle_tokens(batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {
        "family_index": batch["target_family"],
        "primitive_index": batch["target_primitive"],
        "duration_bucket": batch["target_duration"],
        "dynamics_bucket": batch["target_dynamics"],
    }


def predict_batch_tokens(pipeline: Sonata3Pipeline, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if pipeline.planner is None:
        return build_oracle_tokens(batch)
    outputs = pipeline.planner(batch)
    return {
        "family_index": outputs["family_logits"].argmax(dim=-1),
        "primitive_index": outputs["primitive_logits"].argmax(dim=-1),
        "duration_bucket": outputs["duration_logits"].argmax(dim=-1),
        "dynamics_bucket": outputs["dynamics_logits"].argmax(dim=-1),
    }


def build_controlled_batch(
    *,
    batch: dict[str, Any],
    metadata: Any,
    prior_lookup: dict[str, np.ndarray],
    control_mode: str,
    oracle_tokens: dict[str, torch.Tensor],
) -> tuple[dict[str, Any], dict[str, torch.Tensor] | None]:
    control = CONTROL_CONFIGS[control_mode]
    controlled = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    if control.use_oracle_tokens:
        controlled["primitive_index"] = oracle_tokens["primitive_index"].clone()
        controlled["duration_bucket"] = oracle_tokens["duration_bucket"].clone()
        controlled["dynamics_bucket"] = oracle_tokens["dynamics_bucket"].clone()
        primitive_ids = [int(item) for item in oracle_tokens["primitive_index"].detach().cpu().tolist()]
        oracle_prior = np.stack(
            [prior_lookup[str(metadata.primitive_ids[index])] for index in primitive_ids],
            axis=0,
        ).astype(np.float32)
        controlled["gmr_prior"] = torch.from_numpy(oracle_prior).to(batch["gmr_prior"].device)
        if control.zero_prior:
            controlled["gmr_prior"] = torch.zeros_like(controlled["gmr_prior"])
        return controlled, {key: value.clone() for key, value in oracle_tokens.items()}
    if control.zero_prior:
        controlled["gmr_prior"] = torch.zeros_like(controlled["gmr_prior"])
    return controlled, None


def summarize_attribution(
    *,
    segment_df: pd.DataFrame,
    episode_df: pd.DataFrame,
    bootstrap_samples: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    delta_specs = {
        "planner_impact": ("oracle_full", "predicted_full"),
        "prior_benefit": ("oracle_full", "oracle_no_prior"),
        "diffusion_benefit": ("oracle_full", "oracle_gmr_only"),
    }
    episode_wide = pivot_episode_metrics(episode_df)
    bootstrap_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    scorecard: dict[str, float] = {}
    for label, (lhs, rhs) in delta_specs.items():
        for metric in UTILITY_METRICS:
            delta = paired_delta(episode_wide, lhs=lhs, rhs=rhs, metric=metric)
            bootstrap = bootstrap_mean(delta, samples=bootstrap_samples, seed=seed)
            bootstrap_rows.append({"delta_name": label, "metric": metric, **bootstrap})
            summary[f"{label}/{metric}_mean"] = bootstrap["mean"]
            summary[f"{label}/{metric}_ci_low"] = bootstrap["ci_low"]
            summary[f"{label}/{metric}_ci_high"] = bootstrap["ci_high"]
        for metric in LOSS_METRICS:
            lhs_col = f"{lhs}/{metric}"
            rhs_col = f"{rhs}/{metric}"
            if lhs_col in segment_df.columns and rhs_col in segment_df.columns:
                delta = pd.to_numeric(segment_df[rhs_col], errors="coerce") - pd.to_numeric(segment_df[lhs_col], errors="coerce")
                summary[f"{label}/{metric}_mean"] = float(delta.mean()) if not delta.empty else 0.0
        scorecard[label] = confidence_score(summary, label)
    dominant = max(scorecard, key=scorecard.get) if scorecard else "mixed_low_confidence"
    if scorecard.get(dominant, 0.0) <= 0.0:
        dominant = "mixed_low_confidence"
    summary["dominant_bottleneck"] = dominant
    summary["confidence_score"] = float(scorecard.get(dominant, 0.0)) if dominant != "mixed_low_confidence" else 0.0
    summary["num_rollout_episodes"] = int(episode_wide.shape[0])
    summary["num_segments"] = int(len(segment_df))
    return summary, pd.DataFrame(bootstrap_rows)


def pivot_episode_metrics(episode_df: pd.DataFrame) -> pd.DataFrame:
    if episode_df.empty:
        return pd.DataFrame()
    metric_cols = ["episode_id", "control_mode", *UTILITY_METRICS]
    available = [column for column in metric_cols if column in episode_df.columns]
    frame = episode_df[available].copy()
    return frame.pivot_table(index="episode_id", columns="control_mode", values=list(UTILITY_METRICS), aggfunc="first")


def paired_delta(frame: pd.DataFrame, *, lhs: str, rhs: str, metric: str) -> np.ndarray:
    if frame.empty or (metric, lhs) not in frame.columns or (metric, rhs) not in frame.columns:
        return np.zeros((0,), dtype=np.float32)
    lhs_values = pd.to_numeric(frame[(metric, lhs)], errors="coerce")
    rhs_values = pd.to_numeric(frame[(metric, rhs)], errors="coerce")
    valid = lhs_values.notna() & rhs_values.notna()
    return (lhs_values[valid] - rhs_values[valid]).to_numpy(dtype=np.float32)


def bootstrap_mean(values: np.ndarray, *, samples: int, seed: int) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "support": 0}
    rng = np.random.default_rng(seed)
    draws = np.empty((max(int(samples), 1),), dtype=np.float32)
    for index in range(draws.shape[0]):
        sampled = rng.choice(array, size=array.shape[0], replace=True)
        draws[index] = float(np.mean(sampled))
    return {
        "mean": float(np.mean(array)),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "support": int(array.shape[0]),
    }


def confidence_score(summary: dict[str, Any], label: str) -> float:
    score = 0.0
    for metric in ("f1", "reward", "sustain_f1"):
        mean = float(summary.get(f"{label}/{metric}_mean", 0.0))
        ci_low = float(summary.get(f"{label}/{metric}_ci_low", 0.0))
        if mean > 0.0 and ci_low > 0.0:
            score += 1.0
    return score
