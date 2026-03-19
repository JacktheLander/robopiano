from __future__ import annotations

from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from sonata.diffusion.dataset import DiffusionChunkDataset, diffusion_collate_fn, load_diffusion_inputs
from sonata.models.pipeline import Sonata3Pipeline
from sonata.utils.io import save_npz, write_json, write_table


def evaluate_offline_pipeline(
    *,
    primitive_root: Path,
    diffusion_checkpoint: Path,
    output_root: Path,
    split: str = "val",
    variant: str | None = None,
    max_batches: int | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    pipeline = Sonata3Pipeline(primitive_root=primitive_root, diffusion_checkpoint=diffusion_checkpoint, device=device)
    action_horizon = int(pipeline.config["action_horizon"])
    state_context_steps = int(pipeline.config["state_context_steps"])
    token_df, metadata, prior_lookup = load_diffusion_inputs(primitive_root, action_horizon=action_horizon, state_context_steps=state_context_steps)
    dataset = DiffusionChunkDataset(
        token_df=token_df,
        primitive_root=primitive_root,
        split=split,
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
    losses: list[float] = []
    l1_losses: list[float] = []
    smoothness: list[float] = []
    prior_losses: list[float] = []
    predictions_by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sample_payload = None
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            predicted = pipeline.predict_batch(batch, variant=variant)
            target = batch["action_target"].to(pipeline.device)
            prior = batch["gmr_prior"].to(pipeline.device)
            losses.append(float(torch.mean((predicted - target) ** 2).item()))
            l1_losses.append(float(torch.mean(torch.abs(predicted - target)).item()))
            smoothness.append(float(torch.mean((predicted[:, 1:] - predicted[:, :-1]) ** 2).item()))
            prior_losses.append(float(torch.mean((prior - target) ** 2).item()))
            if sample_payload is None:
                sample_payload = {
                    "predicted": predicted[:8].cpu().numpy(),
                    "target": target[:8].cpu().numpy(),
                    "prior": prior[:8].cpu().numpy(),
                }
            for offset, episode_id in enumerate(batch["episode_id"] if "episode_id" in batch else []):
                predictions_by_episode[str(episode_id)].append(
                    {
                        "predicted": predicted[offset].cpu().numpy(),
                        "prior": prior[offset].cpu().numpy(),
                        "onset_step": int(batch["onset_step"][offset]),
                        "end_step": int(batch["end_step"][offset]),
                    }
                )
            if max_batches is not None and batch_index + 1 >= max_batches:
                break
    metrics = {
        "split": split,
        "variant": variant or str(pipeline.config["variant"]),
        "action_mse": float(np.mean(losses)) if losses else 0.0,
        "action_l1": float(np.mean(l1_losses)) if l1_losses else 0.0,
        "smoothness": float(np.mean(smoothness)) if smoothness else 0.0,
        "prior_action_mse": float(np.mean(prior_losses)) if prior_losses else 0.0,
        "improvement_over_prior": float(np.mean(prior_losses) - np.mean(losses)) if losses else 0.0,
    }
    write_json(metrics, output_root / "offline_metrics.json")
    if sample_payload is not None:
        save_npz(output_root / "offline_samples.npz", **sample_payload)
    write_table(pd.DataFrame([metrics]), output_root / "offline_metrics")
    return {"metrics": metrics, "predictions_by_episode": predictions_by_episode}


def stitch_segment_predictions(token_df: pd.DataFrame, episode_predictions: list[dict[str, Any]], action_horizon: int) -> np.ndarray:
    if not episode_predictions:
        return np.zeros((1, 1), dtype=np.float32)
    ordered = sorted(episode_predictions, key=lambda item: (item["onset_step"], item["end_step"]))
    max_step = max(int(item["end_step"]) for item in ordered)
    action_dim = int(ordered[0]["predicted"].shape[-1])
    stitched = np.zeros((max_step, action_dim), dtype=np.float32)
    counts = np.zeros((max_step, 1), dtype=np.float32)
    for item in ordered:
        duration = max(int(item["end_step"]) - int(item["onset_step"]), 1)
        resampled = resample_prediction(item["predicted"], duration)
        start = int(item["onset_step"])
        end = min(start + duration, stitched.shape[0])
        stitched[start:end] += resampled[: end - start]
        counts[start:end] += 1.0
    return stitched / np.clip(counts, 1.0, None)


def resample_prediction(prediction: np.ndarray, steps: int) -> np.ndarray:
    x_old = np.linspace(0.0, 1.0, prediction.shape[0], dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, steps, dtype=np.float32)
    output = np.zeros((steps, prediction.shape[1]), dtype=np.float32)
    for dim in range(prediction.shape[1]):
        output[:, dim] = np.interp(x_new, x_old, prediction[:, dim])
    return output
