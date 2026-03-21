from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from sonata.transformer.dataset import (
    PrimitiveSequenceDataset,
    TransformerActionDataset,
    action_collate_fn,
    load_transformer_inputs,
    planner_collate_fn,
)
from sonata.transformer.model import PrimitivePlannerTransformer, TransformerActionRegressor
from sonata.utils.checkpointing import find_latest_checkpoint, load_checkpoint, save_checkpoint
from sonata.utils.experiment import make_run_paths
from sonata.utils.io import write_json, write_table
from sonata.utils.metrics import MetricsWriter
from sonata.utils.random import set_seed
from sonata.utils.torch_utils import count_parameters, move_to_device
from sonata.utils.wandb import WandbRun


def run_transformer_training(config: dict[str, Any], logger: logging.Logger) -> dict[str, Path]:
    primitive_root = Path(config["primitive_root"]).resolve()
    output_root = Path(config["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    token_df, metadata = load_transformer_inputs(primitive_root)
    run_paths = make_run_paths(output_root, "transformer", config["experiment_name"], int(config["seed"]), resume=bool(config.get("resume", False)))
    logger.info("Transformer run directory: %s", run_paths.root)
    write_json(config, run_paths.artifacts / "config.json")
    wandb_run = WandbRun(
        config.get("wandb"),
        run_name=run_paths.root.name,
        config_payload=config,
        logger=logger,
        job_type="transformer",
        tags=["sonata", "transformer"],
    )
    try:
        set_seed(int(config["seed"]), deterministic=bool(config.get("deterministic_eval", False)))
        model_type = str(config["model_variant"])
        train_loader, val_loader, model = build_dataloaders_and_model(token_df, metadata, primitive_root, config)
        device = torch.device(config["device"])
        model.to(device)
        num_parameters = count_parameters(model)
        logger.info("Model parameters: %d", num_parameters)
        wandb_run.summary(
            {
                "stage": "transformer",
                "run_root": str(run_paths.root),
                "primitive_root": str(primitive_root),
                "model/num_parameters": num_parameters,
            }
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(config["epochs"]), 1))
        metrics_writer = MetricsWriter(run_paths.metrics)

        start_epoch = 0
        if bool(config.get("resume", False)):
            checkpoint = find_latest_checkpoint(run_paths.checkpoints)
            if checkpoint is not None:
                payload = load_checkpoint(checkpoint, map_location=device)
                model.load_state_dict(payload["model"])
                optimizer.load_state_dict(payload["optimizer"])
                scheduler.load_state_dict(payload["scheduler"])
                start_epoch = int(payload["epoch"]) + 1
                logger.info("Resumed from %s", checkpoint)
                wandb_run.summary({"resume/checkpoint": str(checkpoint), "resume/start_epoch": start_epoch})

        best_metric = float("inf")
        generated_records: list[dict[str, Any]] = []
        for epoch in range(start_epoch, int(config["epochs"])):
            train_metrics = train_one_epoch(model, train_loader, optimizer, device, model_type)
            val_metrics, records = evaluate(model, val_loader, device, model_type, topk=int(config["topk"]))
            generated_records = records
            row = {"epoch": epoch, **{f"train/{key}": value for key, value in train_metrics.items()}, **{f"val/{key}": value for key, value in val_metrics.items()}}
            metrics_writer.log(row)
            wandb_run.log(row, step=epoch)
            logger.info("Epoch %d train=%s val=%s", epoch, train_metrics, val_metrics)
            scheduler.step()
            monitor = float(val_metrics["loss"])
            if epoch == start_epoch or monitor < best_metric:
                best_metric = monitor if np.isfinite(monitor) else best_metric
                best_checkpoint = save_checkpoint(
                    run_paths.checkpoints / "best.pt",
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "config": config,
                        "metadata": metadata.__dict__,
                    },
                )
                wandb_run.summary({"best/epoch": epoch, "best/val_loss": monitor, "best/checkpoint": str(best_checkpoint)})
            if (epoch + 1) % int(config["checkpoint_interval"]) == 0:
                save_checkpoint(
                    run_paths.checkpoints / f"epoch_{epoch:04d}.pt",
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "config": config,
                        "metadata": metadata.__dict__,
                    },
                )

        if generated_records:
            write_table(pd_from_records(generated_records), run_paths.artifacts / "generated_sequences")
        best_checkpoint_path = run_paths.checkpoints / "best.pt"
        wandb_run.log_artifact_bundle(
            artifact_name=f"{run_paths.root.name}-checkpoints",
            artifact_type="model",
            entries={"checkpoints": run_paths.checkpoints},
            aliases=["latest", "best"],
            metadata={"stage": "transformer", "run_root": str(run_paths.root)},
        )
        wandb_run.log_artifact_bundle(
            artifact_name=f"{run_paths.root.name}-run",
            artifact_type="run-output",
            entries={
                "metrics": run_paths.metrics,
                "artifacts": run_paths.artifacts,
                "plots": run_paths.plots,
                "logs": run_paths.logs,
            },
            aliases=["latest"],
            metadata={"stage": "transformer", "run_root": str(run_paths.root)},
        )
        wandb_run.summary({"status": "completed", "best/checkpoint": str(best_checkpoint_path)})
        return {"run_root": run_paths.root, "best_checkpoint": best_checkpoint_path}
    finally:
        wandb_run.finish()


def build_dataloaders_and_model(token_df, metadata, primitive_root: Path, config: dict[str, Any]):
    context_length = int(config["context_length"])
    batch_size = int(config["batch_size"])
    num_workers = int(config.get("num_workers", 0))
    planner = PrimitivePlannerTransformer(
        num_primitives=metadata.num_primitives,
        num_duration_buckets=metadata.num_duration_buckets,
        num_dynamics_buckets=metadata.num_dynamics_buckets,
        score_dim=metadata.score_dim,
        d_model=int(config["d_model"]),
        nhead=int(config["nhead"]),
        num_layers=int(config["num_layers"]),
        dim_feedforward=int(config["dim_feedforward"]),
        dropout=float(config["dropout"]),
        max_length=context_length,
    )
    if config["model_variant"] == "direct_transformer_action":
        train_dataset = TransformerActionDataset(token_df, primitive_root, context_length=context_length, action_horizon=int(config["action_horizon"]), split="train")
        val_dataset = TransformerActionDataset(token_df, primitive_root, context_length=context_length, action_horizon=int(config["action_horizon"]), split="val")
        action_dim = int(train_dataset[0]["action_target"].shape[-1]) if len(train_dataset) > 0 else int(config.get("fallback_action_dim", 39))
        model = TransformerActionRegressor(planner=planner, action_horizon=int(config["action_horizon"]), action_dim=action_dim)
        collate = partial(action_collate_fn, metadata=metadata)
    else:
        train_dataset = PrimitiveSequenceDataset(token_df, context_length=context_length, split="train")
        val_dataset = PrimitiveSequenceDataset(token_df, context_length=context_length, split="val")
        model = planner
        collate = partial(planner_collate_fn, metadata=metadata)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate)
    return train_loader, val_loader, model


def train_one_epoch(model, loader, optimizer, device, model_type: str) -> dict[str, float]:
    model.train()
    losses: list[float] = []
    primitive_acc: list[float] = []
    for batch in tqdm(loader, desc="Train transformer", leave=False):
        batch = move_to_device(batch, str(device))
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch)
        loss, metrics = compute_loss(outputs, batch, model_type)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        primitive_acc.append(float(metrics.get("primitive_accuracy", 0.0)))
    return {"loss": float(np.mean(losses)), "primitive_accuracy": float(np.mean(primitive_acc))}


@torch.no_grad()
def evaluate(model, loader, device, model_type: str, topk: int) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    losses: list[float] = []
    primitive_acc: list[float] = []
    topk_acc: list[float] = []
    generated: list[dict[str, Any]] = []
    for batch in tqdm(loader, desc="Eval transformer", leave=False):
        batch = move_to_device(batch, str(device))
        outputs = model(batch)
        loss, metrics = compute_loss(outputs, batch, model_type)
        loss_value = float(loss.item())
        if np.isfinite(loss_value):
            losses.append(loss_value)
        primitive_acc.append(float(metrics.get("primitive_accuracy", 0.0)))
        if "primitive_logits" in outputs:
            primitive_logits = torch.nan_to_num(outputs["primitive_logits"])
            topk_hits = topk_accuracy(primitive_logits, batch["target_primitive"], topk=topk)
            topk_acc.append(float(topk_hits))
            predictions = primitive_logits.argmax(dim=-1)
            for truth, pred in zip(batch["target_primitive"].tolist(), predictions.tolist()):
                generated.append({"target_primitive": int(truth), "predicted_primitive": int(pred)})
    val_loss = float(np.mean(losses)) if losses else float("inf")
    perplexity = float(np.exp(val_loss)) if model_type == "token_prediction" and np.isfinite(val_loss) else float("inf")
    return {
        "loss": val_loss,
        "primitive_accuracy": float(np.mean(primitive_acc)) if primitive_acc else 0.0,
        "topk_accuracy": float(np.mean(topk_acc)) if topk_acc else 0.0,
        "perplexity": perplexity,
    }, generated


def compute_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], model_type: str) -> tuple[torch.Tensor, dict[str, float]]:
    if model_type == "direct_transformer_action":
        pred = outputs["action_pred"]
        target = batch["action_target"]
        loss = F.mse_loss(pred, target)
        return loss, {"primitive_accuracy": 0.0}
    primitive_logits = torch.nan_to_num(outputs["primitive_logits"])
    duration_logits = torch.nan_to_num(outputs["duration_logits"])
    dynamics_logits = torch.nan_to_num(outputs["dynamics_logits"])
    primitive_loss = F.cross_entropy(primitive_logits, batch["target_primitive"])
    duration_loss = F.cross_entropy(duration_logits, batch["target_duration"])
    dynamics_loss = F.cross_entropy(dynamics_logits, batch["target_dynamics"])
    loss = primitive_loss + 0.5 * duration_loss + 0.25 * dynamics_loss
    predictions = primitive_logits.argmax(dim=-1)
    primitive_accuracy = (predictions == batch["target_primitive"]).float().mean().item()
    return loss, {"primitive_accuracy": float(primitive_accuracy)}


def topk_accuracy(logits: torch.Tensor, target: torch.Tensor, topk: int) -> float:
    _, indices = logits.topk(min(topk, logits.shape[-1]), dim=-1)
    hits = (indices == target.unsqueeze(-1)).any(dim=-1).float().mean().item()
    return float(hits)


def pd_from_records(records: list[dict[str, Any]]):
    import pandas as pd

    return pd.DataFrame(records)
