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
        loss_config = build_loss_config(train_loader.dataset, metadata, config, model_type=model_type, device=device)
        num_parameters = count_parameters(model)
        logger.info("Model parameters: %d", num_parameters)
        logger.info(
            "Optimizer settings: lr=%.2e, min_lr=%.2e, warmup_epochs=%d, weight_decay=%.2e, dropout=%.2f, label_smoothing=%.2f, gradient_clip_norm=%.2f, class_weights=%s",
            float(config["learning_rate"]),
            float(config.get("min_learning_rate", 0.0)),
            int(config.get("lr_warmup_epochs", 0)),
            float(config.get("weight_decay", 0.0)),
            float(config.get("dropout", 0.0)),
            float(loss_config["label_smoothing"]),
            float(config.get("gradient_clip_norm", 0.0)),
            "enabled" if loss_config["primitive_class_weights"] is not None else "disabled",
        )
        if loss_config["primitive_class_weights"] is not None:
            weights = loss_config["primitive_class_weights"].detach().cpu()
            logger.info(
                "Primitive class weights: min=%.3f, max=%.3f, mean=%.3f",
                float(weights.min().item()),
                float(weights.max().item()),
                float(weights.mean().item()),
            )
        wandb_run.summary(
            {
                "stage": "transformer",
                "run_root": str(run_paths.root),
                "primitive_root": str(primitive_root),
                "model/num_parameters": num_parameters,
                "train/learning_rate": float(config["learning_rate"]),
                "train/min_learning_rate": float(config.get("min_learning_rate", 0.0)),
                "train/lr_warmup_epochs": int(config.get("lr_warmup_epochs", 0)),
                "train/weight_decay": float(config.get("weight_decay", 0.0)),
                "train/dropout": float(config.get("dropout", 0.0)),
                "train/label_smoothing": float(loss_config["label_smoothing"]),
                "train/gradient_clip_norm": float(config.get("gradient_clip_norm", 0.0)),
                "train/class_weights_enabled": loss_config["primitive_class_weights"] is not None,
            }
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
        scheduler = build_scheduler(optimizer, config)
        metrics_writer = MetricsWriter(run_paths.metrics)

        start_epoch = 0
        best_metric = float("inf")
        best_epoch = -1
        if bool(config.get("resume", False)):
            checkpoint = find_latest_checkpoint(run_paths.checkpoints)
            if checkpoint is not None:
                payload = load_checkpoint(checkpoint, map_location=device)
                model.load_state_dict(payload["model"])
                optimizer.load_state_dict(payload["optimizer"])
                scheduler.load_state_dict(payload["scheduler"])
                start_epoch = int(payload["epoch"]) + 1
                best_metric = float(payload.get("best_metric", best_metric))
                best_epoch = int(payload.get("best_epoch", best_epoch))
                logger.info("Resumed from %s", checkpoint)
                wandb_run.summary(
                    {
                        "resume/checkpoint": str(checkpoint),
                        "resume/start_epoch": start_epoch,
                        "resume/best_epoch": best_epoch,
                        "resume/best_val_loss": best_metric,
                    }
                )

        best_records: list[dict[str, Any]] = []
        min_delta = float(config.get("early_stopping_min_delta", 0.0))
        patience = int(config.get("early_stopping_patience", 0))
        patience_counter = 0
        stopped_early = False
        for epoch in range(start_epoch, int(config["epochs"])):
            current_lr = float(optimizer.param_groups[0]["lr"])
            train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                model_type,
                loss_config=loss_config,
                gradient_clip_norm=float(config.get("gradient_clip_norm", 0.0)),
            )
            val_metrics, records = evaluate(model, val_loader, device, model_type, loss_config=loss_config, topk=int(config["topk"]))
            monitor = float(val_metrics["loss"])
            improved = np.isfinite(monitor) and (best_epoch < 0 or monitor < (best_metric - min_delta))
            checkpoint_saved = False
            if improved:
                best_metric = monitor
                best_epoch = epoch
                best_records = records
                patience_counter = 0
                best_checkpoint = save_checkpoint(
                    run_paths.checkpoints / "best.pt",
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "config": config,
                        "metadata": metadata.__dict__,
                        "best_metric": best_metric,
                        "best_epoch": best_epoch,
                    },
                )
                checkpoint_saved = True
                wandb_run.summary({"best/epoch": epoch, "best/val_loss": monitor, "best/checkpoint": str(best_checkpoint)})
            else:
                patience_counter += 1
            row = {
                "epoch": epoch,
                "lr": current_lr,
                "best_epoch": best_epoch,
                "checkpoint_saved": checkpoint_saved,
                **{f"train/{key}": value for key, value in train_metrics.items()},
                **{f"val/{key}": value for key, value in val_metrics.items()},
            }
            metrics_writer.log(row)
            wandb_run.log(row, step=epoch)
            logger.info(
                "Epoch %d/%d lr=%.2e train=%s val=%s best_epoch=%d checkpoint_saved=%s patience=%d/%d",
                epoch,
                int(config["epochs"]) - 1,
                current_lr,
                train_metrics,
                val_metrics,
                best_epoch,
                checkpoint_saved,
                patience_counter,
                patience,
            )
            scheduler.step()
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
                        "best_metric": best_metric,
                        "best_epoch": best_epoch,
                    },
                )
            if patience > 0 and patience_counter >= patience:
                logger.info(
                    "Early stopping triggered at epoch %d after %d epoch(s) without val loss improvement beyond %.4f. Best epoch=%d best_val_loss=%.4f",
                    epoch,
                    patience_counter,
                    min_delta,
                    best_epoch,
                    best_metric,
                )
                wandb_run.summary(
                    {
                        "status": "early_stopped",
                        "early_stop/epoch": epoch,
                        "early_stop/patience": patience,
                        "early_stop/min_delta": min_delta,
                    }
                )
                stopped_early = True
                break

        if best_records:
            write_table(pd_from_records(best_records), run_paths.artifacts / "generated_sequences")
        best_checkpoint_path = run_paths.checkpoints / "best.pt"
        if not best_checkpoint_path.exists():
            logger.warning("Validation loss never produced a finite improvement; saving fallback best checkpoint from the final model state.")
            best_checkpoint_path = save_checkpoint(
                best_checkpoint_path,
                {
                    "epoch": max(start_epoch, int(config["epochs"]) - 1),
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "config": config,
                    "metadata": metadata.__dict__,
                    "best_metric": best_metric,
                    "best_epoch": best_epoch,
                },
            )
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
        wandb_run.summary(
            {
                "status": "early_stopped" if stopped_early else "completed",
                "best/checkpoint": str(best_checkpoint_path),
                "best/epoch": best_epoch,
                "best/val_loss": best_metric,
            }
        )
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


def build_scheduler(optimizer: torch.optim.Optimizer, config: dict[str, Any]) -> torch.optim.lr_scheduler.LRScheduler:
    total_epochs = max(int(config["epochs"]), 1)
    warmup_epochs = max(int(config.get("lr_warmup_epochs", 0)), 0)
    min_lr = float(config.get("min_learning_rate", 0.0))
    if warmup_epochs <= 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=min_lr)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=float(config.get("warmup_start_factor", 0.2)),
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine_epochs = max(total_epochs - warmup_epochs, 1)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=min_lr)
    return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


def build_loss_config(
    dataset,
    metadata,
    config: dict[str, Any],
    *,
    model_type: str,
    device: torch.device,
) -> dict[str, Any]:
    label_smoothing = float(config.get("label_smoothing", 0.0))
    class_weights = None
    if model_type == "token_prediction" and bool(config.get("use_class_weights", False)):
        class_weights = compute_primitive_class_weights(
            dataset,
            num_primitives=int(metadata.num_primitives),
            power=float(config.get("class_weight_power", 0.5)),
            max_weight=float(config.get("class_weight_max", 5.0)),
        ).to(device)
    return {"label_smoothing": label_smoothing, "primitive_class_weights": class_weights}


def compute_primitive_class_weights(dataset, *, num_primitives: int, power: float, max_weight: float) -> torch.Tensor:
    if not hasattr(dataset, "samples") or not dataset.samples:
        return torch.ones(num_primitives, dtype=torch.float32)
    targets = np.fromiter((int(sample["target_primitive"]) for sample in dataset.samples), dtype=np.int64)
    counts = np.bincount(targets, minlength=num_primitives).astype(np.float32)
    counts = np.clip(counts, a_min=1.0, a_max=None)
    weights = np.power(counts, -power, dtype=np.float32)
    weights = weights / max(float(weights.mean()), 1e-8)
    if max_weight > 0.0:
        weights = np.clip(weights, a_min=0.0, a_max=max_weight)
        weights = weights / max(float(weights.mean()), 1e-8)
    return torch.from_numpy(weights.astype(np.float32))


def train_one_epoch(model, loader, optimizer, device, model_type: str, *, loss_config: dict[str, Any], gradient_clip_norm: float) -> dict[str, float]:
    model.train()
    losses: list[float] = []
    primitive_acc: list[float] = []
    for batch in tqdm(loader, desc="Train transformer", leave=False):
        batch = move_to_device(batch, str(device))
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch)
        loss, metrics = compute_loss(outputs, batch, model_type, loss_config=loss_config)
        loss.backward()
        if gradient_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
        optimizer.step()
        losses.append(float(loss.item()))
        primitive_acc.append(float(metrics.get("primitive_accuracy", 0.0)))
    return {"loss": float(np.mean(losses)), "primitive_accuracy": float(np.mean(primitive_acc))}


@torch.no_grad()
def evaluate(model, loader, device, model_type: str, *, loss_config: dict[str, Any], topk: int) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    losses: list[float] = []
    primitive_acc: list[float] = []
    topk_acc: list[float] = []
    confidence: list[float] = []
    generated: list[dict[str, Any]] = []
    for batch in tqdm(loader, desc="Eval transformer", leave=False):
        batch = move_to_device(batch, str(device))
        outputs = model(batch)
        loss, metrics = compute_loss(outputs, batch, model_type, loss_config=loss_config)
        loss_value = float(loss.item())
        if np.isfinite(loss_value):
            losses.append(loss_value)
        primitive_acc.append(float(metrics.get("primitive_accuracy", 0.0)))
        if "primitive_logits" in outputs:
            primitive_logits = torch.nan_to_num(outputs["primitive_logits"])
            topk_hits = topk_accuracy(primitive_logits, batch["target_primitive"], topk=topk)
            topk_acc.append(float(topk_hits))
            confidence.append(float(F.softmax(primitive_logits, dim=-1).amax(dim=-1).mean().item()))
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
        "avg_max_confidence": float(np.mean(confidence)) if confidence else 0.0,
    }, generated


def compute_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    model_type: str,
    *,
    loss_config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    if model_type == "direct_transformer_action":
        pred = outputs["action_pred"]
        target = batch["action_target"]
        loss = F.mse_loss(pred, target)
        return loss, {"primitive_accuracy": 0.0}
    primitive_logits = torch.nan_to_num(outputs["primitive_logits"])
    duration_logits = torch.nan_to_num(outputs["duration_logits"])
    dynamics_logits = torch.nan_to_num(outputs["dynamics_logits"])
    label_smoothing = float(loss_config.get("label_smoothing", 0.0))
    primitive_loss = F.cross_entropy(
        primitive_logits,
        batch["target_primitive"],
        weight=loss_config.get("primitive_class_weights"),
        label_smoothing=label_smoothing,
    )
    duration_loss = F.cross_entropy(duration_logits, batch["target_duration"], label_smoothing=label_smoothing)
    dynamics_loss = F.cross_entropy(dynamics_logits, batch["target_dynamics"], label_smoothing=label_smoothing)
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
