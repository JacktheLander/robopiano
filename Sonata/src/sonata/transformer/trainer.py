from __future__ import annotations

import logging
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from sonata.transformer.dataset import (
    PlannerMetadata,
    PrimitiveSequenceDataset,
    TransformerActionDataset,
    action_collate_fn,
    family_mapping_records,
    family_mask_tensor,
    load_transformer_inputs,
    planner_collate_fn,
)
from sonata.transformer.model import (
    FACTORED_PLANNER_ARCHITECTURE,
    PrimitivePlannerTransformer,
    TransformerActionRegressor,
    build_planner_from_config,
)
from sonata.utils.checkpointing import find_latest_checkpoint, load_checkpoint, save_checkpoint
from sonata.utils.experiment import make_run_paths
from sonata.utils.io import write_json, write_table
from sonata.utils.metrics import MetricsWriter
from sonata.utils.random import set_seed
from sonata.utils.torch_utils import count_parameters, move_to_device
from sonata.utils.wandb import WandbRun

LEGACY_VARIANT_ALIASES = {"token_prediction": "factored_goal_conditioned"}
SUPPORTED_MODEL_VARIANTS = {"factored_goal_conditioned", "direct_transformer_action"}


def run_transformer_training(config: dict[str, Any], logger: logging.Logger) -> dict[str, Path]:
    config = normalize_transformer_config(config)
    validate_transformer_config(config)
    primitive_root = Path(config["primitive_root"]).resolve()
    output_root = Path(config["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    token_df, metadata = load_transformer_inputs(
        primitive_root,
        family_mapping_mode=str(config.get("family_mapping_mode", "heuristic_stats")),
        continuous_param_names=config.get("continuous_param_names"),
    )
    run_paths = make_run_paths(output_root, "transformer", config["experiment_name"], int(config["seed"]), resume=bool(config.get("resume", False)))
    logger.info("Transformer run directory: %s", run_paths.root)
    write_json(config, run_paths.artifacts / "config.json")
    write_json(metadata.to_payload(), run_paths.artifacts / "planner_metadata.json")
    write_table(pd.DataFrame(family_mapping_records(metadata)), run_paths.artifacts / "primitive_family_mapping")
    if metadata.continuous_param_dim > 0:
        write_table(
            pd.DataFrame(
                {
                    "continuous_param": metadata.continuous_param_names,
                    "mean": metadata.continuous_param_mean,
                    "std": metadata.continuous_param_std,
                }
            ),
            run_paths.artifacts / "continuous_param_stats",
        )
    wandb_run = WandbRun(
        config.get("wandb"),
        run_name=run_paths.root.name,
        config_payload=config,
        logger=logger,
        job_type="transformer",
        tags=["sonata", "transformer", str(config["model_variant"])],
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
            "Optimizer settings: lr=%.2e min_lr=%.2e warmup_epochs=%d weight_decay=%.2e dropout=%.2f label_smoothing=%.2f focal=%s class_balanced=%s balanced_sampler=%s",
            float(config["learning_rate"]),
            float(config.get("min_learning_rate", 0.0)),
            int(config.get("lr_warmup_epochs", 0)),
            float(config.get("weight_decay", 0.0)),
            float(config.get("dropout", 0.0)),
            float(loss_config["label_smoothing"]),
            "enabled" if loss_config["focal_heads"] else "disabled",
            "enabled" if loss_config["primitive_class_weights"] is not None else "disabled",
            bool(config.get("use_balanced_sampler", False)),
        )
        wandb_run.summary(
            {
                "stage": "transformer",
                "run_root": str(run_paths.root),
                "primitive_root": str(primitive_root),
                "model/num_parameters": num_parameters,
                "planner_architecture": FACTORED_PLANNER_ARCHITECTURE,
                "planner/variant": model_type,
                "planner/family_mapping_mode": metadata.family_mapping_mode,
                "planner/num_families": metadata.num_families,
                "planner/plan_embedding_dim": int(config["plan_embedding_dim"]),
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
                try:
                    model.load_state_dict(payload["model"])
                except RuntimeError as exc:
                    raise ValueError(
                        f"Failed to resume planner checkpoint {checkpoint}. "
                        "If this checkpoint predates the factored planner redesign, retrain Stage 2 with the updated configs."
                    ) from exc
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
        best_confusion: pd.DataFrame | None = None
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
            val_metrics, records, artifacts = evaluate(
                model,
                val_loader,
                device,
                model_type,
                metadata=metadata,
                loss_config=loss_config,
                topk=int(config["topk"]),
                eval_temperature=float(config.get("eval_temperature", 1.0)),
            )
            monitor = float(val_metrics["loss"])
            improved = np.isfinite(monitor) and (best_epoch < 0 or monitor < (best_metric - min_delta))
            checkpoint_saved = False
            if improved:
                best_metric = monitor
                best_epoch = epoch
                best_records = records
                best_confusion = artifacts.get("family_confusion")
                patience_counter = 0
                best_checkpoint = save_checkpoint(
                    run_paths.checkpoints / "best.pt",
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "config": config,
                        "metadata": metadata.to_payload(),
                        "planner_architecture": FACTORED_PLANNER_ARCHITECTURE,
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
                        "metadata": metadata.to_payload(),
                        "planner_architecture": FACTORED_PLANNER_ARCHITECTURE,
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
        if best_confusion is not None:
            write_table(best_confusion, run_paths.artifacts / "family_confusion_best")
        best_checkpoint_path = run_paths.checkpoints / "best.pt"
        if not best_checkpoint_path.exists():
            logger.warning("Validation never improved with a finite loss; saving fallback best checkpoint from the final model state.")
            best_checkpoint_path = save_checkpoint(
                best_checkpoint_path,
                {
                    "epoch": max(start_epoch, int(config["epochs"]) - 1),
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "config": config,
                    "metadata": metadata.to_payload(),
                    "planner_architecture": FACTORED_PLANNER_ARCHITECTURE,
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


def normalize_transformer_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(config)
    raw_variant = str(normalized.get("model_variant", "factored_goal_conditioned"))
    normalized["model_variant"] = LEGACY_VARIANT_ALIASES.get(raw_variant, raw_variant)
    normalized.setdefault("plan_embedding_dim", int(normalized.get("d_model", 256)))
    normalized.setdefault("family_mapping_mode", "heuristic_stats")
    normalized.setdefault("continuous_param_names", ["motion_energy", "chord_size", "start_state_norm", "end_state_norm"])
    normalized.setdefault("family_loss_weight", 1.0)
    normalized.setdefault("primitive_loss_weight", 1.0)
    normalized.setdefault("duration_loss_weight", 0.35)
    normalized.setdefault("dynamics_loss_weight", 0.25)
    normalized.setdefault("param_loss_weight", 0.1)
    normalized.setdefault("normalize_loss_by_active_weights", True)
    normalized.setdefault("use_focal_loss", True)
    normalized.setdefault("focal_heads", ["family", "primitive"])
    normalized.setdefault("focal_gamma", 1.5)
    normalized.setdefault("use_class_balanced_loss", bool(normalized.get("use_class_weights", False)))
    normalized.setdefault("class_balance_strategy", "effective_num")
    normalized.setdefault("class_balance_beta", 0.999)
    normalized.setdefault("class_weight_power", 0.5)
    normalized.setdefault("class_weight_max", 5.0)
    normalized.setdefault("use_balanced_sampler", False)
    normalized.setdefault("balanced_sampler_target", "family")
    normalized.setdefault("eval_temperature", 1.0)
    normalized.setdefault("topk", 5)
    return normalized


def validate_transformer_config(config: dict[str, Any]) -> None:
    variant = str(config["model_variant"])
    if variant not in SUPPORTED_MODEL_VARIANTS:
        raise ValueError(
            f"Unsupported model_variant={variant!r}. Expected one of: {sorted(SUPPORTED_MODEL_VARIANTS)}."
        )
    if int(config["plan_embedding_dim"]) <= 0:
        raise ValueError("plan_embedding_dim must be positive.")
    if int(config["context_length"]) <= 0:
        raise ValueError("context_length must be positive.")
    if float(config.get("label_smoothing", 0.0)) < 0.0 or float(config.get("label_smoothing", 0.0)) >= 1.0:
        raise ValueError("label_smoothing must be in [0, 1).")
    if float(config.get("eval_temperature", 1.0)) <= 0.0:
        raise ValueError("eval_temperature must be > 0.")
    if float(config.get("focal_gamma", 0.0)) < 0.0:
        raise ValueError("focal_gamma must be non-negative.")
    if str(config.get("balanced_sampler_target", "family")) not in {"family", "primitive", "hybrid"}:
        raise ValueError("balanced_sampler_target must be one of: family, primitive, hybrid.")


def build_dataloaders_and_model(token_df, metadata: PlannerMetadata, primitive_root: Path, config: dict[str, Any]):
    context_length = int(config["context_length"])
    batch_size = int(config["batch_size"])
    num_workers = int(config.get("num_workers", 0))
    planner = build_planner_from_config(metadata, config)
    if config["model_variant"] == "direct_transformer_action":
        train_dataset = TransformerActionDataset(
            token_df,
            metadata,
            primitive_root,
            context_length=context_length,
            action_horizon=int(config["action_horizon"]),
            split="train",
        )
        val_dataset = TransformerActionDataset(
            token_df,
            metadata,
            primitive_root,
            context_length=context_length,
            action_horizon=int(config["action_horizon"]),
            split="val",
        )
        action_dim = int(train_dataset[0]["action_target"].shape[-1]) if len(train_dataset) > 0 else int(config.get("fallback_action_dim", 39))
        model = TransformerActionRegressor(planner=planner, action_horizon=int(config["action_horizon"]), action_dim=action_dim)
        collate = partial(action_collate_fn, metadata=metadata)
    else:
        train_dataset = PrimitiveSequenceDataset(token_df, metadata, context_length=context_length, split="train")
        val_dataset = PrimitiveSequenceDataset(token_df, metadata, context_length=context_length, split="val")
        model = planner
        collate = partial(planner_collate_fn, metadata=metadata)
    train_sampler = build_balanced_sampler(train_dataset, config, seed=int(config["seed"]))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate)
    return train_loader, val_loader, model


def build_balanced_sampler(dataset, config: dict[str, Any], *, seed: int) -> WeightedRandomSampler | None:
    if not bool(config.get("use_balanced_sampler", False)):
        return None
    if not hasattr(dataset, "get_target_array"):
        return None
    if len(dataset) == 0:
        return None
    mode = str(config.get("balanced_sampler_target", "family"))
    if mode == "family":
        targets = dataset.get_target_array("family")
        weights = inverse_frequency_weights(targets, int(np.max(targets) + 1) if targets.size else 1)
    elif mode == "primitive":
        targets = dataset.get_target_array("primitive")
        weights = inverse_frequency_weights(targets, int(np.max(targets) + 1) if targets.size else 1)
    else:
        family_targets = dataset.get_target_array("family")
        primitive_targets = dataset.get_target_array("primitive")
        family_weights = inverse_frequency_weights(family_targets, int(np.max(family_targets) + 1) if family_targets.size else 1)
        primitive_weights = inverse_frequency_weights(primitive_targets, int(np.max(primitive_targets) + 1) if primitive_targets.size else 1)
        weights = np.sqrt(family_weights * primitive_weights).astype(np.float32)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights.astype(np.float32)),
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )


def inverse_frequency_weights(targets: np.ndarray, num_classes: int) -> np.ndarray:
    if targets.size == 0:
        return np.ones((0,), dtype=np.float32)
    counts = np.bincount(targets.astype(np.int64), minlength=max(int(num_classes), 1)).astype(np.float32)
    counts = np.clip(counts, a_min=1.0, a_max=None)
    class_weights = 1.0 / counts
    class_weights = class_weights / max(float(class_weights.mean()), 1e-8)
    return class_weights[targets.astype(np.int64)]


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
    metadata: PlannerMetadata,
    config: dict[str, Any],
    *,
    model_type: str,
    device: torch.device,
) -> dict[str, Any]:
    family_class_weights = None
    primitive_class_weights = None
    if model_type != "direct_transformer_action" and bool(config.get("use_class_balanced_loss", False)):
        family_class_weights = compute_class_weights(
            dataset.get_target_array("family"),
            num_classes=metadata.num_families,
            strategy=str(config.get("class_balance_strategy", "effective_num")),
            beta=float(config.get("class_balance_beta", 0.999)),
            power=float(config.get("class_weight_power", 0.5)),
            max_weight=float(config.get("class_weight_max", 5.0)),
        ).to(device)
        primitive_class_weights = compute_class_weights(
            dataset.get_target_array("primitive"),
            num_classes=metadata.num_primitives,
            strategy=str(config.get("class_balance_strategy", "effective_num")),
            beta=float(config.get("class_balance_beta", 0.999)),
            power=float(config.get("class_weight_power", 0.5)),
            max_weight=float(config.get("class_weight_max", 5.0)),
        ).to(device)
    return {
        "label_smoothing": float(config.get("label_smoothing", 0.0)),
        "focal_heads": set(config.get("focal_heads", [])) if bool(config.get("use_focal_loss", False)) else set(),
        "focal_gamma": float(config.get("focal_gamma", 0.0)),
        "family_class_weights": family_class_weights,
        "primitive_class_weights": primitive_class_weights,
        "family_primitive_mask": family_mask_tensor(metadata, device=device),
        "loss_weights": {
            "family": float(config.get("family_loss_weight", 1.0)),
            "primitive": float(config.get("primitive_loss_weight", 1.0)),
            "duration": float(config.get("duration_loss_weight", 0.35)),
            "dynamics": float(config.get("dynamics_loss_weight", 0.25)),
            "params": float(config.get("param_loss_weight", 0.0)),
        },
        "normalize_loss_by_active_weights": bool(config.get("normalize_loss_by_active_weights", True)),
    }


def compute_class_weights(
    targets: np.ndarray,
    *,
    num_classes: int,
    strategy: str,
    beta: float,
    power: float,
    max_weight: float,
) -> torch.Tensor:
    counts = np.bincount(targets.astype(np.int64), minlength=max(num_classes, 1)).astype(np.float32)
    counts = np.clip(counts, a_min=1.0, a_max=None)
    if strategy == "effective_num":
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / np.clip(effective_num, 1e-8, None)
    elif strategy == "inverse_frequency":
        weights = np.power(counts, -power, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported class_balance_strategy={strategy!r}.")
    weights = weights / max(float(weights.mean()), 1e-8)
    if max_weight > 0.0:
        weights = np.clip(weights, a_min=0.0, a_max=max_weight)
        weights = weights / max(float(weights.mean()), 1e-8)
    return torch.from_numpy(weights.astype(np.float32))


def train_one_epoch(model, loader, optimizer, device, model_type: str, *, loss_config: dict[str, Any], gradient_clip_norm: float) -> dict[str, float]:
    model.train()
    aggregates: dict[str, list[float]] = defaultdict(list)
    for batch in tqdm(loader, desc="Train transformer", leave=False):
        batch = move_to_device(batch, str(device))
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch)
        loss, metrics = compute_loss(outputs, batch, model_type, loss_config=loss_config)
        loss.backward()
        if gradient_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
        optimizer.step()
        aggregates["loss"].append(float(loss.item()))
        for key, value in metrics.items():
            if np.isfinite(value):
                aggregates[key].append(float(value))
    return {key: float(np.mean(values)) for key, values in aggregates.items() if values}


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    model_type: str,
    *,
    metadata: PlannerMetadata,
    loss_config: dict[str, Any],
    topk: int,
    eval_temperature: float,
) -> tuple[dict[str, float], list[dict[str, Any]], dict[str, Any]]:
    model.eval()
    aggregates: dict[str, list[float]] = defaultdict(list)
    family_truth: list[int] = []
    family_pred: list[int] = []
    primitive_truth: list[int] = []
    primitive_pred: list[int] = []
    duration_truth: list[int] = []
    duration_pred: list[int] = []
    dynamics_truth: list[int] = []
    dynamics_pred: list[int] = []
    family_topk_hits: list[float] = []
    primitive_topk_hits: list[float] = []
    family_confidence: list[float] = []
    primitive_confidence: list[float] = []
    family_entropy: list[float] = []
    primitive_entropy: list[float] = []
    generated: list[dict[str, Any]] = []

    for batch in tqdm(loader, desc="Eval transformer", leave=False):
        batch = move_to_device(batch, str(device))
        outputs = model(batch)
        loss, metrics = compute_loss(outputs, batch, model_type, loss_config=loss_config)
        loss_value = float(loss.item())
        if np.isfinite(loss_value):
            aggregates["loss"].append(loss_value)
        for key, value in metrics.items():
            if np.isfinite(value):
                aggregates[key].append(float(value))
        if model_type == "direct_transformer_action":
            continue

        scaled_family_logits = scale_logits(torch.nan_to_num(outputs["family_logits"]), eval_temperature)
        predicted_family = scaled_family_logits.argmax(dim=-1)
        masked_primitive_logits = mask_logits_to_family(
            scale_logits(torch.nan_to_num(outputs["primitive_logits"]), eval_temperature),
            predicted_family,
            loss_config["family_primitive_mask"],
        )
        predicted_primitive = masked_primitive_logits.argmax(dim=-1)
        scaled_duration_logits = scale_logits(torch.nan_to_num(outputs["duration_logits"]), eval_temperature)
        scaled_dynamics_logits = scale_logits(torch.nan_to_num(outputs["dynamics_logits"]), eval_temperature)
        predicted_duration = scaled_duration_logits.argmax(dim=-1)
        predicted_dynamics = scaled_dynamics_logits.argmax(dim=-1)

        family_truth.extend(batch["target_family"].tolist())
        family_pred.extend(predicted_family.tolist())
        primitive_truth.extend(batch["target_primitive"].tolist())
        primitive_pred.extend(predicted_primitive.tolist())
        duration_truth.extend(batch["target_duration"].tolist())
        duration_pred.extend(predicted_duration.tolist())
        dynamics_truth.extend(batch["target_dynamics"].tolist())
        dynamics_pred.extend(predicted_dynamics.tolist())

        family_topk_hits.append(topk_accuracy(scaled_family_logits, batch["target_family"], topk=topk))
        primitive_topk_hits.append(topk_accuracy(masked_primitive_logits, batch["target_primitive"], topk=topk))
        family_confidence.append(max_confidence(scaled_family_logits))
        primitive_confidence.append(max_confidence(masked_primitive_logits))
        family_entropy.append(mean_entropy(scaled_family_logits))
        primitive_entropy.append(mean_entropy(masked_primitive_logits))

        for truth_family, pred_family, truth_primitive, pred_primitive, truth_duration, pred_duration, truth_dynamics, pred_dynamics in zip(
            batch["target_family"].tolist(),
            predicted_family.tolist(),
            batch["target_primitive"].tolist(),
            predicted_primitive.tolist(),
            batch["target_duration"].tolist(),
            predicted_duration.tolist(),
            batch["target_dynamics"].tolist(),
            predicted_dynamics.tolist(),
        ):
            generated.append(
                {
                    "target_family": int(truth_family),
                    "predicted_family": int(pred_family),
                    "target_family_name": metadata.primitive_family_names[int(truth_family)],
                    "predicted_family_name": metadata.primitive_family_names[int(pred_family)],
                    "target_primitive": int(truth_primitive),
                    "predicted_primitive": int(pred_primitive),
                    "target_primitive_id": metadata.primitive_ids[int(truth_primitive)],
                    "predicted_primitive_id": metadata.primitive_ids[int(pred_primitive)],
                    "target_duration": int(truth_duration),
                    "predicted_duration": int(pred_duration),
                    "target_dynamics": int(truth_dynamics),
                    "predicted_dynamics": int(pred_dynamics),
                }
            )

    val_loss = float(np.mean(aggregates["loss"])) if aggregates["loss"] else float("inf")
    metrics = {key: float(np.mean(values)) for key, values in aggregates.items() if values}
    if model_type == "direct_transformer_action":
        return metrics, generated, {}

    family_confusion = build_confusion_matrix(family_truth, family_pred, metadata.num_families)
    family_macro_f1 = macro_f1_from_confusion(family_confusion)
    family_balanced_acc = balanced_accuracy_from_confusion(family_confusion)
    metrics.update(
        {
            "loss": val_loss,
            "family_topk_accuracy": float(np.mean(family_topk_hits)) if family_topk_hits else 0.0,
            "primitive_topk_accuracy": float(np.mean(primitive_topk_hits)) if primitive_topk_hits else 0.0,
            "family_macro_f1": family_macro_f1,
            "family_balanced_accuracy": family_balanced_acc,
            "family_avg_confidence": float(np.mean(family_confidence)) if family_confidence else 0.0,
            "primitive_avg_confidence": float(np.mean(primitive_confidence)) if primitive_confidence else 0.0,
            "family_entropy": float(np.mean(family_entropy)) if family_entropy else 0.0,
            "primitive_entropy": float(np.mean(primitive_entropy)) if primitive_entropy else 0.0,
            "primitive_perplexity": safe_perplexity(metrics.get("primitive_loss", float("inf"))),
            "family_perplexity": safe_perplexity(metrics.get("family_loss", float("inf"))),
        }
    )
    return metrics, generated, {"family_confusion": confusion_dataframe(family_confusion, metadata)}


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
        return loss, {"action_mse": float(loss.item())}

    family_logits = torch.nan_to_num(outputs["family_logits"])
    primitive_logits = torch.nan_to_num(outputs["primitive_logits"])
    duration_logits = torch.nan_to_num(outputs["duration_logits"])
    dynamics_logits = torch.nan_to_num(outputs["dynamics_logits"])
    family_mask = loss_config["family_primitive_mask"]
    label_smoothing = float(loss_config.get("label_smoothing", 0.0))
    focal_heads = set(loss_config.get("focal_heads", set()))
    focal_gamma = float(loss_config.get("focal_gamma", 0.0))
    loss_weights = dict(loss_config["loss_weights"])

    family_loss = classification_loss(
        family_logits,
        batch["target_family"],
        class_weights=loss_config.get("family_class_weights"),
        label_smoothing=label_smoothing,
        focal_gamma=focal_gamma if "family" in focal_heads else 0.0,
    )
    masked_target_primitive_logits = mask_logits_to_family(primitive_logits, batch["target_family"], family_mask)
    primitive_loss = classification_loss(
        masked_target_primitive_logits,
        batch["target_primitive"],
        class_weights=loss_config.get("primitive_class_weights"),
        label_smoothing=label_smoothing,
        focal_gamma=focal_gamma if "primitive" in focal_heads else 0.0,
        valid_mask=family_mask[batch["target_family"]],
    )
    duration_loss = classification_loss(
        duration_logits,
        batch["target_duration"],
        class_weights=None,
        label_smoothing=label_smoothing,
        focal_gamma=focal_gamma if "duration" in focal_heads else 0.0,
    )
    dynamics_loss = classification_loss(
        dynamics_logits,
        batch["target_dynamics"],
        class_weights=None,
        label_smoothing=label_smoothing,
        focal_gamma=focal_gamma if "dynamics" in focal_heads else 0.0,
    )
    if outputs.get("continuous_param_pred") is not None and batch["target_params"].numel() > 0:
        param_loss = F.smooth_l1_loss(outputs["continuous_param_pred"], batch["target_params"])
    else:
        param_loss = family_loss.new_tensor(0.0)

    weighted_losses = {
        "family": loss_weights["family"] * family_loss,
        "primitive": loss_weights["primitive"] * primitive_loss,
        "duration": loss_weights["duration"] * duration_loss,
        "dynamics": loss_weights["dynamics"] * dynamics_loss,
        "params": loss_weights["params"] * param_loss,
    }
    total_loss = sum(weighted_losses.values())
    if bool(loss_config.get("normalize_loss_by_active_weights", True)):
        active_weight_sum = sum(weight for weight in loss_weights.values() if weight > 0.0)
        if active_weight_sum > 0.0:
            total_loss = total_loss / active_weight_sum

    predicted_family = family_logits.argmax(dim=-1)
    predicted_primitive = mask_logits_to_family(primitive_logits, predicted_family, family_mask).argmax(dim=-1)
    predicted_duration = duration_logits.argmax(dim=-1)
    predicted_dynamics = dynamics_logits.argmax(dim=-1)

    metrics = {
        "family_loss": float(family_loss.item()),
        "primitive_loss": float(primitive_loss.item()),
        "duration_loss": float(duration_loss.item()),
        "dynamics_loss": float(dynamics_loss.item()),
        "param_loss": float(param_loss.item()),
        "family_accuracy": float((predicted_family == batch["target_family"]).float().mean().item()),
        "primitive_accuracy": float((predicted_primitive == batch["target_primitive"]).float().mean().item()),
        "duration_accuracy": float((predicted_duration == batch["target_duration"]).float().mean().item()),
        "dynamics_accuracy": float((predicted_dynamics == batch["target_dynamics"]).float().mean().item()),
    }
    if outputs.get("continuous_param_pred") is not None and batch["target_params"].numel() > 0:
        metrics["param_mae"] = float(torch.mean(torch.abs(outputs["continuous_param_pred"] - batch["target_params"])).item())
    return total_loss, metrics


def classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    class_weights: torch.Tensor | None,
    label_smoothing: float,
    focal_gamma: float,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    num_classes = logits.shape[-1]
    log_probs = F.log_softmax(logits, dim=-1)
    if label_smoothing > 0.0 and num_classes > 1:
        if valid_mask is not None:
            target_dist = torch.zeros_like(log_probs)
            target_dist.scatter_(1, targets.unsqueeze(1), 1.0 - label_smoothing)
            smoothing_support = valid_mask.clone()
            smoothing_support.scatter_(1, targets.unsqueeze(1), False)
            support_counts = smoothing_support.sum(dim=-1, keepdim=True)
            smoothed_mass = torch.where(
                support_counts > 0,
                label_smoothing / support_counts.clamp(min=1).to(log_probs.dtype),
                torch.zeros_like(support_counts, dtype=log_probs.dtype),
            )
            target_dist = target_dist + smoothing_support.to(log_probs.dtype) * smoothed_mass
        else:
            smooth = label_smoothing / max(num_classes - 1, 1)
            target_dist = torch.full_like(log_probs, smooth)
            target_dist.scatter_(1, targets.unsqueeze(1), 1.0 - label_smoothing)
        sample_loss = -(target_dist * log_probs).sum(dim=-1)
    else:
        sample_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
    if focal_gamma > 0.0:
        pt = torch.exp(log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1))
        sample_loss = sample_loss * torch.pow(1.0 - pt, focal_gamma)
    if class_weights is not None:
        sample_loss = sample_loss * class_weights[targets]
    return sample_loss.mean()


def mask_logits_to_family(logits: torch.Tensor, family_index: torch.Tensor, family_mask: torch.Tensor) -> torch.Tensor:
    mask = family_mask[family_index]
    fill_value = -1.0e4
    return logits.masked_fill(~mask, fill_value)


def scale_logits(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 1.0:
        return logits
    return logits / max(float(temperature), 1e-6)


def topk_accuracy(logits: torch.Tensor, target: torch.Tensor, topk: int) -> float:
    _, indices = logits.topk(min(topk, logits.shape[-1]), dim=-1)
    hits = (indices == target.unsqueeze(-1)).any(dim=-1).float().mean().item()
    return float(hits)


def max_confidence(logits: torch.Tensor) -> float:
    return float(torch.softmax(logits, dim=-1).amax(dim=-1).mean().item())


def mean_entropy(logits: torch.Tensor) -> float:
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs.clamp(min=1e-8))).sum(dim=-1)
    return float(entropy.mean().item())


def build_confusion_matrix(targets: list[int], predictions: list[int], num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, pred in zip(targets, predictions):
        matrix[int(truth), int(pred)] += 1
    return matrix


def balanced_accuracy_from_confusion(matrix: np.ndarray) -> float:
    support = matrix.sum(axis=1).astype(np.float32)
    recalls = np.divide(
        np.diag(matrix).astype(np.float32),
        np.clip(support, 1.0, None),
        out=np.zeros_like(support, dtype=np.float32),
        where=support > 0,
    )
    valid = support > 0
    return float(recalls[valid].mean()) if np.any(valid) else 0.0


def macro_f1_from_confusion(matrix: np.ndarray) -> float:
    true_positive = np.diag(matrix).astype(np.float32)
    precision = np.divide(
        true_positive,
        np.clip(matrix.sum(axis=0).astype(np.float32), 1.0, None),
        out=np.zeros_like(true_positive),
        where=matrix.sum(axis=0) > 0,
    )
    recall = np.divide(
        true_positive,
        np.clip(matrix.sum(axis=1).astype(np.float32), 1.0, None),
        out=np.zeros_like(true_positive),
        where=matrix.sum(axis=1) > 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        np.clip(precision + recall, 1e-8, None),
        out=np.zeros_like(true_positive),
        where=(precision + recall) > 0,
    )
    valid = matrix.sum(axis=1) > 0
    return float(f1[valid].mean()) if np.any(valid) else 0.0


def confusion_dataframe(matrix: np.ndarray, metadata: PlannerMetadata) -> pd.DataFrame:
    frame = pd.DataFrame(
        matrix,
        columns=[f"pred_{name}" for name in metadata.primitive_family_names],
    )
    frame.insert(0, "target_family", metadata.primitive_family_names)
    frame["support"] = matrix.sum(axis=1)
    return frame


def safe_perplexity(loss_value: float) -> float:
    if not np.isfinite(loss_value):
        return float("inf")
    return float(np.exp(min(loss_value, 20.0)))


def pd_from_records(records: list[dict[str, Any]]):
    return pd.DataFrame(records)
