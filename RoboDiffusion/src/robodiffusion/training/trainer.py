from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from robodiffusion.data.windows import CachedWindowDataset, collate_window_batch
from robodiffusion.model.diffusion import DiffusionConfig, DiffusionTransformer, GaussianDiffusionScheduler
from robodiffusion.model.policy import PolicyMetadata
from robodiffusion.utils.checkpointing import find_latest_checkpoint, load_checkpoint, save_checkpoint
from robodiffusion.utils.experiment import make_run_paths
from robodiffusion.utils.io import save_npz, write_json
from robodiffusion.utils.metrics import MetricsWriter
from robodiffusion.utils.random import set_seed
from robodiffusion.utils.torch_utils import count_parameters, move_to_device
from robodiffusion.utils.wandb import WandbRun

LOGGER = logging.getLogger(__name__)


def run_training(config: dict[str, Any], logger: logging.Logger | None = None) -> dict[str, Path]:
    logger = logger or LOGGER
    dataset_root = Path(config["dataset_root"]).resolve()
    output_root = Path(config["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_paths = make_run_paths(output_root, "training", config["experiment_name"], int(config["seed"]), resume=bool(config.get("resume", False)))
    write_json(config, run_paths.artifacts / "config.json")
    wandb_run = WandbRun(
        config.get("wandb"),
        run_name=run_paths.root.name,
        config_payload=config,
        logger=logger,
        job_type="training",
        tags=["robodiffusion", "training"],
    )

    try:
        set_seed(int(config["seed"]), deterministic=bool(config.get("deterministic_eval", False)))
        train_dataset = CachedWindowDataset(dataset_root, split="train")
        val_dataset = CachedWindowDataset(dataset_root, split="val") if _split_count(dataset_root, "val") > 0 else CachedWindowDataset(dataset_root, split="train")

        batch_size = int(config["batch_size"])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=int(config.get("num_workers", 0)), collate_fn=collate_window_batch)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=int(config.get("num_workers", 0)), collate_fn=collate_window_batch)

        metadata = train_dataset.metadata
        diffusion_config = DiffusionConfig(
            score_dim=int(metadata["score_dim"]),
            state_dim=int(metadata["state_dim"]),
            action_dim=int(metadata["action_dim"]),
            obs_horizon=int(metadata["obs_horizon"]),
            pred_horizon=int(metadata["pred_horizon"]),
            model_dim=int(config["model_dim"]),
            num_layers=int(config["num_layers"]),
            num_heads=int(config["num_heads"]),
            dropout=float(config.get("dropout", 0.1)),
            diffusion_steps=int(config["diffusion_steps"]),
            beta_start=float(config["beta_start"]),
            beta_end=float(config["beta_end"]),
        )
        policy_metadata = PolicyMetadata(
            score_dim=diffusion_config.score_dim,
            state_dim=diffusion_config.state_dim,
            action_dim=diffusion_config.action_dim,
            obs_horizon=diffusion_config.obs_horizon,
            pred_horizon=diffusion_config.pred_horizon,
            action_execute_horizon=int(config["action_execute_horizon"]),
            observation_spec=dict(metadata.get("observation_spec", {})),
        )
        device = torch.device(config["device"])
        model = DiffusionTransformer(
            score_dim=diffusion_config.score_dim,
            state_dim=diffusion_config.state_dim,
            action_dim=diffusion_config.action_dim,
            obs_horizon=diffusion_config.obs_horizon,
            pred_horizon=diffusion_config.pred_horizon,
            model_dim=diffusion_config.model_dim,
            num_layers=diffusion_config.num_layers,
            num_heads=diffusion_config.num_heads,
            dropout=diffusion_config.dropout,
        ).to(device)
        scheduler = GaussianDiffusionScheduler(
            steps=diffusion_config.diffusion_steps,
            beta_start=diffusion_config.beta_start,
            beta_end=diffusion_config.beta_end,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(config["epochs"]), 1))
        metrics_writer = MetricsWriter(run_paths.metrics)

        start_epoch = 0
        best_loss = float("inf")
        if bool(config.get("resume", False)):
            checkpoint_path = find_latest_checkpoint(run_paths.checkpoints)
            if checkpoint_path is not None:
                payload = load_checkpoint(checkpoint_path, map_location=device)
                model.load_state_dict(payload["model"])
                optimizer.load_state_dict(payload["optimizer"])
                lr_scheduler.load_state_dict(payload["scheduler"])
                start_epoch = int(payload["epoch"]) + 1
                best_loss = float(payload.get("best_loss", best_loss))
                wandb_run.summary({"resume/checkpoint": str(checkpoint_path), "resume/start_epoch": start_epoch})

        num_parameters = count_parameters(model)
        logger.info("Training samples: %d", len(train_dataset))
        logger.info("Validation samples: %d", len(val_dataset))
        logger.info("Trainable parameters: %d", num_parameters)
        wandb_run.summary(
            {
                "run_root": str(run_paths.root),
                "dataset_root": str(dataset_root),
                "dataset/train_samples": len(train_dataset),
                "dataset/val_samples": len(val_dataset),
                "dataset/action_dim": metadata["action_dim"],
                "dataset/score_dim": metadata["score_dim"],
                "dataset/state_dim": metadata["state_dim"],
                "model/num_parameters": num_parameters,
            }
        )

        for epoch in range(start_epoch, int(config["epochs"])):
            train_metrics = diffusion_epoch(model, scheduler, train_loader, device, train=True, optimizer=optimizer, config=config)
            val_metrics, sample_bundle = diffusion_epoch(model, scheduler, val_loader, device, train=False, optimizer=None, config=config)
            sampler_ms = benchmark_sampler(model, scheduler, val_loader, device)
            row = {
                "epoch": epoch,
                "lr": float(optimizer.param_groups[0]["lr"]),
                **{f"train/{key}": value for key, value in train_metrics.items()},
                **{f"val/{key}": value for key, value in val_metrics.items()},
                "val/sampler_ms": sampler_ms,
            }
            metrics_writer.log(row)
            wandb_run.log(row, step=epoch)
            logger.info("Epoch %d train=%s val=%s sampler_ms=%.2f", epoch, train_metrics, val_metrics, sampler_ms)
            lr_scheduler.step()
            monitor = float(val_metrics["loss"])
            payload = {
                "epoch": epoch,
                "best_loss": min(best_loss, monitor),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": lr_scheduler.state_dict(),
                "diffusion_config": diffusion_config.to_payload(),
                "policy_metadata": policy_metadata.to_payload(),
                "dataset_metadata": metadata,
                "training_config": config,
            }
            latest_checkpoint = save_checkpoint(run_paths.checkpoints / "latest.pt", payload)
            wandb_run.summary({"latest/epoch": epoch, "latest/checkpoint": str(latest_checkpoint)})
            if np.isfinite(monitor) and monitor < best_loss:
                best_loss = monitor
                best_checkpoint = save_checkpoint(run_paths.checkpoints / "best.pt", payload)
                wandb_run.summary({"best/epoch": epoch, "best/val_loss": best_loss, "best/checkpoint": str(best_checkpoint)})
                if sample_bundle is not None:
                    save_npz(run_paths.artifacts / "val_samples.npz", **sample_bundle)
            if (epoch + 1) % int(config["checkpoint_interval"]) == 0:
                save_checkpoint(run_paths.checkpoints / f"epoch_{epoch:04d}.pt", payload)

        best_checkpoint_path = run_paths.checkpoints / "best.pt"
        wandb_run.log_artifact_bundle(
            artifact_name=f"{run_paths.root.name}-checkpoints",
            artifact_type="model",
            entries={"checkpoints": run_paths.checkpoints},
            aliases=["latest", "best"],
            metadata={"stage": "training", "run_root": str(run_paths.root)},
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
            metadata={"stage": "training", "run_root": str(run_paths.root)},
        )
        wandb_run.summary({"status": "completed", "best/checkpoint": str(best_checkpoint_path)})
        return {"run_root": run_paths.root, "best_checkpoint": best_checkpoint_path}
    finally:
        wandb_run.finish()


def diffusion_epoch(
    model: DiffusionTransformer,
    scheduler: GaussianDiffusionScheduler,
    loader: DataLoader,
    device: torch.device,
    *,
    train: bool,
    optimizer: torch.optim.Optimizer | None,
    config: dict[str, Any],
) -> tuple[dict[str, float], dict[str, np.ndarray] | None] | dict[str, float]:
    model.train(mode=train)
    noise_loss_sum = 0.0
    action_mse_sum = 0.0
    smoothness_sum = 0.0
    loss_sum = 0.0
    batches = 0
    captured_samples: dict[str, np.ndarray] | None = None
    iterator = tqdm(loader, leave=False, disable=not bool(config.get("progress_bar", False)))
    for batch in iterator:
        batch = move_to_device(batch, device)
        target = batch["action_target"]
        timesteps = scheduler.sample_timesteps(target.shape[0], device)
        noise = torch.randn_like(target)
        noisy = scheduler.q_sample(target, timesteps, noise=noise)
        predicted_noise = model(noisy, batch["score_window"], batch["state_window"], timesteps)
        predicted_action = scheduler.predict_x0(noisy, timesteps, predicted_noise)

        noise_loss = F.mse_loss(predicted_noise, noise)
        action_mse = F.mse_loss(predicted_action, target)
        smoothness = F.l1_loss(predicted_action[:, 1:] - predicted_action[:, :-1], target[:, 1:] - target[:, :-1]) if target.shape[1] > 1 else torch.tensor(0.0, device=device)
        loss = noise_loss + float(config.get("action_loss_weight", 1.0)) * action_mse + float(config.get("smoothness_weight", 0.05)) * smoothness

        if train and optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("grad_clip_norm", 1.0)))
            optimizer.step()

        batches += 1
        noise_loss_sum += float(noise_loss.item())
        action_mse_sum += float(action_mse.item())
        smoothness_sum += float(smoothness.item())
        loss_sum += float(loss.item())

        if not train and captured_samples is None:
            captured_samples = {
                "predicted": predicted_action[: min(4, predicted_action.shape[0])].detach().cpu().numpy(),
                "target": target[: min(4, target.shape[0])].detach().cpu().numpy(),
            }
    metrics = {
        "loss": loss_sum / max(batches, 1),
        "noise_loss": noise_loss_sum / max(batches, 1),
        "action_mse": action_mse_sum / max(batches, 1),
        "smoothness": smoothness_sum / max(batches, 1),
    }
    if train:
        return metrics
    return metrics, captured_samples


@torch.no_grad()
def benchmark_sampler(
    model: DiffusionTransformer,
    scheduler: GaussianDiffusionScheduler,
    loader: DataLoader,
    device: torch.device,
) -> float:
    try:
        batch = next(iter(loader))
    except StopIteration:
        return 0.0
    batch = move_to_device(batch, device)
    start = time.perf_counter()
    _ = scheduler.sample(
        model,
        score_window=batch["score_window"][:1],
        state_window=batch["state_window"][:1],
        action_shape=(1, batch["action_target"].shape[1], batch["action_target"].shape[2]),
        device=device,
    )
    elapsed = time.perf_counter() - start
    return float(elapsed * 1000.0)


def _split_count(dataset_root: Path, split: str) -> int:
    payload = CachedWindowDataset(dataset_root, split=split).split_manifest
    return int(payload.get("count", 0))
