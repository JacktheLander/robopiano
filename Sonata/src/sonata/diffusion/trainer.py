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

from sonata.diffusion.dataset import DiffusionChunkDataset, diffusion_collate_fn, load_diffusion_inputs
from sonata.diffusion.diffusion import GaussianDiffusion1D
from sonata.diffusion.model import ConditionalTemporalDenoiser
from sonata.transformer.model import PrimitivePlannerTransformer
from sonata.utils.checkpointing import find_latest_checkpoint, load_checkpoint, save_checkpoint
from sonata.utils.experiment import make_run_paths
from sonata.utils.io import save_npz, write_json
from sonata.utils.metrics import MetricsWriter
from sonata.utils.random import set_seed
from sonata.utils.torch_utils import count_parameters, move_to_device
from sonata.utils.wandb import WandbRun


def run_diffusion_training(config: dict[str, Any], logger: logging.Logger, joint_refine: bool = False) -> dict[str, Path]:
    primitive_root = Path(config["primitive_root"]).resolve()
    output_root = Path(config["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    token_df, metadata, prior_lookup = load_diffusion_inputs(primitive_root, int(config["action_horizon"]), int(config["state_context_steps"]))
    run_paths = make_run_paths(output_root, "joint_refine" if joint_refine else "diffusion", config["experiment_name"], int(config["seed"]), resume=bool(config.get("resume", False)))
    write_json(config, run_paths.artifacts / "config.json")
    stage_name = "joint_refine" if joint_refine else "diffusion"
    wandb_run = WandbRun(
        config.get("wandb"),
        run_name=run_paths.root.name,
        config_payload=config,
        logger=logger,
        job_type=stage_name,
        tags=["sonata", stage_name],
    )
    try:
        set_seed(int(config["seed"]), deterministic=bool(config.get("deterministic_eval", False)))
        train_dataset = DiffusionChunkDataset(
            token_df=token_df,
            primitive_root=primitive_root,
            split="train",
            context_length=int(config["context_length"]),
            action_horizon=int(config["action_horizon"]),
            state_context_steps=int(config["state_context_steps"]),
        )
        val_dataset = DiffusionChunkDataset(
            token_df=token_df,
            primitive_root=primitive_root,
            split="val",
            context_length=int(config["context_length"]),
            action_horizon=int(config["action_horizon"]),
            state_context_steps=int(config["state_context_steps"]),
        )
        collate = partial(diffusion_collate_fn, metadata=metadata, prior_lookup=prior_lookup)
        train_loader = DataLoader(train_dataset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=int(config.get("num_workers", 0)), collate_fn=collate)
        val_loader = DataLoader(val_dataset, batch_size=int(config["batch_size"]), shuffle=False, num_workers=int(config.get("num_workers", 0)), collate_fn=collate)

        device = torch.device(config["device"])
        planner, planner_dim = maybe_load_planner(config, metadata, device)
        if planner is not None and not joint_refine:
            planner.eval()
            for parameter in planner.parameters():
                parameter.requires_grad = False

        global_condition_dim = metadata.score_dim + metadata.state_dim + 3
        if planner is not None:
            global_condition_dim += int(planner_dim)
        else:
            global_condition_dim += int(config["primitive_embedding_dim"])
        model = ConditionalTemporalDenoiser(
            action_dim=metadata.action_dim,
            prior_dim=metadata.action_dim,
            global_cond_dim=global_condition_dim,
            model_dim=int(config["model_dim"]),
            num_blocks=int(config["num_blocks"]),
        ).to(device)
        primitive_embed = torch.nn.Embedding(metadata.num_primitives + 1, int(config["primitive_embedding_dim"])).to(device)

        parameters = list(model.parameters()) + list(primitive_embed.parameters())
        if planner is not None and joint_refine:
            parameters += list(planner.parameters())

        optimizer = torch.optim.AdamW(parameters, lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(config["epochs"]), 1))
        diffusion = GaussianDiffusion1D(int(config["diffusion_steps"]), float(config["beta_start"]), float(config["beta_end"]), device)
        metrics_writer = MetricsWriter(run_paths.metrics)

        num_parameters = count_parameters(model)
        logger.info("Diffusion parameters: %d", num_parameters)
        wandb_run.summary(
            {
                "stage": stage_name,
                "run_root": str(run_paths.root),
                "primitive_root": str(primitive_root),
                "model/num_parameters": num_parameters,
                "planner_checkpoint": config.get("planner_checkpoint"),
            }
        )
        best_loss = float("inf")
        start_epoch = 0
        if bool(config.get("resume", False)):
            checkpoint = find_latest_checkpoint(run_paths.checkpoints)
            if checkpoint is not None:
                payload = load_checkpoint(checkpoint, map_location=device)
                model.load_state_dict(payload["model"])
                primitive_embed.load_state_dict(payload["primitive_embed"])
                optimizer.load_state_dict(payload["optimizer"])
                scheduler.load_state_dict(payload["scheduler"])
                if planner is not None and "planner" in payload:
                    planner.load_state_dict(payload["planner"])
                start_epoch = int(payload["epoch"]) + 1
                wandb_run.summary({"resume/checkpoint": str(checkpoint), "resume/start_epoch": start_epoch})

        for epoch in range(start_epoch, int(config["epochs"])):
            train_metrics = diffusion_epoch(
                model=model,
                primitive_embed=primitive_embed,
                planner=planner,
                loader=train_loader,
                optimizer=optimizer,
                diffusion=diffusion,
                device=device,
                config=config,
                train=True,
            )
            val_metrics, samples = diffusion_epoch(
                model=model,
                primitive_embed=primitive_embed,
                planner=planner,
                loader=val_loader,
                optimizer=None,
                diffusion=diffusion,
                device=device,
                config=config,
                train=False,
            )
            row = {"epoch": epoch, **{f"train/{key}": value for key, value in train_metrics.items()}, **{f"val/{key}": value for key, value in val_metrics.items()}}
            metrics_writer.log(row)
            wandb_run.log(row, step=epoch)
            logger.info("Epoch %d train=%s val=%s", epoch, train_metrics, val_metrics)
            scheduler.step()
            if val_metrics["loss"] < best_loss:
                best_loss = val_metrics["loss"]
                best_checkpoint = save_checkpoint(
                    run_paths.checkpoints / "best.pt",
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "primitive_embed": primitive_embed.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "planner": planner.state_dict() if planner is not None else None,
                        "config": config,
                    },
                )
                wandb_run.summary({"best/epoch": epoch, "best/val_loss": best_loss, "best/checkpoint": str(best_checkpoint)})
                if samples is not None:
                    save_npz(run_paths.artifacts / "val_samples.npz", predicted=samples["predicted"], target=samples["target"], prior=samples["prior"])
            if (epoch + 1) % int(config["checkpoint_interval"]) == 0:
                save_checkpoint(
                    run_paths.checkpoints / f"epoch_{epoch:04d}.pt",
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "primitive_embed": primitive_embed.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "planner": planner.state_dict() if planner is not None else None,
                        "config": config,
                    },
                )
        best_checkpoint_path = run_paths.checkpoints / "best.pt"
        wandb_run.log_artifact_bundle(
            artifact_name=f"{run_paths.root.name}-checkpoints",
            artifact_type="model",
            entries={"checkpoints": run_paths.checkpoints},
            aliases=["latest", "best"],
            metadata={"stage": stage_name, "run_root": str(run_paths.root)},
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
            metadata={"stage": stage_name, "run_root": str(run_paths.root)},
        )
        wandb_run.summary({"status": "completed", "best/checkpoint": str(best_checkpoint_path)})
        return {"run_root": run_paths.root, "best_checkpoint": best_checkpoint_path}
    finally:
        wandb_run.finish()


def maybe_load_planner(config: dict[str, Any], metadata, device: torch.device):
    variant = str(config["variant"])
    if variant in {"diffusion_only", "gmr_only"}:
        return None, 0
    checkpoint_path = Path(config["planner_checkpoint"]).resolve() if config.get("planner_checkpoint") else None
    if checkpoint_path is None or not checkpoint_path.exists():
        return None, 0
    payload = load_checkpoint(checkpoint_path, map_location=device)
    planner_config = payload.get("config", {})
    d_model = int(planner_config.get("d_model", config["planner_embedding_dim"]))
    planner = PrimitivePlannerTransformer(
        num_primitives=metadata.num_primitives,
        num_duration_buckets=metadata.num_duration_buckets,
        num_dynamics_buckets=metadata.num_dynamics_buckets,
        score_dim=metadata.score_dim,
        d_model=d_model,
        nhead=int(planner_config.get("nhead", config["planner_nhead"])),
        num_layers=int(planner_config.get("num_layers", config["planner_layers"])),
        dim_feedforward=int(planner_config.get("dim_feedforward", config["planner_ffn"])),
        dropout=float(planner_config.get("dropout", config["planner_dropout"])),
        max_length=int(planner_config.get("context_length", config["context_length"])),
    ).to(device)
    planner.load_state_dict(payload["model"], strict=False)
    return planner, d_model


def diffusion_epoch(model, primitive_embed, planner, loader, optimizer, diffusion, device, config: dict[str, Any], train: bool):
    if train:
        model.train()
        primitive_embed.train()
        if planner is not None and any(parameter.requires_grad for parameter in planner.parameters()):
            planner.train()
    else:
        model.eval()
        primitive_embed.eval()
        if planner is not None:
            planner.eval()
    losses: list[float] = []
    imitation_losses: list[float] = []
    smoothness_losses: list[float] = []
    sample_bundle = None
    iterator = tqdm(loader, desc="Train diffusion" if train else "Eval diffusion", leave=False)
    for batch in iterator:
        batch = move_to_device(batch, str(device))
        prior = batch["gmr_prior"]
        if str(config["variant"]) == "gmr_only":
            predicted = prior
            target = batch["action_target"]
            imitation = F.mse_loss(predicted, target)
            smooth = ((predicted[:, 1:] - predicted[:, :-1]) ** 2).mean()
            total = imitation + float(config["smoothness_weight"]) * smooth
            losses.append(float(total.item()))
            imitation_losses.append(float(imitation.item()))
            smoothness_losses.append(float(smooth.item()))
            if sample_bundle is None:
                sample_bundle = {"predicted": predicted[:8].detach().cpu().numpy(), "target": target[:8].detach().cpu().numpy(), "prior": prior[:8].detach().cpu().numpy()}
            continue
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        target = batch["action_target"]
        timestep = diffusion.sample_timesteps(target.shape[0])
        noise = torch.randn_like(target)
        noisy = diffusion.q_sample(target, timestep, noise)
        condition = build_condition_vector(batch, primitive_embed, planner, config)
        if str(config["variant"]) in {"diffusion_only", "planner_no_prior"}:
            prior = torch.zeros_like(prior)
        predicted_noise = model(noisy, prior, timestep, condition)
        diffusion_loss = F.mse_loss(predicted_noise, noise)
        predicted_x0 = diffusion.predict_x0(noisy, timestep, predicted_noise)
        imitation = F.l1_loss(predicted_x0, target)
        smooth = ((predicted_x0[:, 1:] - predicted_x0[:, :-1]) ** 2).mean()
        total = diffusion_loss + float(config["imitation_weight"]) * imitation + float(config["smoothness_weight"]) * smooth
        if train and optimizer is not None:
            total.backward()
            optimizer.step()
        losses.append(float(total.item()))
        imitation_losses.append(float(imitation.item()))
        smoothness_losses.append(float(smooth.item()))
        if sample_bundle is None:
            sample_bundle = {"predicted": predicted_x0[:8].detach().cpu().numpy(), "target": target[:8].detach().cpu().numpy(), "prior": prior[:8].detach().cpu().numpy()}
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "imitation_l1": float(np.mean(imitation_losses)) if imitation_losses else 0.0,
        "smoothness": float(np.mean(smoothness_losses)) if smoothness_losses else 0.0,
    }, sample_bundle


def build_condition_vector(batch, primitive_embed, planner, config: dict[str, Any]) -> torch.Tensor:
    score = batch["score_context"]
    state = batch["state_context"]
    scalar = torch.stack(
        [
            batch["duration_bucket"].float(),
            batch["dynamics_bucket"].float(),
            batch["primitive_index"].float(),
        ],
        dim=-1,
    )
    if planner is not None and str(config["variant"]) != "diffusion_only":
        with torch.set_grad_enabled(any(parameter.requires_grad for parameter in planner.parameters())):
            planner_outputs = planner(batch)
            plan = planner_outputs["plan_embedding"]
        return torch.cat([score, state, scalar, plan], dim=-1)
    primitive = primitive_embed(batch["primitive_index"])
    return torch.cat([score, state, scalar, primitive], dim=-1)
