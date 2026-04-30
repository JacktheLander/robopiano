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

from sonata.diffusion.dataset import DiffusionChunkDataset, DiffusionMetadata, diffusion_collate_fn, load_diffusion_inputs, metadata_to_planner
from sonata.diffusion.diffusion import GaussianDiffusion1D
from sonata.diffusion.model import ConditionalTemporalDenoiser
from sonata.diffusion.note_surrogate import LinearActionNoteSurrogate, fit_linear_note_surrogate
from sonata.transformer.model import (
    FACTORED_PLANNER_ARCHITECTURE,
    build_planner_from_config,
    planner_output_dim_from_config,
)
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
    token_df, metadata, prior_lookup = load_diffusion_inputs(
        primitive_root,
        int(config["action_horizon"]),
        int(config["state_context_steps"]),
        family_mapping_mode=str(config.get("family_mapping_mode", "heuristic_stats")),
        continuous_param_names=config.get("continuous_param_names"),
    )
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
            metadata=metadata,
            primitive_root=primitive_root,
            split="train",
            context_length=int(config["context_length"]),
            action_horizon=int(config["action_horizon"]),
            state_context_steps=int(config["state_context_steps"]),
        )
        val_dataset = DiffusionChunkDataset(
            token_df=token_df,
            metadata=metadata,
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
        planner, planner_dim, planner_config, planner_metadata = maybe_load_planner(config, metadata, device)
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
        note_surrogate, surrogate_fit = fit_linear_note_surrogate(
            train_dataset,
            action_dim=metadata.action_dim,
            ridge_lambda=float(config.get("surrogate_ridge_lambda", 1e-2)),
        )
        note_surrogate = note_surrogate.to(device)
        metrics_writer = MetricsWriter(run_paths.metrics)

        num_parameters = count_parameters(model)
        logger.info("Diffusion parameters: %d", num_parameters)
        wandb_run.summary(
            {
                "stage": stage_name,
                "run_root": str(run_paths.root),
                "primitive_root": str(primitive_root),
                "model/num_parameters": num_parameters,
                "control/predict_residual": bool(config.get("predict_residual", False)),
                "planner_checkpoint": config.get("planner_checkpoint"),
                "planner_architecture": planner_config.get("planner_architecture") if planner_config else None,
                **{f"surrogate/{key}": value for key, value in surrogate_fit.items()},
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
                if planner is not None and "planner" in payload and payload["planner"] is not None:
                    planner.load_state_dict(payload["planner"])
                if payload.get("note_surrogate") is not None:
                    note_surrogate.load_state_dict(payload["note_surrogate"])
                start_epoch = int(payload["epoch"]) + 1
                wandb_run.summary({"resume/checkpoint": str(checkpoint), "resume/start_epoch": start_epoch})

        for epoch in range(start_epoch, int(config["epochs"])):
            train_metrics, _ = diffusion_epoch(
                model=model,
                primitive_embed=primitive_embed,
                planner=planner,
                loader=train_loader,
                optimizer=optimizer,
                diffusion=diffusion,
                note_surrogate=note_surrogate,
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
                note_surrogate=note_surrogate,
                device=device,
                config=config,
                train=False,
            )
            row = {"epoch": epoch, **{f"train/{key}": value for key, value in train_metrics.items()}, **{f"val/{key}": value for key, value in val_metrics.items()}}
            metrics_writer.log(row)
            wandb_run.log(row, step=epoch)
            logger.info("Epoch %d train=%s val=%s", epoch, train_metrics, val_metrics)
            scheduler.step()
            monitor = float(val_metrics["loss"])
            if epoch == start_epoch or (np.isfinite(monitor) and monitor < best_loss):
                best_loss = monitor if np.isfinite(monitor) else best_loss
                best_checkpoint = save_checkpoint(
                    run_paths.checkpoints / "best.pt",
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "primitive_embed": primitive_embed.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "planner": planner.state_dict() if planner is not None else None,
                        "note_surrogate": note_surrogate.state_dict(),
                        "planner_config": planner_config,
                        "planner_metadata": planner_metadata,
                        "config": config,
                    },
                )
                wandb_run.summary({"best/epoch": epoch, "best/val_loss": best_loss, "best/checkpoint": str(best_checkpoint)})
                if samples is not None:
                    save_npz(
                        run_paths.artifacts / "val_samples.npz",
                        predicted=samples["predicted"],
                        target=samples["target"],
                        prior=samples["prior"],
                        residual=samples["residual"],
                        predicted_note=samples["predicted_note"],
                        target_note=samples["target_note"],
                        predicted_hold=samples["predicted_hold"],
                        target_hold=samples["target_hold"],
                        predicted_sustain=samples["predicted_sustain"],
                        target_sustain=samples["target_sustain"],
                    )
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
                        "note_surrogate": note_surrogate.state_dict(),
                        "planner_config": planner_config,
                        "planner_metadata": planner_metadata,
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


def maybe_load_planner(config: dict[str, Any], metadata: DiffusionMetadata, device: torch.device):
    variant = str(config["variant"])
    if variant in {"diffusion_only", "gmr_only"}:
        return None, 0, None, None
    checkpoint_path = Path(config["planner_checkpoint"]).resolve() if config.get("planner_checkpoint") else None
    if checkpoint_path is None or not checkpoint_path.exists():
        return None, 0, None, None
    payload = load_checkpoint(checkpoint_path, map_location=device)
    planner_config = resolve_planner_checkpoint_config(payload)
    planner_metadata_payload = payload.get("metadata") or payload.get("planner_metadata")
    if payload.get("planner_architecture") not in {FACTORED_PLANNER_ARCHITECTURE, None}:
        raise ValueError(f"Unsupported planner architecture in checkpoint {checkpoint_path}: {payload.get('planner_architecture')!r}")
    if payload.get("planner_architecture") is None and str(planner_config.get("model_variant", "")) != "factored_goal_conditioned":
        raise ValueError(
            f"Planner checkpoint {checkpoint_path} predates the factored planner redesign. Retrain Stage 2 with the updated transformer configs."
        )
    validate_planner_metadata_compatibility(planner_metadata_payload, metadata, checkpoint_path)
    planner = build_planner_from_config(metadata_to_planner(metadata), planner_config).to(device)
    planner_state = extract_planner_state_dict(payload)
    try:
        planner.load_state_dict(planner_state)
    except RuntimeError as exc:
        raise ValueError(
            f"Planner checkpoint {checkpoint_path} is incompatible with the current factored planner architecture."
        ) from exc
    planner_config = dict(planner_config)
    planner_config["planner_architecture"] = FACTORED_PLANNER_ARCHITECTURE
    return planner, planner_output_dim_from_config(planner_config), planner_config, planner_metadata_payload


def resolve_planner_checkpoint_config(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("planner_config"):
        return dict(payload["planner_config"])
    return dict(payload.get("config", {}))


def extract_planner_state_dict(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("planner") is not None:
        return payload["planner"]
    model_state = payload.get("model", {})
    if any(str(key).startswith("planner.") for key in model_state.keys()):
        return {str(key)[len("planner."):]: value for key, value in model_state.items() if str(key).startswith("planner.")}
    return model_state


def validate_planner_metadata_compatibility(payload: dict[str, Any] | None, metadata: DiffusionMetadata, checkpoint_path: Path) -> None:
    if not payload:
        return
    if int(payload.get("num_primitives", metadata.num_primitives)) != int(metadata.num_primitives):
        raise ValueError(f"Planner checkpoint {checkpoint_path} was trained with a different primitive vocabulary size.")
    if [int(item) for item in payload.get("primitive_to_family", metadata.primitive_to_family)] != [int(item) for item in metadata.primitive_to_family]:
        raise ValueError(
            f"Planner checkpoint {checkpoint_path} uses a different primitive-family mapping. "
            "Use the same primitive_root and family_mapping_mode as the planner training run."
        )
    if int(payload.get("num_families", metadata.num_families)) != int(metadata.num_families):
        raise ValueError(f"Planner checkpoint {checkpoint_path} was trained with a different number of planner families.")


def diffusion_epoch(
    model,
    primitive_embed,
    planner,
    loader,
    optimizer,
    diffusion,
    note_surrogate,
    device,
    config: dict[str, Any],
    train: bool,
):
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
    note_surrogate.eval()
    losses: list[float] = []
    note_losses: list[float] = []
    hold_losses: list[float] = []
    diffusion_losses: list[float] = []
    imitation_losses: list[float] = []
    smoothness_losses: list[float] = []
    residual_regs: list[float] = []
    residual_magnitudes: list[float] = []
    note_precisions: list[float] = []
    note_recalls: list[float] = []
    note_f1s: list[float] = []
    hold_precisions: list[float] = []
    hold_recalls: list[float] = []
    hold_f1s: list[float] = []
    sustain_precisions: list[float] = []
    sustain_recalls: list[float] = []
    sustain_f1s: list[float] = []
    sample_bundle = None
    iterator = tqdm(loader, desc="Train diffusion" if train else "Eval diffusion", leave=False)
    for batch in iterator:
        batch = move_to_device(batch, str(device))
        prior = prepare_variant_prior(batch["gmr_prior"], str(config["variant"]))
        target = batch["action_target"]
        predict_residual = bool(config.get("predict_residual", False))
        if str(config["variant"]) == "gmr_only":
            predicted_model_output = torch.zeros_like(target) if predict_residual else prior
            predicted_action = compose_final_action(prior, predicted_model_output, predict_residual=predict_residual)
            predicted_residual = predicted_action - prior
            objective = compute_control_objective(
                predicted_action=predicted_action,
                predicted_residual=predicted_residual,
                diffusion_loss=target.new_tensor(0.0),
                batch=batch,
                note_surrogate=note_surrogate,
                config=config,
            )
            total = objective["total_loss"]
            losses.append(float(total.item()))
            diffusion_losses.append(float(objective["diffusion_loss"].item()))
            note_losses.append(float(objective["note_loss"].item()))
            hold_losses.append(float(objective["hold_loss"].item()))
            imitation_losses.append(float(objective["imitation_loss"].item()))
            smoothness_losses.append(float(objective["smoothness_loss"].item()))
            residual_regs.append(float(objective["residual_reg_loss"].item()))
            residual_magnitudes.append(float(objective["residual_magnitude"].item()))
            note_precisions.append(objective["metrics"]["note_precision"])
            note_recalls.append(objective["metrics"]["note_recall"])
            note_f1s.append(objective["metrics"]["note_f1"])
            hold_precisions.append(objective["metrics"]["hold_precision"])
            hold_recalls.append(objective["metrics"]["hold_recall"])
            hold_f1s.append(objective["metrics"]["hold_f1"])
            sustain_precisions.append(objective["metrics"]["sustain_precision"])
            sustain_recalls.append(objective["metrics"]["sustain_recall"])
            sustain_f1s.append(objective["metrics"]["sustain_f1"])
            if sample_bundle is None:
                sample_bundle = build_sample_bundle(
                    predicted_action=predicted_action,
                    predicted_residual=predicted_residual,
                    prior=prior,
                    batch=batch,
                    surrogate_outputs=objective["surrogate_outputs"],
                )
            continue
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        diffusion_target = target - prior if predict_residual else target
        timestep = diffusion.sample_timesteps(target.shape[0])
        noise = torch.randn_like(diffusion_target)
        noisy = diffusion.q_sample(diffusion_target, timestep, noise)
        condition = build_condition_vector(batch, primitive_embed, planner, config)
        predicted_noise = model(noisy, prior, timestep, condition)
        diffusion_loss = F.mse_loss(predicted_noise, noise)
        predicted_denoised = diffusion.predict_x0(noisy, timestep, predicted_noise)
        predicted_action = compose_final_action(prior, predicted_denoised, predict_residual=predict_residual)
        predicted_residual = predicted_action - prior
        objective = compute_control_objective(
            predicted_action=predicted_action,
            predicted_residual=predicted_residual,
            diffusion_loss=diffusion_loss,
            batch=batch,
            note_surrogate=note_surrogate,
            config=config,
        )
        total = objective["total_loss"]
        if train and optimizer is not None:
            total.backward()
            optimizer.step()
        losses.append(float(total.item()))
        diffusion_losses.append(float(objective["diffusion_loss"].item()))
        note_losses.append(float(objective["note_loss"].item()))
        hold_losses.append(float(objective["hold_loss"].item()))
        imitation_losses.append(float(objective["imitation_loss"].item()))
        smoothness_losses.append(float(objective["smoothness_loss"].item()))
        residual_regs.append(float(objective["residual_reg_loss"].item()))
        residual_magnitudes.append(float(objective["residual_magnitude"].item()))
        note_precisions.append(objective["metrics"]["note_precision"])
        note_recalls.append(objective["metrics"]["note_recall"])
        note_f1s.append(objective["metrics"]["note_f1"])
        hold_precisions.append(objective["metrics"]["hold_precision"])
        hold_recalls.append(objective["metrics"]["hold_recall"])
        hold_f1s.append(objective["metrics"]["hold_f1"])
        sustain_precisions.append(objective["metrics"]["sustain_precision"])
        sustain_recalls.append(objective["metrics"]["sustain_recall"])
        sustain_f1s.append(objective["metrics"]["sustain_f1"])
        if sample_bundle is None:
            sample_bundle = build_sample_bundle(
                predicted_action=predicted_action,
                predicted_residual=predicted_residual,
                prior=prior,
                batch=batch,
                surrogate_outputs=objective["surrogate_outputs"],
            )
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "diffusion_loss": float(np.mean(diffusion_losses)) if diffusion_losses else 0.0,
        "note_loss": float(np.mean(note_losses)) if note_losses else 0.0,
        "hold_loss": float(np.mean(hold_losses)) if hold_losses else 0.0,
        "imitation_l1": float(np.mean(imitation_losses)) if imitation_losses else 0.0,
        "smoothness": float(np.mean(smoothness_losses)) if smoothness_losses else 0.0,
        "residual_reg": float(np.mean(residual_regs)) if residual_regs else 0.0,
        "residual_magnitude": float(np.mean(residual_magnitudes)) if residual_magnitudes else 0.0,
        "note_precision": float(np.mean(note_precisions)) if note_precisions else 0.0,
        "note_recall": float(np.mean(note_recalls)) if note_recalls else 0.0,
        "note_f1": float(np.mean(note_f1s)) if note_f1s else 0.0,
        "hold_precision": float(np.mean(hold_precisions)) if hold_precisions else 0.0,
        "hold_recall": float(np.mean(hold_recalls)) if hold_recalls else 0.0,
        "hold_f1": float(np.mean(hold_f1s)) if hold_f1s else 0.0,
        "sustain_precision": float(np.mean(sustain_precisions)) if sustain_precisions else 0.0,
        "sustain_recall": float(np.mean(sustain_recalls)) if sustain_recalls else 0.0,
        "sustain_f1": float(np.mean(sustain_f1s)) if sustain_f1s else 0.0,
    }, sample_bundle


def build_condition_vector(batch, primitive_embed, planner, config: dict[str, Any]) -> torch.Tensor:
    score = torch.nan_to_num(batch["score_context"])
    state = torch.nan_to_num(batch["state_context"])
    scalar = torch.nan_to_num(
        torch.stack(
            [
                batch["duration_bucket"].float(),
                batch["dynamics_bucket"].float(),
                batch["primitive_index"].float(),
            ],
            dim=-1,
        )
    )
    if planner is not None and str(config["variant"]) != "diffusion_only":
        with torch.set_grad_enabled(any(parameter.requires_grad for parameter in planner.parameters())):
            planner_outputs = planner(batch)
            plan = torch.nan_to_num(planner_outputs["plan_embedding"])
        return torch.nan_to_num(torch.cat([score, state, scalar, plan], dim=-1))
    primitive = torch.nan_to_num(primitive_embed(batch["primitive_index"]))
    return torch.nan_to_num(torch.cat([score, state, scalar, primitive], dim=-1))


def prepare_variant_prior(prior: torch.Tensor, variant: str) -> torch.Tensor:
    if variant in {"diffusion_only", "planner_no_prior"}:
        return torch.zeros_like(prior)
    return prior


def compose_final_action(prior: torch.Tensor, model_output: torch.Tensor, *, predict_residual: bool) -> torch.Tensor:
    if predict_residual:
        return prior + model_output
    return model_output


def asymmetric_binary_loss(logits: torch.Tensor, targets: torch.Tensor, *, false_negative_weight: float, false_positive_weight: float) -> torch.Tensor:
    positive = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits), reduction="none")
    negative = F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits), reduction="none")
    target = targets.float()
    weights = target * float(false_negative_weight) + (1.0 - target) * float(false_positive_weight)
    losses = target * positive + (1.0 - target) * negative
    return (weights * losses).mean()


def compute_note_loss(logits: torch.Tensor, targets: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
    return asymmetric_binary_loss(
        logits,
        targets,
        false_negative_weight=float(config.get("false_negative_weight", 1.0)),
        false_positive_weight=float(config.get("false_positive_weight", 1.0)),
    )


def compute_hold_loss(
    hold_logits: torch.Tensor,
    hold_targets: torch.Tensor,
    sustain_logits: torch.Tensor,
    sustain_targets: torch.Tensor,
    config: dict[str, Any],
) -> torch.Tensor:
    hold_loss = asymmetric_binary_loss(
        hold_logits,
        hold_targets,
        false_negative_weight=float(config.get("false_negative_weight", 1.0)),
        false_positive_weight=float(config.get("false_positive_weight", 1.0)),
    )
    sustain_loss = asymmetric_binary_loss(
        sustain_logits,
        sustain_targets,
        false_negative_weight=float(config.get("false_negative_weight", 1.0)),
        false_positive_weight=float(config.get("false_positive_weight", 1.0)),
    )
    return 0.5 * (hold_loss + sustain_loss)


def compute_binary_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    prediction = logits > 0.0
    truth = targets > 0.5
    tp = torch.logical_and(prediction, truth).sum().item()
    fp = torch.logical_and(prediction, torch.logical_not(truth)).sum().item()
    fn = torch.logical_and(torch.logical_not(prediction), truth).sum().item()
    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = float(2.0 * precision * recall / (precision + recall))
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_control_objective(
    *,
    predicted_action: torch.Tensor,
    predicted_residual: torch.Tensor,
    diffusion_loss: torch.Tensor,
    batch: dict[str, torch.Tensor],
    note_surrogate: LinearActionNoteSurrogate,
    config: dict[str, Any],
) -> dict[str, Any]:
    surrogate_outputs = note_surrogate(predicted_action)
    note_loss = compute_note_loss(surrogate_outputs["note_logits"], batch["note_target"], config)
    hold_loss = compute_hold_loss(
        surrogate_outputs["hold_logits"],
        batch["hold_target"],
        surrogate_outputs["sustain_logits"],
        batch["sustain_target"],
        config,
    )
    imitation_loss = F.l1_loss(predicted_action, batch["action_target"])
    smoothness_loss = ((predicted_action[:, 1:] - predicted_action[:, :-1]) ** 2).mean() if predicted_action.shape[1] > 1 else predicted_action.new_tensor(0.0)
    residual_reg_loss = (predicted_residual ** 2).mean()
    residual_magnitude = predicted_residual.abs().mean()
    total_loss = (
        float(config.get("note_loss_weight", 1.0)) * note_loss
        + float(config.get("hold_loss_weight", 0.0)) * hold_loss
        + float(config.get("imitation_weight", 0.0)) * imitation_loss
        + float(config.get("smoothness_weight", 0.0)) * smoothness_loss
        + float(config.get("residual_reg_weight", 0.0)) * residual_reg_loss
        + float(config.get("diffusion_weight", 1.0)) * diffusion_loss
    )
    note_metrics = compute_binary_metrics(surrogate_outputs["note_logits"], batch["note_target"])
    hold_metrics = compute_binary_metrics(surrogate_outputs["hold_logits"], batch["hold_target"])
    sustain_metrics = compute_binary_metrics(surrogate_outputs["sustain_logits"], batch["sustain_target"])
    return {
        "total_loss": total_loss,
        "diffusion_loss": diffusion_loss,
        "note_loss": note_loss,
        "hold_loss": hold_loss,
        "imitation_loss": imitation_loss,
        "smoothness_loss": smoothness_loss,
        "residual_reg_loss": residual_reg_loss,
        "residual_magnitude": residual_magnitude,
        "surrogate_outputs": surrogate_outputs,
        "metrics": {
            "note_precision": note_metrics["precision"],
            "note_recall": note_metrics["recall"],
            "note_f1": note_metrics["f1"],
            "hold_precision": hold_metrics["precision"],
            "hold_recall": hold_metrics["recall"],
            "hold_f1": hold_metrics["f1"],
            "sustain_precision": sustain_metrics["precision"],
            "sustain_recall": sustain_metrics["recall"],
            "sustain_f1": sustain_metrics["f1"],
        },
    }


def build_sample_bundle(
    *,
    predicted_action: torch.Tensor,
    predicted_residual: torch.Tensor,
    prior: torch.Tensor,
    batch: dict[str, torch.Tensor],
    surrogate_outputs: dict[str, torch.Tensor],
) -> dict[str, np.ndarray]:
    return {
        "predicted": predicted_action[:8].detach().cpu().numpy(),
        "target": batch["action_target"][:8].detach().cpu().numpy(),
        "prior": prior[:8].detach().cpu().numpy(),
        "residual": predicted_residual[:8].detach().cpu().numpy(),
        "predicted_note": (surrogate_outputs["note_logits"][:8] > 0.0).detach().cpu().numpy().astype(np.float32),
        "target_note": batch["note_target"][:8].detach().cpu().numpy(),
        "predicted_hold": (surrogate_outputs["hold_logits"][:8] > 0.0).detach().cpu().numpy().astype(np.float32),
        "target_hold": batch["hold_target"][:8].detach().cpu().numpy(),
        "predicted_sustain": (surrogate_outputs["sustain_logits"][:8] > 0.0).detach().cpu().numpy().astype(np.float32),
        "target_sustain": batch["sustain_target"][:8].detach().cpu().numpy(),
    }
