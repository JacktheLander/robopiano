from __future__ import annotations

import csv
import json
import random
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from variations.data.dataset import PressPairsDataset, build_splits, compute_norm_stats, split_target_coverage
from variations.diffusion.model import VariationsDenoiser
from variations.diffusion.schedule import GaussianDiffusion
from variations.utils.config import diffusion_run_root, extraction_root, save_config
from variations.utils.io import ensure_dir, save_json


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow = {name: param.detach().clone() for name, param in model.state_dict().items() if param.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        state = model.state_dict()
        for name, value in state.items():
            if name not in self.shadow or not value.dtype.is_floating_point:
                continue
            self.shadow[name].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: value.detach().clone() for name, value in self.shadow.items()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.shadow = {name: value.detach().clone() for name, value in state.items()}

    @contextmanager
    def average_parameters(self, model: torch.nn.Module) -> Iterator[None]:
        backup = {}
        state = model.state_dict()
        for name, value in state.items():
            if name in self.shadow:
                backup[name] = value.detach().clone()
                value.copy_(self.shadow[name].to(value.device))
        try:
            yield
        finally:
            state = model.state_dict()
            for name, value in backup.items():
                state[name].copy_(value)


class MetricsWriter:
    def __init__(self, metrics_dir: Path) -> None:
        self.metrics_dir = ensure_dir(metrics_dir)
        self.csv_path = self.metrics_dir / "metrics.csv"
        self.jsonl_path = self.metrics_dir / "metrics.jsonl"
        self.fieldnames: list[str] | None = None

    def log(self, row: dict[str, Any]) -> None:
        flat = {key: (float(value) if isinstance(value, np.floating) else value) for key, value in row.items()}
        if self.fieldnames is None:
            self.fieldnames = list(flat.keys())
            write_header = not self.csv_path.exists()
        else:
            for key in flat.keys():
                if key not in self.fieldnames:
                    self.fieldnames.append(key)
            write_header = not self.csv_path.exists()
        with self.csv_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({key: flat.get(key, "") for key in self.fieldnames})
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(flat, sort_keys=True) + "\n")


@dataclass
class ValidationResult:
    metrics: dict[str, float]
    sample_bundle: dict[str, np.ndarray] | None


def count_parameters(model: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def dataset_stats(dataset: PressPairsDataset) -> dict[str, float]:
    if len(dataset) == 0:
        return {
            "pairs": 0,
            "mean_chord_size": 0.0,
            "max_chord_size": 0.0,
            "single_key_fraction": 0.0,
        }
    chord_sizes = dataset.target_keys.sum(axis=1)
    return {
        "pairs": int(len(dataset)),
        "mean_chord_size": float(np.mean(chord_sizes)),
        "max_chord_size": float(np.max(chord_sizes)),
        "single_key_fraction": float(np.mean(chord_sizes == 1)),
    }


def build_model_from_config(config: dict[str, Any]) -> VariationsDenoiser:
    model_cfg = config.get("model", {})
    return VariationsDenoiser(
        hidden_dim=int(model_cfg.get("hidden_dim", 256)),
        num_blocks=int(model_cfg.get("num_blocks", 6)),
        time_dim=int(model_cfg.get("time_dim", 128)),
        cond_dim=int(model_cfg.get("cond_dim", model_cfg.get("hidden_dim", 256))),
    )


def build_diffusion_from_config(config: dict[str, Any], device: torch.device) -> GaussianDiffusion:
    cfg = config.get("diffusion", {})
    return GaussianDiffusion(
        timesteps=int(cfg.get("timesteps", 1000)),
        beta_start=float(cfg.get("beta_start", 0.0001)),
        beta_end=float(cfg.get("beta_end", 0.02)),
        schedule=str(cfg.get("schedule", "linear")),
        device=device,
    )


def _extraction_root(config: dict[str, Any]) -> Path:
    return extraction_root(config)


def _prepare_datasets(config: dict[str, Any]) -> tuple[PressPairsDataset, PressPairsDataset, Path]:
    extraction_root = _extraction_root(config)
    split_cfg = config.get("splits", {})
    split_path = extraction_root / "splits" / "split_index.csv"
    if not split_path.exists():
        build_splits(
            extraction_root,
            val_fraction=float(split_cfg.get("val_fraction", 0.1)),
            seed=int(split_cfg.get("seed", 42)),
            min_pairs_per_split=int(split_cfg.get("min_pairs_per_split", 1000)),
        )
    norm_path = extraction_root / "splits" / "norm_stats.npz"
    if not norm_path.exists():
        compute_norm_stats(extraction_root)
    train_dataset = PressPairsDataset(extraction_root, split="train", norm_stats_path=norm_path, assert_unique_goals=True)
    val_dataset = PressPairsDataset(extraction_root, split="val", norm_stats_path=norm_path, assert_unique_goals=True)
    return train_dataset, val_dataset, norm_path


def _unnormalize(x: torch.Tensor, dataset: PressPairsDataset, device: torch.device) -> torch.Tensor:
    mean = torch.as_tensor(dataset.mean, dtype=x.dtype, device=device)
    std = torch.as_tensor(dataset.std, dtype=x.dtype, device=device)
    return x * std + mean


def train_epoch(
    model: VariationsDenoiser,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    diffusion: GaussianDiffusion,
    device: torch.device,
    *,
    amp: bool,
    scaler,
    ema: EMA | None,
) -> dict[str, float]:
    model.train()
    losses = []
    iterator = tqdm(loader, desc="train", leave=False)
    for batch in iterator:
        target_keys = batch["target_keys"].to(device)
        x0 = batch["hand_state_normalized"].to(device)
        timestep = diffusion.sample_timesteps(x0.shape[0])
        noise = torch.randn_like(x0)
        noisy = diffusion.q_sample(x0, timestep, noise)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            pred = model(noisy, timestep, target_keys)
            loss = F.mse_loss(pred, noise)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if ema is not None:
            ema.update(model)
        losses.append(float(loss.detach().cpu().item()))
        iterator.set_postfix(loss=f"{losses[-1]:.4f}")
    return {"eps_mse": float(np.mean(losses)) if losses else 0.0}


@torch.no_grad()
def validate(
    model: VariationsDenoiser,
    loader: DataLoader,
    diffusion: GaussianDiffusion,
    device: torch.device,
    val_dataset: PressPairsDataset,
    *,
    best_of_k: int,
    ddim_steps: int,
    max_batches: int | None,
    sample_artifact_count: int = 32,
) -> ValidationResult:
    model.eval()
    eps_losses = []
    best_x0_mse = []
    fingertip_l2 = []
    sample_bundle = None
    for batch_idx, batch in enumerate(tqdm(loader, desc="val", leave=False)):
        target_keys = batch["target_keys"].to(device)
        x0 = batch["hand_state_normalized"].to(device)
        timestep = diffusion.sample_timesteps(x0.shape[0])
        noise = torch.randn_like(x0)
        noisy = diffusion.q_sample(x0, timestep, noise)
        pred = model(noisy, timestep, target_keys)
        eps_losses.append(float(F.mse_loss(pred, noise).detach().cpu().item()))

        samples = []
        for _ in range(max(int(best_of_k), 1)):
            sample_norm = diffusion.p_sample_loop(
                model,
                (x0.shape[0], x0.shape[1]),
                target_keys,
                num_inference_steps=ddim_steps,
            )
            samples.append(_unnormalize(sample_norm, val_dataset, device))
        stacked = torch.stack(samples, dim=1)
        truth = batch["hand_state"].to(device).unsqueeze(1)
        mse = ((stacked - truth) ** 2).mean(dim=2)
        best_idx = mse.argmin(dim=1)
        best = stacked[torch.arange(stacked.shape[0], device=device), best_idx]
        batch_best_mse = mse.min(dim=1).values
        batch_fingertip_l2 = torch.linalg.norm(best[:, -30:] - truth[:, 0, -30:], dim=1)
        best_x0_mse.extend(batch_best_mse.detach().cpu().tolist())
        fingertip_l2.extend(batch_fingertip_l2.detach().cpu().tolist())
        if sample_bundle is None and sample_artifact_count > 0:
            n = min(int(sample_artifact_count), stacked.shape[0])
            sample_bundle = {
                "target_keys": batch["target_keys"][:n].detach().cpu().numpy(),
                "hand_state_true": batch["hand_state"][:n].detach().cpu().numpy(),
                "hand_state_samples": stacked[:n].detach().cpu().numpy(),
                "hand_state_best": best[:n].detach().cpu().numpy(),
                "best_sample_index": best_idx[:n].detach().cpu().numpy().astype(np.int32),
                "x0_mse_best": batch_best_mse[:n].detach().cpu().numpy(),
                "fingertip_l2": batch_fingertip_l2[:n].detach().cpu().numpy(),
                "chord_size": batch["target_keys"][:n].sum(dim=1).detach().cpu().numpy(),
            }
        if max_batches is not None and batch_idx + 1 >= max_batches:
            break
    x0_arr = np.asarray(best_x0_mse, dtype=np.float32)
    fingertip_arr = np.asarray(fingertip_l2, dtype=np.float32)
    metrics = {
        "val_eps_mse": float(np.mean(eps_losses)) if eps_losses else 0.0,
        "val_x0_mse_best_of_k": float(np.mean(x0_arr)) if x0_arr.size else 0.0,
        "val_x0_mse_p50": float(np.percentile(x0_arr, 50)) if x0_arr.size else 0.0,
        "val_x0_mse_p90": float(np.percentile(x0_arr, 90)) if x0_arr.size else 0.0,
        "val_fingertip_l2_mean": float(np.mean(fingertip_arr)) if fingertip_arr.size else 0.0,
        "val_fingertip_l2_p50": float(np.percentile(fingertip_arr, 50)) if fingertip_arr.size else 0.0,
        "val_fingertip_l2_p90": float(np.percentile(fingertip_arr, 90)) if fingertip_arr.size else 0.0,
    }
    if sample_bundle is not None:
        sample_bundle["all_x0_mse_best"] = x0_arr
        sample_bundle["all_fingertip_l2"] = fingertip_arr
    return ValidationResult(metrics=metrics, sample_bundle=sample_bundle)


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    torch.save(payload, path)


def run_training(config: dict[str, Any]) -> dict[str, Path]:
    set_seed(int(config.get("seed", 42)))
    train_dataset, val_dataset, norm_path = _prepare_datasets(config)
    device = resolve_device(str(config.get("training", {}).get("device", "auto")))
    data_cfg = config.get("data", {})
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(data_cfg.get("batch_size", 256)),
        shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 0)),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(data_cfg.get("batch_size", 256)),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 0)),
        drop_last=False,
    )
    run_root = diffusion_run_root(config)
    checkpoints = ensure_dir(run_root / "checkpoints")
    artifacts = ensure_dir(run_root / "artifacts")
    metrics_dir = ensure_dir(run_root / "metrics")
    ensure_dir(run_root / "logs")
    save_config(artifacts / "config.yaml", config)
    dataset_summary = {
        "train": dataset_stats(train_dataset),
        "val": dataset_stats(val_dataset),
        "norm_stats": str(norm_path),
    }
    save_json(artifacts / "dataset_summary.json", dataset_summary)

    model = build_model_from_config(config).to(device)
    diffusion = build_diffusion_from_config(config, device)
    train_cfg = config.get("training", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    epochs = int(train_cfg.get("epochs", 100))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    amp = bool(train_cfg.get("amp", False)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp) if amp else None
    ema = EMA(model, float(train_cfg.get("ema_decay", 0.999)))
    metrics = MetricsWriter(metrics_dir)
    best_metric = float("inf")
    best_checkpoint = checkpoints / "best.pt"
    wandb_run = _maybe_start_wandb(config, run_root)
    coverage = split_target_coverage(train_dataset.target_keys, val_dataset.target_keys)
    if wandb_run is not None:
        _wandb_summary(
            wandb_run,
            {
                "run_root": str(run_root),
                "extraction_root": str(_extraction_root(config)),
                "norm_stats": str(norm_path),
                "device": str(device),
                "model/num_parameters": count_parameters(model),
                "data/train_pairs": len(train_dataset),
                "data/val_pairs": len(val_dataset),
                "data/val_target_keys_coverage": coverage,
                "data/train_mean_chord_size": dataset_summary["train"]["mean_chord_size"],
                "data/val_mean_chord_size": dataset_summary["val"]["mean_chord_size"],
            },
        )
        if bool(config.get("wandb", {}).get("watch_model", True)):
            try:
                wandb_run.watch(model, log=str(config.get("wandb", {}).get("watch_log", "gradients")), log_freq=int(config.get("wandb", {}).get("watch_log_freq", 100)))
            except Exception as exc:
                print(f"wandb watch failed; continuing without model watch: {exc}")

    for epoch in range(epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, diffusion, device, amp=amp, scaler=scaler, ema=ema)
        scheduler.step()
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_eps_mse": train_metrics["eps_mse"],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "val_target_keys_coverage": coverage,
        }
        validation_result: ValidationResult | None = None
        if (epoch + 1) % int(train_cfg.get("val_every", 1)) == 0 or epoch == epochs - 1:
            with ema.average_parameters(model):
                validation_result = validate(
                    model,
                    val_loader,
                    diffusion,
                    device,
                    val_dataset,
                    best_of_k=int(train_cfg.get("best_of_k", 8)),
                    ddim_steps=int(config.get("diffusion", {}).get("ddim_steps", 50)),
                    max_batches=train_cfg.get("max_val_batches"),
                    sample_artifact_count=int(train_cfg.get("sample_artifact_count", 32)),
                )
            row.update(validation_result.metrics)
            _save_validation_samples(artifacts / "latest_val_samples.npz", validation_result.sample_bundle)
            monitor = float(validation_result.metrics["val_x0_mse_best_of_k"])
            if monitor < best_metric:
                best_metric = monitor
                save_checkpoint(best_checkpoint, _checkpoint_payload(config, model, ema, optimizer, scheduler, epoch, best_metric, norm_path))
                _save_validation_samples(artifacts / "best_val_samples.npz", validation_result.sample_bundle)
                if wandb_run is not None and bool(config.get("wandb", {}).get("log_artifacts", True)):
                    _wandb_log_file(wandb_run, best_checkpoint, f"{run_root.name}-best-checkpoint", "model", aliases=["best", f"epoch-{epoch}"])
                    best_samples = artifacts / "best_val_samples.npz"
                    if best_samples.exists():
                        _wandb_log_file(wandb_run, best_samples, f"{run_root.name}-best-val-samples", "validation-samples", aliases=["best", "latest"])
        save_checkpoint(checkpoints / "last.pt", _checkpoint_payload(config, model, ema, optimizer, scheduler, epoch, best_metric, norm_path))
        if (epoch + 1) % int(train_cfg.get("checkpoint_interval", 10)) == 0:
            save_checkpoint(checkpoints / f"epoch_{epoch:04d}.pt", _checkpoint_payload(config, model, ema, optimizer, scheduler, epoch, best_metric, norm_path))
        metrics.log(row)
        if wandb_run is not None:
            _wandb_log(wandb_run, _wandb_payload(row, validation_result), step=epoch)
        print(row)
    if wandb_run is not None:
        if bool(config.get("wandb", {}).get("log_artifacts", True)):
            _wandb_log_dir(wandb_run, metrics_dir, f"{run_root.name}-metrics", "metrics", aliases=["latest"])
            _wandb_log_dir(wandb_run, artifacts, f"{run_root.name}-artifacts", "run-output", aliases=["latest"])
            last_checkpoint = checkpoints / "last.pt"
            if last_checkpoint.exists():
                _wandb_log_file(wandb_run, last_checkpoint, f"{run_root.name}-last-checkpoint", "model", aliases=["latest"])
        _wandb_summary(wandb_run, {"best/val_x0_mse_best_of_k": best_metric, "best/checkpoint": str(best_checkpoint), "status": "completed"})
        wandb_run.finish()
    return {"run_root": run_root, "best_checkpoint": best_checkpoint}


def _checkpoint_payload(
    config: dict[str, Any],
    model: VariationsDenoiser,
    ema: EMA,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_metric: float,
    norm_path: Path,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_metric": best_metric,
        "config": config,
        "norm_stats_path": str(norm_path),
    }


def _maybe_start_wandb(config: dict[str, Any], run_root: Path):
    wandb_cfg = config.get("wandb", {})
    if not bool(wandb_cfg.get("enabled", False)):
        return None
    try:
        import wandb
    except ImportError:
        print("wandb is enabled but not installed; continuing without wandb.")
        return None
    try:
        return wandb.init(
            project=str(wandb_cfg.get("project", "variations")),
            name=run_root.name,
            config=config,
            dir=str(run_root),
            job_type=str(wandb_cfg.get("job_type", "diffusion")),
            tags=list(wandb_cfg.get("tags", ["variations", "diffusion"])),
            mode=str(wandb_cfg.get("mode", "online")),
        )
    except Exception as exc:
        print(f"wandb init failed; continuing without wandb: {exc}")
        return None


def _wandb_summary(wandb_run, payload: dict[str, Any]) -> None:
    try:
        wandb_run.summary.update(payload)
    except Exception as exc:
        print(f"wandb summary update failed; continuing: {exc}")


def _wandb_log(wandb_run, payload: dict[str, Any], *, step: int) -> None:
    try:
        wandb_run.log(payload, step=step)
    except Exception as exc:
        print(f"wandb log failed at step {step}; continuing: {exc}")


def _save_validation_samples(path: Path, sample_bundle: dict[str, np.ndarray] | None) -> None:
    if not sample_bundle:
        return
    ensure_dir(path.parent)
    np.savez_compressed(path, **sample_bundle)


def _wandb_payload(row: dict[str, Any], validation_result: ValidationResult | None) -> dict[str, Any]:
    payload = dict(row)
    if validation_result is None or validation_result.sample_bundle is None:
        return payload
    try:
        import wandb
    except ImportError:
        return payload
    bundle = validation_result.sample_bundle
    if "all_x0_mse_best" in bundle:
        payload["val/x0_mse_best_hist"] = wandb.Histogram(bundle["all_x0_mse_best"])
    if "all_fingertip_l2" in bundle:
        payload["val/fingertip_l2_hist"] = wandb.Histogram(bundle["all_fingertip_l2"])
    if {"chord_size", "x0_mse_best", "fingertip_l2"} <= set(bundle.keys()):
        table = wandb.Table(columns=["idx", "chord_size", "x0_mse_best", "fingertip_l2", "best_sample_index"])
        count = len(bundle["chord_size"])
        for idx in range(count):
            table.add_data(
                idx,
                float(bundle["chord_size"][idx]),
                float(bundle["x0_mse_best"][idx]),
                float(bundle["fingertip_l2"][idx]),
                int(bundle["best_sample_index"][idx]),
            )
        payload["val/sample_error_table"] = table
    return payload


def _wandb_log_file(wandb_run, path: Path, name: str, artifact_type: str, aliases: list[str]) -> None:
    try:
        import wandb
    except ImportError:
        return
    artifact = wandb.Artifact(name=name, type=artifact_type)
    artifact.add_file(str(path))
    try:
        wandb_run.log_artifact(artifact, aliases=aliases)
    except Exception as exc:
        print(f"wandb artifact upload failed for {path}; continuing: {exc}")


def _wandb_log_dir(wandb_run, path: Path, name: str, artifact_type: str, aliases: list[str]) -> None:
    try:
        import wandb
    except ImportError:
        return
    artifact = wandb.Artifact(name=name, type=artifact_type)
    artifact.add_dir(str(path))
    try:
        wandb_run.log_artifact(artifact, aliases=aliases)
    except Exception as exc:
        print(f"wandb artifact upload failed for {path}; continuing: {exc}")


def load_model_for_inference(checkpoint_path: str | Path, device: torch.device | str = "auto") -> tuple[VariationsDenoiser, GaussianDiffusion, dict[str, Any], np.ndarray, np.ndarray]:
    device = resolve_device(str(device))
    payload = torch.load(checkpoint_path, map_location=device)
    config = payload["config"]
    model = build_model_from_config(config).to(device)
    model.load_state_dict(payload["model"])
    if payload.get("ema"):
        ema_state = payload["ema"]
        model_state = model.state_dict()
        for name, value in ema_state.items():
            if name in model_state:
                model_state[name].copy_(value.to(device))
    model.eval()
    diffusion = build_diffusion_from_config(config, device)
    norm_path = Path(payload.get("norm_stats_path") or (_extraction_root(config) / "splits" / "norm_stats.npz"))
    data = np.load(norm_path, allow_pickle=False)
    return model, diffusion, config, np.asarray(data["mean"], dtype=np.float32), np.asarray(data["std"], dtype=np.float32)


@torch.no_grad()
def sample_hand_states(
    checkpoint_path: str | Path,
    target_keys: np.ndarray,
    *,
    num_samples: int = 1,
    ddim_steps: int | None = None,
    device: str = "auto",
) -> np.ndarray:
    resolved_device = resolve_device(device)
    model, diffusion, config, mean, std = load_model_for_inference(checkpoint_path, resolved_device)
    target = torch.as_tensor(target_keys, dtype=torch.float32, device=resolved_device)
    if target.ndim != 2 or target.shape[1] != 88:
        raise ValueError(f"target_keys must have shape (N, 88), got {tuple(target.shape)}")
    steps = int(ddim_steps or config.get("diffusion", {}).get("ddim_steps", 50))
    outputs = []
    mean_t = torch.as_tensor(mean, dtype=torch.float32, device=resolved_device)
    std_t = torch.as_tensor(std, dtype=torch.float32, device=resolved_device)
    for _ in range(max(int(num_samples), 1)):
        sample = diffusion.p_sample_loop(model, (target.shape[0], 76), target, num_inference_steps=steps)
        outputs.append((sample * std_t + mean_t).detach().cpu().numpy())
    return np.stack(outputs, axis=1)


def evaluate_checkpoint(config: dict[str, Any], checkpoint_path: str | Path) -> dict[str, Any]:
    _, val_dataset, _ = _prepare_datasets(config)
    device = resolve_device(str(config.get("training", {}).get("device", "auto")))
    model, diffusion, _, _, _ = load_model_for_inference(checkpoint_path, device)
    loader = DataLoader(
        val_dataset,
        batch_size=int(config.get("data", {}).get("batch_size", 256)),
        shuffle=False,
        num_workers=int(config.get("data", {}).get("num_workers", 0)),
    )
    result = validate(
        model,
        loader,
        diffusion,
        device,
        val_dataset,
        best_of_k=int(config.get("training", {}).get("best_of_k", 8)),
        ddim_steps=int(config.get("diffusion", {}).get("ddim_steps", 50)),
        max_batches=config.get("training", {}).get("max_val_batches"),
        sample_artifact_count=256,
    )
    sample_count = min(len(val_dataset), 256)
    samples = sample_hand_states(
        checkpoint_path,
        val_dataset.target_keys[:sample_count],
        num_samples=int(config.get("training", {}).get("best_of_k", 8)),
        ddim_steps=int(config.get("diffusion", {}).get("ddim_steps", 50)),
        device=str(device),
    )
    run_root = diffusion_run_root(config)
    artifacts = ensure_dir(run_root / "artifacts")
    np.savez_compressed(
        artifacts / "samples_val.npz",
        samples=samples,
        target_keys=val_dataset.target_keys[:sample_count],
        hand_state=val_dataset.hand_state[:sample_count],
    )
    save_json(artifacts / "evaluation_metrics.json", result.metrics)
    return {"metrics": result.metrics, "samples": artifacts / "samples_val.npz"}
