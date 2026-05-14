from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

VARIATIONS_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = VARIATIONS_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from variations.data.dataset import PressPairsDataset, build_splits, compute_fingerpred_norm_stats
from variations.fingerpred import (
    FINGERPRED_OUTPUT_MODE,
    FINGERPRED_TARGET_DIM,
    FingerPredNormalizer,
    active_fingertip_metrics,
    masked_autoencoder_input,
    masked_mse,
)
from variations.losses.mdn_loss import mdn_nll_loss
from variations.losses.latent_mdn_loss import select_best_component_mean
from variations.models.latent_mdn import build_latent_mdn
from variations.models.pose_autoencoder import build_pose_autoencoder
from variations.utils.config import extraction_root, load_config, save_config
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


def count_parameters(model: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def output_root(config: dict[str, Any]) -> Path:
    import os

    root = Path(config.get("logging", {}).get("output_root", "Variations/outputs/fingerpred"))
    if os.environ.get("VARIATIONS_FINGERPRED_OUTPUT_ROOT"):
        return Path(os.environ["VARIATIONS_FINGERPRED_OUTPUT_ROOT"])
    if os.environ.get("VARIATIONS_OUTPUT_ROOT") and not root.is_absolute() and root.parts[:2] == ("Variations", "outputs"):
        return Path(os.environ["VARIATIONS_OUTPUT_ROOT"]).joinpath(*root.parts[2:])
    if not root.is_absolute():
        root = Path(config.get("_repo_root", Path.cwd())) / root
    return root / str(config.get("logging", {}).get("run_name", config.get("experiment_name", "fingerpred")))


class MetricsWriter:
    def __init__(self, path: Path) -> None:
        self.path = ensure_dir(path)
        self.csv_path = self.path / "metrics.csv"
        self.jsonl_path = self.path / "metrics.jsonl"
        self.fieldnames: list[str] | None = None

    def log(self, row: dict[str, Any]) -> None:
        row = {key: (float(value) if isinstance(value, np.floating) else value) for key, value in row.items()}
        if self.fieldnames is None:
            self.fieldnames = list(row.keys())
        else:
            for key in row:
                if key not in self.fieldnames:
                    self.fieldnames.append(key)
        write_header = not self.csv_path.exists()
        with self.csv_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({key: row.get(key, "") for key in self.fieldnames})
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def prepare_datasets(config: dict[str, Any]) -> tuple[PressPairsDataset, PressPairsDataset, Path]:
    root = extraction_root(config)
    split_cfg = config.get("splits", {})
    if not (root / "splits" / "split_index.csv").exists():
        build_splits(
            root,
            val_fraction=float(split_cfg.get("val_fraction", 0.1)),
            seed=int(split_cfg.get("seed", 42)),
            min_pairs_per_split=int(split_cfg.get("min_pairs_per_split", 1000)),
        )
    norm_path = root / "splits" / "norm_stats_fingerpred.npz"
    if not norm_path.exists():
        norm_path = compute_fingerpred_norm_stats(root)
    train_dataset = PressPairsDataset(root, split="train", norm_stats_path=norm_path, assert_unique_goals=True, output_mode="fingerpred")
    val_dataset = PressPairsDataset(root, split="val", norm_stats_path=norm_path, assert_unique_goals=True, output_mode="fingerpred")
    return train_dataset, val_dataset, norm_path


def mean_dict(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row})
    return {key: float(np.mean([row[key] for row in rows if key in row])) for key in keys}


def autoencoder_loss(recon: torch.Tensor, target: torch.Tensor, coord_mask: torch.Tensor, z: torch.Tensor, cfg: dict[str, Any]) -> dict[str, torch.Tensor]:
    active_mse = masked_mse(recon, target, coord_mask)
    latent_l2 = z.pow(2).mean()
    total = float(cfg.get("fingertips_weight", 1.0)) * active_mse + float(cfg.get("latent_l2_weight", 1e-4)) * latent_l2
    return {"loss": total, "active_fingertip_mse": active_mse.detach(), "latent_l2": latent_l2.detach()}


def run_autoencoder_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
    normalizer: FingerPredNormalizer,
    device: torch.device,
    loss_cfg: dict[str, Any],
    grad_clip_norm: float | None,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    rows = []
    for batch in tqdm(loader, desc="train fingerpred ae" if train else "val fingerpred ae", leave=False):
        target = batch["target_state_normalized"].to(device)
        coord_mask = batch["target_coord_mask"].to(device)
        model_input = masked_autoencoder_input(target, coord_mask)
        if train:
            optimizer.zero_grad(set_to_none=True)
        recon, z = model(model_input)
        losses = autoencoder_loss(recon, target, coord_mask, z, loss_cfg)
        if train:
            losses["loss"].backward()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
        row = {key: float(value.detach().cpu().item()) for key, value in losses.items()}
        with torch.no_grad():
            row.update(active_fingertip_metrics(
                normalizer.denormalize(recon.detach()),
                batch["target_state"].to(device),
                batch["active_tip_mask"].to(device),
                pred_norm=recon.detach(),
                target_norm=target,
                coord_mask=coord_mask,
            ))
        rows.append(row)
    return mean_dict(rows)


@torch.no_grad()
def compute_latent_stats(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, torch.Tensor]:
    model.eval()
    chunks = []
    for batch in tqdm(loader, desc="fingerpred latent stats", leave=False):
        target = batch["target_state_normalized"].to(device)
        coord_mask = batch["target_coord_mask"].to(device)
        chunks.append(model.encode(masked_autoencoder_input(target, coord_mask)).detach().cpu())
    z = torch.cat(chunks, dim=0)
    return {"z_mean": z.mean(dim=0), "z_std": z.std(dim=0).clamp(min=1e-6)}


def standardized_decoder(autoencoder, z_mean: torch.Tensor, z_std: torch.Tensor):
    def decode(z_standardized: torch.Tensor) -> torch.Tensor:
        return autoencoder.decode(z_standardized * z_std.to(z_standardized.device, z_standardized.dtype) + z_mean.to(z_standardized.device, z_standardized.dtype))

    return decode


def run_mdn_epoch(
    mdn_model: torch.nn.Module,
    autoencoder: torch.nn.Module,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
    normalizer: FingerPredNormalizer,
    latent_stats: dict[str, torch.Tensor],
    device: torch.device,
    loss_cfg: dict[str, Any],
    grad_clip_norm: float | None,
) -> dict[str, float]:
    train = optimizer is not None
    mdn_model.train(train)
    autoencoder.eval()
    rows = []
    z_mean = latent_stats["z_mean"].to(device)
    z_std = latent_stats["z_std"].to(device)
    decoder = standardized_decoder(autoencoder, z_mean, z_std)
    for batch in tqdm(loader, desc="train fingerpred mdn" if train else "val fingerpred mdn", leave=False):
        target_keys = batch["target_keys"].to(device)
        target = batch["target_state_normalized"].to(device)
        coord_mask = batch["target_coord_mask"].to(device)
        with torch.no_grad():
            z = autoencoder.encode(masked_autoencoder_input(target, coord_mask))
            z_true = (z - z_mean) / z_std
        if train:
            optimizer.zero_grad(set_to_none=True)
        mixture_logits, means, log_stds = mdn_model(target_keys)
        nll = mdn_nll_loss(mixture_logits, means, log_stds, z_true)
        recon = decoder(select_best_component_mean(mixture_logits, means))
        active_mse = masked_mse(recon, target, coord_mask)
        loss = float(loss_cfg.get("mdn_nll_weight", 1.0)) * nll + float(loss_cfg.get("fingertip_aux_weight", 0.5)) * active_mse
        if train:
            loss.backward()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(mdn_model.parameters(), grad_clip_norm)
            optimizer.step()
        row = {"loss": float(loss.detach().cpu().item()), "nll": float(nll.detach().cpu().item()), "active_fingertip_mse": float(active_mse.detach().cpu().item())}
        with torch.no_grad():
            row.update(active_fingertip_metrics(
                normalizer.denormalize(recon.detach()),
                batch["target_state"].to(device),
                batch["active_tip_mask"].to(device),
                pred_norm=recon.detach(),
                target_norm=target,
                coord_mask=coord_mask,
            ))
        row["val_score"] = row["nll"] + float(loss_cfg.get("fingertip_aux_weight", 0.5)) * row["active_fingertip_mse"]
        rows.append(row)
    return mean_dict(rows)


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    torch.save(payload, path)


def maybe_start_wandb(config: dict[str, Any], run_root: Path):
    if not bool(config.get("logging", {}).get("use_wandb", False)):
        return None
    try:
        import wandb
    except ImportError:
        print("wandb requested but not installed; continuing without wandb.")
        return None
    try:
        return wandb.init(
            project=str(config.get("logging", {}).get("project", "variations")),
            name=str(config.get("logging", {}).get("run_name", run_root.name)),
            tags=list(config.get("logging", {}).get("tags", ["variations", "fingerpred"])),
            job_type="fingerpred",
            config=config,
            dir=str(run_root),
        )
    except Exception as exc:
        print(f"wandb init failed; continuing without wandb: {exc}")
        return None


def train_autoencoder(config: dict[str, Any], train_loader: DataLoader, val_loader: DataLoader, normalizer: FingerPredNormalizer, run_root: Path, wandb_run, device: torch.device) -> tuple[torch.nn.Module, Path]:
    model = build_pose_autoencoder(config).to(device)
    train_cfg = config["autoencoder"]["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["learning_rate"]), weight_decay=float(train_cfg.get("weight_decay", 1e-4)))
    writer = MetricsWriter(run_root / "autoencoder" / "metrics")
    checkpoints = ensure_dir(run_root / "autoencoder" / "checkpoints")
    best_path = checkpoints / "best.pt"
    best_loss = float("inf")
    bad_epochs = 0
    patience = int(train_cfg.get("early_stopping_patience", 30))
    grad_clip = train_cfg.get("grad_clip_norm")
    grad_clip = float(grad_clip) if grad_clip is not None else None
    for epoch in range(int(train_cfg.get("epochs", 120))):
        train_metrics = run_autoencoder_epoch(model, train_loader, optimizer=optimizer, normalizer=normalizer, device=device, loss_cfg=config["autoencoder"].get("loss", {}), grad_clip_norm=grad_clip)
        with torch.no_grad():
            val_metrics = run_autoencoder_epoch(model, val_loader, optimizer=None, normalizer=normalizer, device=device, loss_cfg=config["autoencoder"].get("loss", {}), grad_clip_norm=None)
        row = {"epoch": epoch, **{f"ae/train/{k}": v for k, v in train_metrics.items()}, **{f"ae/val/{k}": v for k, v in val_metrics.items()}}
        writer.log(row)
        if wandb_run is not None:
            wandb_run.log(row, step=epoch)
        print(row)
        val_loss = float(val_metrics["loss"])
        payload = {
            "model_state_dict": model.state_dict(),
            "model_type": "fingerpred",
            "output_mode": FINGERPRED_OUTPUT_MODE,
            "target_dim": FINGERPRED_TARGET_DIM,
            "latent_dim": int(config["autoencoder"].get("latent_dim", 16)),
            "input_dim": int(config["autoencoder"].get("input_dim", FINGERPRED_TARGET_DIM)),
            "normalizer": normalizer.state_dict(),
            "config": config,
            "best_val_loss": min(best_loss, val_loss),
            "active_tip_policy": "nearest_key_site",
        }
        save_checkpoint(checkpoints / "last.pt", payload)
        if val_loss < best_loss:
            best_loss = val_loss
            bad_epochs = 0
            save_checkpoint(best_path, payload)
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            print(f"FingerPred autoencoder early stopping after {bad_epochs} non-improving epochs.")
            break
    payload = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, best_path


def train_mdn(config: dict[str, Any], autoencoder: torch.nn.Module, autoencoder_path: Path, latent_stats: dict[str, torch.Tensor], train_loader: DataLoader, val_loader: DataLoader, normalizer: FingerPredNormalizer, run_root: Path, wandb_run, device: torch.device) -> Path:
    for parameter in autoencoder.parameters():
        parameter.requires_grad = False
    model = build_latent_mdn(config).to(device)
    train_cfg = config["mdn"]["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["learning_rate"]), weight_decay=float(train_cfg.get("weight_decay", 1e-4)))
    writer = MetricsWriter(run_root / "mdn" / "metrics")
    checkpoints = ensure_dir(run_root / "mdn" / "checkpoints")
    best_path = checkpoints / "best.pt"
    best_score = float("inf")
    bad_epochs = 0
    patience = int(train_cfg.get("early_stopping_patience", 30))
    grad_clip = train_cfg.get("grad_clip_norm")
    grad_clip = float(grad_clip) if grad_clip is not None else None
    for epoch in range(int(train_cfg.get("epochs", 120))):
        train_metrics = run_mdn_epoch(model, autoencoder, train_loader, optimizer=optimizer, normalizer=normalizer, latent_stats=latent_stats, device=device, loss_cfg=config["mdn"].get("loss", {}), grad_clip_norm=grad_clip)
        with torch.no_grad():
            val_metrics = run_mdn_epoch(model, autoencoder, val_loader, optimizer=None, normalizer=normalizer, latent_stats=latent_stats, device=device, loss_cfg=config["mdn"].get("loss", {}), grad_clip_norm=None)
        row = {"epoch": epoch, **{f"mdn/train/{k}": v for k, v in train_metrics.items()}, **{f"mdn/val/{k}": v for k, v in val_metrics.items()}}
        writer.log(row)
        if wandb_run is not None:
            wandb_run.log(row, step=epoch)
        print(row)
        val_score = float(val_metrics["val_score"])
        payload = {
            "model_state_dict": model.state_dict(),
            "model_type": "fingerpred",
            "output_mode": FINGERPRED_OUTPUT_MODE,
            "target_dim": FINGERPRED_TARGET_DIM,
            "latent_dim": int(config["mdn"].get("latent_dim", 16)),
            "num_components": int(config["mdn"].get("num_components", 3)),
            "latent_stats": latent_stats,
            "autoencoder_checkpoint": str(autoencoder_path),
            "normalizer": normalizer.state_dict(),
            "config": config,
            "best_val_score": min(best_score, val_score),
            "active_tip_policy": "nearest_key_site",
        }
        save_checkpoint(checkpoints / "last.pt", payload)
        if val_score < best_score:
            best_score = val_score
            bad_epochs = 0
            save_checkpoint(best_path, payload)
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            print(f"FingerPred MDN early stopping after {bad_epochs} non-improving epochs.")
            break
    return best_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Variations FingerPred active-fingertip model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--extraction-root", default=None, help="Override extraction_root from YAML.")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None, help="Optional smoke-test epoch cap for both stages.")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.extraction_root:
        config["extraction_root"] = args.extraction_root
    if args.no_wandb:
        config.setdefault("logging", {})["use_wandb"] = False
    if args.max_epochs is not None:
        config.setdefault("autoencoder", {}).setdefault("training", {})["epochs"] = int(args.max_epochs)
        config.setdefault("mdn", {}).setdefault("training", {})["epochs"] = int(args.max_epochs)
    set_seed(int(config.get("seed", 42)))
    device = resolve_device(str(config.get("autoencoder", {}).get("training", {}).get("device", "auto")))
    train_dataset, val_dataset, norm_path = prepare_datasets(config)
    if args.max_train_samples is not None:
        train_dataset = Subset(train_dataset, range(min(args.max_train_samples, len(train_dataset))))  # type: ignore[assignment]
    if args.max_val_samples is not None:
        val_dataset = Subset(val_dataset, range(min(args.max_val_samples, len(val_dataset))))  # type: ignore[assignment]
    normalizer = FingerPredNormalizer.from_npz(norm_path, device=device)
    run_root = output_root(config)
    ensure_dir(run_root)
    save_config(run_root / "config.yaml", config)
    save_json(run_root / "dataset_summary.json", {
        "extraction_root": str(extraction_root(config)),
        "norm_stats": str(norm_path),
        "train_pairs": len(train_dataset),
        "val_pairs": len(val_dataset),
        "output_mode": FINGERPRED_OUTPUT_MODE,
        "target_dim": FINGERPRED_TARGET_DIM,
        "active_tip_policy": "nearest_key_site",
    })
    wandb_run = maybe_start_wandb(config, run_root)
    if wandb_run is not None:
        wandb_run.summary.update({"run_root": str(run_root), "data/train_pairs": len(train_dataset), "data/val_pairs": len(val_dataset)})
    num_workers = int(config.get("data", {}).get("num_workers", 0))
    ae_batch = int(config["autoencoder"]["training"].get("batch_size", 256))
    train_loader_ae = DataLoader(train_dataset, batch_size=ae_batch, shuffle=True, num_workers=num_workers)
    val_loader_ae = DataLoader(val_dataset, batch_size=ae_batch, shuffle=False, num_workers=num_workers)
    autoencoder, autoencoder_path = train_autoencoder(config, train_loader_ae, val_loader_ae, normalizer, run_root, wandb_run, device)
    latent_stats = compute_latent_stats(autoencoder, DataLoader(train_dataset, batch_size=ae_batch, shuffle=False, num_workers=num_workers), device)
    torch.save(latent_stats, run_root / "autoencoder" / "latent_stats.pt")
    mdn_batch = int(config["mdn"]["training"].get("batch_size", 256))
    train_loader_mdn = DataLoader(train_dataset, batch_size=mdn_batch, shuffle=True, num_workers=num_workers)
    val_loader_mdn = DataLoader(val_dataset, batch_size=mdn_batch, shuffle=False, num_workers=num_workers)
    mdn_path = train_mdn(config, autoencoder, autoencoder_path, latent_stats, train_loader_mdn, val_loader_mdn, normalizer, run_root, wandb_run, device)
    if wandb_run is not None:
        wandb_run.summary.update({"autoencoder/best_checkpoint": str(autoencoder_path), "mdn/best_checkpoint": str(mdn_path), "status": "completed"})
        wandb_run.finish()
    print(f"FingerPred autoencoder checkpoint: {autoencoder_path}")
    print(f"FingerPred MDN checkpoint: {mdn_path}")


if __name__ == "__main__":
    main()
