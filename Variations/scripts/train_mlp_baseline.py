from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

VARIATIONS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = VARIATIONS_ROOT.parent
SRC_ROOT = VARIATIONS_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
for _path in (VARIATIONS_ROOT, REPO_ROOT):
    _text = str(_path)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from variations.data.dataset import (
    EXTRACTED_HAND_STATE_DIM,
    FINGERTIP_STATE_DIM,
    JOINT_STATE_DIM,
    PressPairsDataset,
    build_splits,
    compute_fingerpred_norm_stats,
    compute_joint_fingertip_norm_stats,
    compute_norm_stats,
    norm_stats_path_for_mode,
)
from variations.data.fingerpred import canonical_piano_key_positions
from variations.inference.predict_press_pose import HandStateNormalizer
from variations.losses.supervised_pose_loss import supervised_pose_loss
from variations.models.mlp_baseline import build_mlp_baseline
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
    logging_cfg = config.get("logging", {})
    root = Path(logging_cfg.get("output_root", "Variations/outputs/mlp_baseline"))
    env_root = None
    import os

    if os.environ.get("VARIATIONS_MLP_OUTPUT_ROOT"):
        env_root = Path(os.environ["VARIATIONS_MLP_OUTPUT_ROOT"])
    elif os.environ.get("VARIATIONS_OUTPUT_ROOT") and not root.is_absolute() and root.parts[:2] == ("Variations", "outputs"):
        suffix = root.parts[2:]
        env_root = Path(os.environ["VARIATIONS_OUTPUT_ROOT"]).joinpath(*suffix)
    if env_root is not None:
        root = env_root
    elif not root.is_absolute():
        root = Path(config.get("_repo_root", Path.cwd())) / root
    return root / str(logging_cfg.get("run_name") or config.get("experiment_name", "mlp_baseline"))


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


def prepare_datasets(config: dict[str, Any]) -> tuple[PressPairsDataset, PressPairsDataset, Path]:
    root = extraction_root(config)
    split_cfg = config.get("splits", {})
    output_mode = str(config.get("data", {}).get("output_mode", "joints_only"))
    if not (root / "splits" / "split_index.csv").exists():
        build_splits(
            root,
            val_fraction=float(split_cfg.get("val_fraction", 0.1)),
            seed=int(split_cfg.get("seed", 42)),
            min_pairs_per_split=int(split_cfg.get("min_pairs_per_split", 1000)),
        )
    norm_path = norm_stats_path_for_mode(root, output_mode)
    if output_mode == "joints_with_fingertips":
        norm_path = compute_joint_fingertip_norm_stats(root)
        key_positions = canonical_piano_key_positions()
    elif output_mode == "fingerpred":
        key_positions = canonical_piano_key_positions()
        norm_path = compute_fingerpred_norm_stats(root, key_positions=key_positions)
    else:
        key_positions = None
        compute_norm_stats(root)
        norm_path = root / "splits" / "norm_stats.npz"
    train_dataset = PressPairsDataset(
        root,
        split="train",
        norm_stats_path=norm_path,
        assert_unique_goals=True,
        output_mode=output_mode,
        key_positions=key_positions,
    )
    val_dataset = PressPairsDataset(
        root,
        split="val",
        norm_stats_path=norm_path,
        assert_unique_goals=True,
        output_mode=output_mode,
        key_positions=key_positions,
    )
    return train_dataset, val_dataset, norm_path


def batch_target(batch: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    return batch["target_state_normalized"].to(device)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    typed_mask = mask.to(device=pred.device, dtype=pred.dtype)
    return (((pred - target) ** 2) * typed_mask).sum() / typed_mask.sum().clamp_min(1.0)


def compute_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    batch: dict[str, torch.Tensor],
    loss_cfg: dict[str, Any],
    *,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if pred.shape[1] == JOINT_STATE_DIM:
        return supervised_pose_loss(
            pred,
            target[:, :JOINT_STATE_DIM],
            joints_weight=float(loss_cfg.get("joints_weight", 1.0)),
        )
    if pred.shape[1] != EXTRACTED_HAND_STATE_DIM:
        raise ValueError(f"Expected MLP output width 46 or 76, got {pred.shape[1]}")

    joint_loss = torch.nn.functional.mse_loss(pred[:, :JOINT_STATE_DIM], target[:, :JOINT_STATE_DIM])
    fingertip_norm_loss = masked_mse(
        pred[:, JOINT_STATE_DIM:EXTRACTED_HAND_STATE_DIM],
        target[:, JOINT_STATE_DIM:EXTRACTED_HAND_STATE_DIM],
        batch["active_tip_mask"].repeat_interleave(3, dim=1).to(pred.device),
    )
    pred_denorm = pred * std.to(pred.device, pred.dtype) + mean.to(pred.device, pred.dtype)
    pred_tips = pred_denorm[:, JOINT_STATE_DIM:EXTRACTED_HAND_STATE_DIM]
    key_targets = batch["active_key_fingertip_targets"].to(pred.device, pred.dtype)
    coord_mask = batch["active_tip_mask"].repeat_interleave(3, dim=1).to(pred.device, pred.dtype)
    contact_loss = masked_mse(pred_tips, key_targets, coord_mask)
    loss = (
        float(loss_cfg.get("joints_weight", 1.0)) * joint_loss
        + float(loss_cfg.get("active_fingertip_weight", 1.0)) * fingertip_norm_loss
        + float(loss_cfg.get("target_key_contact_weight", 1.0)) * contact_loss
    )
    return {
        "loss": loss,
        "joint_loss": joint_loss,
        "active_fingertip_loss": fingertip_norm_loss,
        "target_key_contact_loss": contact_loss,
    }


def denormalized_metrics(
    pred_norm: torch.Tensor,
    true_joint: torch.Tensor,
    normalizer: HandStateNormalizer,
) -> dict[str, float]:
    pred_joint = normalizer.denormalize_hand_state(pred_norm[:, :JOINT_STATE_DIM])
    diff = pred_joint - true_joint
    return {
        "denormalized_pose_mse": float((diff * diff).mean().detach().cpu().item()),
        "joint_mse": float((diff * diff).mean().detach().cpu().item()),
    }


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    normalizer: HandStateNormalizer,
    loss_cfg: dict[str, Any],
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    grad_clip_norm: float | None = None,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    totals: dict[str, list[float]] = {}
    iterator = tqdm(loader, desc="train mlp" if train else "val mlp", leave=False)
    for batch in iterator:
        target_keys = batch["target_keys"].to(device)
        target = batch_target(batch, device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        pred = model(target_keys)
        loss_dict = compute_loss(pred, target, batch, loss_cfg, mean=target_mean, std=target_std)
        if train:
            loss_dict["loss"].backward()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
        for key, value in loss_dict.items():
            totals.setdefault(key, []).append(float(value.detach().cpu().item()))
        with torch.no_grad():
            denorm = denormalized_metrics(
                pred.detach(),
                batch["hand_state"].to(device),
                normalizer,
            )
        for key, value in denorm.items():
            if np.isfinite(value):
                totals.setdefault(key, []).append(float(value))
        iterator.set_postfix(loss=f"{totals['loss'][-1]:.4f}")
    return {key: float(np.mean(values)) if values else float("nan") for key, values in totals.items()}


def maybe_start_wandb(config: dict[str, Any], run_root: Path):
    logging_cfg = config.get("logging", {})
    if not bool(logging_cfg.get("use_wandb", False)):
        return None
    try:
        import wandb
    except ImportError:
        print("wandb requested but not installed; continuing without wandb.")
        return None
    try:
        return wandb.init(
            project=str(logging_cfg.get("project", "variations")),
            name=str(logging_cfg.get("run_name") or run_root.name),
            tags=list(logging_cfg.get("tags", ["variations", "mlp-baseline"])),
            job_type="mlp_baseline",
            config=config,
            dir=str(run_root),
        )
    except Exception as exc:
        print(f"wandb init failed; continuing without wandb: {exc}")
        return None


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    torch.save(payload, path)


def checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    config: dict[str, Any],
    normalizer: HandStateNormalizer,
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "config": config,
        "normalizer": normalizer.state_dict(),
        "model_type": config.get("model_type", "mlp_baseline"),
        "output_mode": str(config.get("data", {}).get("output_mode", "joints_only")),
    }


@torch.no_grad()
def inference_ms_per_sample(model: torch.nn.Module, dataset: PressPairsDataset, device: torch.device, sample_count: int = 1024) -> float:
    if len(dataset) == 0:
        return 0.0
    n = min(sample_count, len(dataset))
    target = torch.as_tensor(dataset.target_keys[:n], dtype=torch.float32, device=device)
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    _ = model(target)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return float((time.perf_counter() - start) * 1000.0 / max(n, 1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a supervised Variations MLP baseline.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional smoke-test cap.")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Optional smoke-test cap.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.no_wandb:
        config.setdefault("logging", {})["use_wandb"] = False
    set_seed(int(config.get("seed", 42)))
    device = resolve_device(str(config.get("training", {}).get("device", "auto")))
    train_dataset, val_dataset, norm_path = prepare_datasets(config)
    if args.max_train_samples is not None:
        train_dataset = Subset(train_dataset, range(min(args.max_train_samples, len(train_dataset))))
    if args.max_val_samples is not None:
        val_dataset = Subset(val_dataset, range(min(args.max_val_samples, len(val_dataset))))

    normalizer = HandStateNormalizer.from_npz(norm_path, device=device)
    norm_data = np.load(norm_path, allow_pickle=False)
    target_mean = torch.as_tensor(np.asarray(norm_data["mean"], dtype=np.float32), device=device)
    target_std = torch.as_tensor(np.asarray(norm_data["std"], dtype=np.float32), device=device)
    model = build_mlp_baseline(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("training", {}).get("learning_rate", 3e-4)),
        weight_decay=float(config.get("training", {}).get("weight_decay", 1e-4)),
    )
    data_cfg = config.get("data", {})
    train_loader = DataLoader(train_dataset, batch_size=int(data_cfg.get("batch_size", 256)), shuffle=True, num_workers=int(data_cfg.get("num_workers", 0)))
    val_loader = DataLoader(val_dataset, batch_size=int(data_cfg.get("batch_size", 256)), shuffle=False, num_workers=int(data_cfg.get("num_workers", 0)))

    run_root = output_root(config)
    checkpoints = ensure_dir(run_root / "checkpoints")
    metrics_writer = MetricsWriter(run_root / "metrics")
    artifacts = ensure_dir(run_root / "artifacts")
    save_config(artifacts / "config.yaml", config)
    save_json(artifacts / "dataset_summary.json", {
        "extraction_root": str(extraction_root(config)),
        "norm_stats": str(norm_path),
        "train_pairs": len(train_dataset),
        "val_pairs": len(val_dataset),
        "output_mode": "joints_only",
        "training_target_dim": int(target_mean.shape[0]),
    })

    wandb_run = maybe_start_wandb(config, run_root)
    num_params = count_parameters(model)
    if wandb_run is not None:
        wandb_run.summary.update({
            "model/num_parameters": num_params,
            "run_root": str(run_root),
            "data/train_pairs": len(train_dataset),
            "data/val_pairs": len(val_dataset),
            "normalizer": str(norm_path),
        })

    epochs = int(config.get("training", {}).get("epochs", 200))
    patience = int(config.get("training", {}).get("early_stopping_patience", 20))
    grad_clip = config.get("training", {}).get("grad_clip_norm")
    grad_clip = float(grad_clip) if grad_clip is not None else None
    best_val_loss = float("inf")
    bad_epochs = 0
    for epoch in range(epochs):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            device=device,
            normalizer=normalizer,
            loss_cfg=config.get("loss", {}),
            target_mean=target_mean,
            target_std=target_std,
            grad_clip_norm=grad_clip,
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model,
                val_loader,
                optimizer=None,
                device=device,
                normalizer=normalizer,
                loss_cfg=config.get("loss", {}),
                target_mean=target_mean,
                target_std=target_std,
            )
        speed = inference_ms_per_sample(model, val_dataset.dataset if isinstance(val_dataset, Subset) else val_dataset, device)
        row = {
            "epoch": epoch,
            "model/num_parameters": num_params,
            "model/inference_ms_per_sample": speed,
            **{f"train/{key}": value for key, value in train_metrics.items()},
            **{f"val/{key}": value for key, value in val_metrics.items()},
        }
        metrics_writer.log(row)
        if wandb_run is not None:
            wandb_run.log(row, step=epoch)
        print(row)

        val_loss = float(val_metrics["loss"])
        payload = checkpoint_payload(model=model, optimizer=optimizer, epoch=epoch, best_val_loss=min(best_val_loss, val_loss), config=config, normalizer=normalizer)
        save_checkpoint(checkpoints / "last.pt", payload)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            bad_epochs = 0
            save_checkpoint(checkpoints / "best.pt", payload)
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            print(f"Early stopping after {bad_epochs} non-improving epochs.")
            break

    save_json(run_root / "metrics" / "summary.json", {"best_val_loss": best_val_loss, "num_parameters": num_params})
    if wandb_run is not None:
        wandb_run.summary.update({
            "best/val_loss": best_val_loss,
            "best/checkpoint": str(checkpoints / "best.pt"),
            "model/inference_ms_per_sample": speed,
            "status": "completed",
        })
        wandb_run.finish()
    print(f"Run root: {run_root}")
    print(f"Best checkpoint: {checkpoints / 'best.pt'}")


if __name__ == "__main__":
    main()
