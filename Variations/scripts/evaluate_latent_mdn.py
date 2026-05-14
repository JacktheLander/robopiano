from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

VARIATIONS_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = VARIATIONS_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from variations.data.dataset import PressPairsDataset, build_splits, compute_norm_stats
from variations.diffusion.trainer import load_model_for_inference
from variations.evaluation.fingertips import fingertip_metrics, measure_fingertips_with_mujoco
from variations.inference.predict_press_pose import load_latent_mdn_checkpoint, load_mlp_baseline_checkpoint, predict_with_latent_mdn
from variations.utils.config import extraction_root, load_config
from variations.utils.io import ensure_dir, save_json


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def count_parameters(model: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def prepare_dataset(config: dict[str, Any], split: str) -> tuple[PressPairsDataset, Path]:
    root = extraction_root(config)
    split_cfg = config.get("splits", {})
    if not (root / "splits" / "split_index.csv").exists():
        build_splits(
            root,
            val_fraction=float(split_cfg.get("val_fraction", 0.1)),
            seed=int(split_cfg.get("seed", 42)),
            min_pairs_per_split=int(split_cfg.get("min_pairs_per_split", 1000)),
        )
    norm_path = root / "splits" / "norm_stats.npz"
    if not norm_path.exists():
        compute_norm_stats(root)
    return PressPairsDataset(root, split=split, norm_stats_path=norm_path, assert_unique_goals=True), norm_path


def output_paths(config: dict[str, Any], output: str | None) -> tuple[Path, Path]:
    import os

    if output:
        csv_path = Path(output)
    else:
        root = Path(config.get("logging", {}).get("output_root", "Variations/outputs/latent_mdn"))
        if os.environ.get("VARIATIONS_OUTPUT_ROOT") and not root.is_absolute() and root.parts[:2] == ("Variations", "outputs"):
            root = Path(os.environ["VARIATIONS_OUTPUT_ROOT"]).joinpath(*root.parts[2:])
        elif not root.is_absolute():
            root = Path(config.get("_repo_root", Path.cwd())) / root
        csv_path = root / "comparisons" / "latent_mdn_metrics.csv"
    return csv_path, csv_path.with_suffix(".json")


def aggregate(pred: torch.Tensor, true: torch.Tensor, pred_norm: torch.Tensor | None, true_norm: torch.Tensor | None) -> dict[str, float]:
    diff = pred - true
    metrics = {
        "denormalized_pose_mse": float((diff * diff).mean().item()),
        "joint_mse": float((diff * diff).mean().item()),
    }
    if pred_norm is not None and true_norm is not None:
        metrics["normalized_pose_mse"] = float(((pred_norm - true_norm) ** 2).mean().item())
    else:
        metrics["normalized_pose_mse"] = float("nan")
    return metrics


def add_exact_fingertip_metrics(
    metrics: dict[str, Any],
    *,
    pred_joints: torch.Tensor,
    target_fingertips: torch.Tensor,
    target_keys: torch.Tensor,
    max_examples: int,
    control_timestep: float,
    settle_steps: int,
    label: str,
) -> None:
    count = min(int(max_examples), int(pred_joints.shape[0]), int(target_fingertips.shape[0]), int(target_keys.shape[0]))
    if count <= 0:
        return
    measured, meta = measure_fingertips_with_mujoco(
        pred_joints[:count].detach().cpu().numpy(),
        target_keys=target_keys[:count].detach().cpu().numpy(),
        control_timestep=float(control_timestep),
        label=label,
        settle_steps=int(settle_steps),
    )
    metrics.update(fingertip_metrics(measured, target_fingertips[:count].detach().cpu().numpy()))
    metrics["fingertip_eval_examples"] = count
    metrics["fingertip_eval_settle_steps"] = int(settle_steps)
    metrics["fingertip_eval_environment"] = meta["environment_name"]


@torch.no_grad()
def evaluate_latent(
    checkpoint: Path,
    dataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    *,
    fingertip_eval_max_examples: int = 0,
    control_timestep: float = 0.05,
    fingertip_settle_steps: int = 0,
) -> dict[str, Any]:
    mdn_model, autoencoder, latent_stats, normalizer, _config, _payload = load_latent_mdn_checkpoint(checkpoint, device=device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    pred_chunks = []
    true_chunks = []
    pred_norm_chunks = []
    true_norm_chunks = []
    target_key_chunks = []
    fingertip_chunks = []
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for batch in tqdm(loader, desc="eval latent mdn", leave=False):
        target = batch["target_keys"].to(device)
        pred = predict_with_latent_mdn(target, mdn_model, autoencoder, latent_stats, normalizer)
        pred_norm = (pred - normalizer.mean.to(device)) / normalizer.std.to(device)
        pred_chunks.append(pred.cpu())
        pred_norm_chunks.append(pred_norm.cpu())
        true_chunks.append(batch["hand_state"])
        true_norm_chunks.append(batch["hand_state_normalized"])
        target_key_chunks.append(batch["target_keys"])
        fingertip_chunks.append(batch["fingertip_state"])
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    pred = torch.cat(pred_chunks, dim=0)
    true = torch.cat(true_chunks, dim=0)
    pred_norm = torch.cat(pred_norm_chunks, dim=0)
    true_norm = torch.cat(true_norm_chunks, dim=0)
    metrics = aggregate(pred, true, pred_norm, true_norm)
    if int(fingertip_eval_max_examples) > 0:
        add_exact_fingertip_metrics(
            metrics,
            pred_joints=pred,
            target_fingertips=torch.cat(fingertip_chunks, dim=0),
            target_keys=torch.cat(target_key_chunks, dim=0),
            max_examples=int(fingertip_eval_max_examples),
            control_timestep=float(control_timestep),
            settle_steps=int(fingertip_settle_steps),
            label="latent_mdn",
        )
    metrics.update(
        {
            "model_name": "latent_mdn_best_component",
            "checkpoint": str(checkpoint),
            "pose_mse": metrics["denormalized_pose_mse"],
            "num_parameters": count_parameters(mdn_model) + count_parameters(autoencoder),
            "inference_time_ms_per_sample": elapsed * 1000.0 / max(len(dataset), 1),
        }
    )
    return metrics


@torch.no_grad()
def evaluate_mlp(
    checkpoint: Path,
    dataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    *,
    fingertip_eval_max_examples: int = 0,
    control_timestep: float = 0.05,
    fingertip_settle_steps: int = 0,
) -> dict[str, Any]:
    model, normalizer, _config, _payload = load_mlp_baseline_checkpoint(checkpoint, device=device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    pred_chunks = []
    true_chunks = []
    pred_norm_chunks = []
    true_norm_chunks = []
    target_key_chunks = []
    fingertip_chunks = []
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for batch in tqdm(loader, desc="eval mlp", leave=False):
        target = batch["target_keys"].to(device)
        raw = model(target)
        pred = normalizer.denormalize_hand_state(raw)
        pred_norm = raw
        pred_chunks.append(pred.cpu())
        pred_norm_chunks.append(pred_norm.cpu())
        true_chunks.append(batch["hand_state"])
        true_norm_chunks.append(batch["hand_state_normalized"])
        target_key_chunks.append(batch["target_keys"])
        fingertip_chunks.append(batch["fingertip_state"])
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    pred = torch.cat(pred_chunks, dim=0)
    true = torch.cat(true_chunks, dim=0)
    pred_norm = torch.cat(pred_norm_chunks, dim=0)
    true_norm = torch.cat(true_norm_chunks, dim=0)
    metrics = aggregate(pred, true, pred_norm, true_norm)
    if int(fingertip_eval_max_examples) > 0:
        add_exact_fingertip_metrics(
            metrics,
            pred_joints=pred,
            target_fingertips=torch.cat(fingertip_chunks, dim=0),
            target_keys=torch.cat(target_key_chunks, dim=0),
            max_examples=int(fingertip_eval_max_examples),
            control_timestep=float(control_timestep),
            settle_steps=int(fingertip_settle_steps),
            label="mlp_baseline",
        )
    metrics.update(
        {
            "model_name": "mlp_baseline",
            "checkpoint": str(checkpoint),
            "pose_mse": metrics["denormalized_pose_mse"],
            "num_parameters": count_parameters(model),
            "inference_time_ms_per_sample": elapsed * 1000.0 / max(len(dataset), 1),
        }
    )
    return metrics


@torch.no_grad()
def evaluate_diffusion(
    checkpoint: Path,
    dataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    steps: int,
    *,
    best_of_k: int | None = None,
    fingertip_eval_max_examples: int = 0,
    control_timestep: float = 0.05,
    fingertip_settle_steps: int = 0,
) -> dict[str, Any]:
    """Same best-of-K rule as training ``validate`` / W&B ``val_x0_mse_best_of_k``."""
    model, diffusion, ckpt_config, mean, std = load_model_for_inference(checkpoint, device)
    k_cfg = ckpt_config.get("training", {}).get("best_of_k", 8)
    k = int(best_of_k) if best_of_k is not None else int(k_cfg)
    k = max(k, 1)
    mean_t = torch.as_tensor(mean, dtype=torch.float32, device=device)
    std_t = torch.as_tensor(std, dtype=torch.float32, device=device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    pred_chunks = []
    true_chunks = []
    pred_norm_chunks = []
    true_norm_chunks = []
    target_key_chunks = []
    fingertip_chunks = []
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for batch in tqdm(loader, desc=f"eval diffusion (best-of-{k})", leave=False):
        target_keys = batch["target_keys"].to(device)
        truth = batch["hand_state"].to(device)
        samples_norm = []
        for _ in range(k):
            sample_norm = diffusion.p_sample_loop(
                model,
                (target_keys.shape[0], mean.shape[0]),
                target_keys,
                num_inference_steps=steps,
            )
            samples_norm.append(sample_norm)
        stacked_norm = torch.stack(samples_norm, dim=1)
        pred_denorm = stacked_norm * std_t.view(1, 1, -1) + mean_t.view(1, 1, -1)
        mse = ((pred_denorm - truth.unsqueeze(1)) ** 2).mean(dim=2)
        best_idx = mse.argmin(dim=1)
        b_idx = torch.arange(pred_denorm.shape[0], device=device)
        pred = pred_denorm[b_idx, best_idx]
        pred_norm = stacked_norm[b_idx, best_idx]
        pred_chunks.append(pred.cpu())
        pred_norm_chunks.append(pred_norm.cpu())
        true_chunks.append(batch["hand_state"])
        true_norm_chunks.append(batch["hand_state_normalized"])
        target_key_chunks.append(batch["target_keys"])
        fingertip_chunks.append(batch["fingertip_state"])
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    pred = torch.cat(pred_chunks, dim=0)
    true = torch.cat(true_chunks, dim=0)
    pred_norm = torch.cat(pred_norm_chunks, dim=0)
    true_norm = torch.cat(true_norm_chunks, dim=0)
    metrics = aggregate(pred, true, pred_norm, true_norm)
    if int(fingertip_eval_max_examples) > 0:
        add_exact_fingertip_metrics(
            metrics,
            pred_joints=pred,
            target_fingertips=torch.cat(fingertip_chunks, dim=0),
            target_keys=torch.cat(target_key_chunks, dim=0),
            max_examples=int(fingertip_eval_max_examples),
            control_timestep=float(control_timestep),
            settle_steps=int(fingertip_settle_steps),
            label="diffusion",
        )
    metrics.update(
        {
            "model_name": "diffusion_variations",
            "checkpoint": str(checkpoint),
            "pose_mse": metrics["denormalized_pose_mse"],
            "num_parameters": count_parameters(model),
            "inference_time_ms_per_sample": elapsed * 1000.0 / max(len(dataset), 1),
            "diffusion_best_of_k": k,
            "diffusion_steps": steps,
        }
    )
    return metrics


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    base_fieldnames = [
        "model_name",
        "checkpoint",
        "pose_mse",
        "normalized_pose_mse",
        "denormalized_pose_mse",
        "joint_mse",
        "num_parameters",
        "inference_time_ms_per_sample",
        "diffusion_best_of_k",
        "diffusion_steps",
    ]
    extra = sorted({key for row in rows for key in row.keys() if key not in base_fieldnames})
    fieldnames = base_fieldnames + extra
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate latent MDN against MLP and diffusion baselines.")
    parser.add_argument("--config", default="Variations/configs/eval_press_pose.yaml")
    parser.add_argument(
        "--extraction-root",
        default=None,
        help="Override extraction_root from YAML (must contain manifest.csv and splits/).",
    )
    parser.add_argument("--latent-mdn-checkpoint", required=True)
    parser.add_argument("--mlp-checkpoint", default=None)
    parser.add_argument("--diffusion-checkpoint", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--fingertip-eval-max-samples", type=int, default=None)
    parser.add_argument("--fingertip-settle-steps", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.extraction_root:
        config["extraction_root"] = args.extraction_root
    device = resolve_device(str(config.get("evaluation", {}).get("device", "auto")))
    dataset, _norm_path = prepare_dataset(config, split=str(config.get("evaluation", {}).get("split", "val")))
    max_samples = args.max_samples if args.max_samples is not None else config.get("evaluation", {}).get("max_samples")
    if max_samples is not None:
        dataset = Subset(dataset, range(min(int(max_samples), len(dataset))))  # type: ignore[assignment]
    batch_size = int(config.get("data", {}).get("batch_size", 512))
    num_workers = int(config.get("data", {}).get("num_workers", 0))
    fingertip_cfg = config.get("fingertip_eval", {})
    fingertip_eval_max = (
        args.fingertip_eval_max_samples
        if args.fingertip_eval_max_samples is not None
        else int(fingertip_cfg.get("max_samples", 0))
    )
    fingertip_settle_steps = (
        args.fingertip_settle_steps
        if args.fingertip_settle_steps is not None
        else int(fingertip_cfg.get("settle_steps", 0))
    )
    control_timestep = float(config.get("evaluation", {}).get("control_timestep", 0.05))
    rows = [
        evaluate_latent(
            Path(args.latent_mdn_checkpoint),
            dataset,
            device,
            batch_size,
            num_workers,
            fingertip_eval_max_examples=int(fingertip_eval_max),
            control_timestep=control_timestep,
            fingertip_settle_steps=int(fingertip_settle_steps),
        )
    ]
    if args.mlp_checkpoint:
        rows.append(
            evaluate_mlp(
                Path(args.mlp_checkpoint),
                dataset,
                device,
                batch_size,
                num_workers,
                fingertip_eval_max_examples=int(fingertip_eval_max),
                control_timestep=control_timestep,
                fingertip_settle_steps=int(fingertip_settle_steps),
            )
        )
    eval_cfg = config.get("evaluation", {})
    diffusion_k = eval_cfg.get("diffusion_best_of_k")
    diffusion_k_arg = int(diffusion_k) if diffusion_k is not None else None
    if args.diffusion_checkpoint:
        rows.append(
            evaluate_diffusion(
                Path(args.diffusion_checkpoint),
                dataset,
                device,
                batch_size,
                num_workers,
                int(eval_cfg.get("diffusion_steps", 50)),
                best_of_k=diffusion_k_arg,
                fingertip_eval_max_examples=int(fingertip_eval_max),
                control_timestep=control_timestep,
                fingertip_settle_steps=int(fingertip_settle_steps),
            )
        )
    csv_path, json_path = output_paths(config, args.output)
    write_csv(csv_path, rows)
    save_json(json_path, rows)
    for row in rows:
        print(row)
    print(f"Saved latent MDN comparison CSV: {csv_path}")
    print(f"Saved latent MDN comparison JSON: {json_path}")


if __name__ == "__main__":
    main()
