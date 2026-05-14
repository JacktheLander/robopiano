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

from variations.data.dataset import PressPairsDataset, build_splits, compute_fingerpred_norm_stats
from variations.fingerpred import active_fingertip_metrics, load_fingerpred_checkpoint, predict_with_fingerpred
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
    norm_path = root / "splits" / "norm_stats_fingerpred.npz"
    if not norm_path.exists():
        norm_path = compute_fingerpred_norm_stats(root)
    return PressPairsDataset(root, split=split, norm_stats_path=norm_path, assert_unique_goals=True, output_mode="fingerpred"), norm_path


def output_paths(config: dict[str, Any], output: str | None) -> tuple[Path, Path]:
    import os

    if output:
        csv_path = Path(output)
    else:
        root = Path(config.get("logging", {}).get("output_root", "Variations/outputs/fingerpred"))
        if os.environ.get("VARIATIONS_OUTPUT_ROOT") and not root.is_absolute() and root.parts[:2] == ("Variations", "outputs"):
            root = Path(os.environ["VARIATIONS_OUTPUT_ROOT"]).joinpath(*root.parts[2:])
        elif not root.is_absolute():
            root = Path(config.get("_repo_root", Path.cwd())) / root
        csv_path = root / "comparisons" / str(config.get("logging", {}).get("filename", "fingerpred_metrics.csv"))
    return csv_path, csv_path.with_suffix(".json")


@torch.no_grad()
def evaluate_fingerpred(
    checkpoint: Path,
    dataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    mdn_model, autoencoder, latent_stats, normalizer, _config, _payload = load_fingerpred_checkpoint(checkpoint, device=device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    pred_chunks = []
    pred_norm_chunks = []
    true_chunks = []
    true_norm_chunks = []
    tip_mask_chunks = []
    coord_mask_chunks = []
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for batch in tqdm(loader, desc="eval fingerpred", leave=False):
        target_keys = batch["target_keys"].to(device)
        pred = predict_with_fingerpred(target_keys, mdn_model, autoencoder, latent_stats, normalizer)
        pred_norm = normalizer.normalize(pred)
        pred_chunks.append(pred.cpu())
        pred_norm_chunks.append(pred_norm.cpu())
        true_chunks.append(batch["target_state"])
        true_norm_chunks.append(batch["target_state_normalized"])
        tip_mask_chunks.append(batch["active_tip_mask"])
        coord_mask_chunks.append(batch["target_coord_mask"])
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    pred = torch.cat(pred_chunks, dim=0)
    pred_norm = torch.cat(pred_norm_chunks, dim=0)
    true = torch.cat(true_chunks, dim=0)
    true_norm = torch.cat(true_norm_chunks, dim=0)
    tip_mask = torch.cat(tip_mask_chunks, dim=0)
    coord_mask = torch.cat(coord_mask_chunks, dim=0)
    metrics = active_fingertip_metrics(pred, true, tip_mask, pred_norm=pred_norm, target_norm=true_norm, coord_mask=coord_mask)
    metrics.update(
        {
            "model_name": "fingerpred",
            "checkpoint": str(checkpoint),
            "test_loss": metrics["active_fingertip_normalized_mse"],
            "num_parameters": count_parameters(mdn_model) + count_parameters(autoencoder),
            "inference_time_ms_per_sample": elapsed * 1000.0 / max(len(dataset), 1),
        }
    )
    return metrics


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    base_fieldnames = [
        "model_name",
        "checkpoint",
        "test_loss",
        "active_fingertip_normalized_mse",
        "active_fingertip_mse",
        "active_fingertip_rmse",
        "active_tip_distance_mean",
        "active_tip_distance_median",
        "active_tip_distance_p95",
        "active_tip_count",
        "active_tip_examples",
        "num_parameters",
        "inference_time_ms_per_sample",
    ]
    extra = sorted({key for row in rows for key in row.keys() if key not in base_fieldnames})
    fieldnames = base_fieldnames + extra
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Variations FingerPred checkpoint.")
    parser.add_argument("--config", default="Variations/configs/fingerpred.yaml")
    parser.add_argument("--extraction-root", default=None, help="Override extraction_root from YAML.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
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
    rows = [evaluate_fingerpred(Path(args.checkpoint), dataset, device, batch_size, num_workers)]
    csv_path, json_path = output_paths(config, args.output)
    write_csv(csv_path, rows)
    save_json(json_path, rows)
    for row in rows:
        print(row)
    print(f"Saved FingerPred metrics CSV: {csv_path}")
    print(f"Saved FingerPred metrics JSON: {json_path}")


if __name__ == "__main__":
    main()
