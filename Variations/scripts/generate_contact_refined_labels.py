#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any

import numpy as np
from tqdm import tqdm

VARIATIONS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = VARIATIONS_ROOT.parent
for path in (VARIATIONS_ROOT / "src", VARIATIONS_ROOT, REPO_ROOT / "Intermezzo" / "src", REPO_ROOT / "partita" / "src", REPO_ROOT):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from intermezzo.io import atomic_save_json  # noqa: E402
from variations.contact_refinement import ContactRefinementConfig, ContactRefiner  # noqa: E402
from variations.data.dataset import PressPairsDataset, build_splits, compute_norm_stats  # noqa: E402
from simulate.model_loader import load_simulation_model  # noqa: E402
from variations.utils.config import extraction_root, load_config  # noqa: E402


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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["index", "active_keys", "assigned_pairs", "initial_loss", "final_loss", "improvement", "success", "message"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate contact-refined 46-D joint labels from a trained Variations checkpoint.")
    parser.add_argument("--config", default="Variations/configs/eval_press_pose.yaml")
    parser.add_argument("--extraction-root", default=None)
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--model-type", required=True, choices=["mlp_baseline", "latent_mdn", "diffusion"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--diffusion-steps", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=40)
    parser.add_argument("--pose-reg-weight", type=float, default=0.03)
    parser.add_argument("--smooth-weight", type=float, default=0.01)
    parser.add_argument("--inactive-weight", type=float, default=0.02)
    parser.add_argument("--inactive-margin-m", type=float, default=0.018)
    parser.add_argument("--z-weight", type=float, default=0.25)
    parser.add_argument("--key-z-offset-m", type=float, default=0.0)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.extraction_root:
        config["extraction_root"] = args.extraction_root
    dataset, norm_path = prepare_dataset(config, args.split)
    count = len(dataset) if args.max_samples is None else min(int(args.max_samples), len(dataset))
    target_keys = dataset.target_keys[:count].astype(np.float32)
    model = load_simulation_model(
        args.checkpoint,
        args.model_type,
        device=str(args.device).strip(),
        diffusion_steps=args.diffusion_steps,
    )
    initial = model.predict_hand_states(target_keys, batch_size=int(args.batch_size)).astype(np.float32)
    refined = np.zeros_like(initial, dtype=np.float32)
    rows: list[dict[str, Any]] = []
    cfg = ContactRefinementConfig(
        max_iter=int(args.max_iter),
        pose_reg_weight=float(args.pose_reg_weight),
        smooth_weight=float(args.smooth_weight),
        inactive_weight=float(args.inactive_weight),
        inactive_margin_m=float(args.inactive_margin_m),
        z_weight=float(args.z_weight),
        key_z_offset_m=float(args.key_z_offset_m),
    )
    previous = None
    with ContactRefiner(output_dir=Path(args.output).with_suffix("").parent / "contact_refinement_env") as refiner:
        for idx in tqdm(range(count), desc=f"refine {args.model_type}"):
            result = refiner.refine_pose(initial[idx], target_keys[idx], previous_pose=previous, config=cfg)
            refined[idx] = result.refined_pose
            previous = refined[idx]
            rows.append(
                {
                    "index": idx,
                    "active_keys": result.active_keys,
                    "assigned_pairs": result.assigned_pairs,
                    "initial_loss": result.initial_loss,
                    "final_loss": result.final_loss,
                    "improvement": result.initial_loss - result.final_loss,
                    "success": result.success,
                    "message": result.message,
                }
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        target_keys=target_keys,
        initial_hand_state=initial,
        refined_hand_state=refined,
        source_model_type=np.asarray(args.model_type),
        source_checkpoint=np.asarray(str(Path(args.checkpoint).expanduser())),
        norm_stats_path=np.asarray(str(norm_path)),
    )
    write_csv(output.with_suffix(".csv"), rows)
    improvements = np.asarray([row["improvement"] for row in rows], dtype=np.float64)
    atomic_save_json(
        output.with_suffix(".json"),
        {
            "output": str(output),
            "model_type": args.model_type,
            "checkpoint": str(args.checkpoint),
            "split": args.split,
            "examples": int(count),
            "mean_initial_loss": float(np.mean([row["initial_loss"] for row in rows])) if rows else None,
            "mean_final_loss": float(np.mean([row["final_loss"] for row in rows])) if rows else None,
            "mean_improvement": float(improvements.mean()) if improvements.size else None,
            "success_fraction": float(np.mean([bool(row["success"]) for row in rows])) if rows else None,
        },
    )
    print(f"Wrote contact-refined labels: {output}")


if __name__ == "__main__":
    main()
