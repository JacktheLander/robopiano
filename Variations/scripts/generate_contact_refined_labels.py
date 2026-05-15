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
    fieldnames = [
        "index",
        "active_keys",
        "assigned_pairs",
        "assignment_mode",
        "crossing_penalty_applied",
        "hand_side_penalty_applied",
        "assigned_tip_indices",
        "assigned_key_indices",
        "initial_loss",
        "final_loss",
        "improvement",
        "initial_mean_error_m",
        "final_mean_error_m",
        "initial_median_error_m",
        "final_median_error_m",
        "initial_max_error_m",
        "final_max_error_m",
        "initial_p95_error_m",
        "final_p95_error_m",
        "initial_mean_xy_error_m",
        "final_mean_xy_error_m",
        "initial_mean_z_error_m",
        "final_mean_z_error_m",
        "initial_within_2mm_rate",
        "final_within_2mm_rate",
        "initial_within_5mm_rate",
        "final_within_5mm_rate",
        "initial_within_10mm_rate",
        "final_within_10mm_rate",
        "initial_wrong_key_nearest_count",
        "final_wrong_key_nearest_count",
        "initial_mean_center_error_m",
        "final_mean_center_error_m",
        "initial_mean_surface_error_m",
        "final_mean_surface_error_m",
        "success",
        "message",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def metric_row(prefix: str, metrics: Any) -> dict[str, Any]:
    return {
        f"{prefix}_mean_error_m": metrics.mean_error_m,
        f"{prefix}_median_error_m": metrics.median_error_m,
        f"{prefix}_max_error_m": metrics.max_error_m,
        f"{prefix}_p95_error_m": metrics.p95_error_m,
        f"{prefix}_mean_xy_error_m": metrics.mean_xy_error_m,
        f"{prefix}_mean_z_error_m": metrics.mean_z_error_m,
        f"{prefix}_within_2mm_rate": metrics.within_2mm_rate,
        f"{prefix}_within_5mm_rate": metrics.within_5mm_rate,
        f"{prefix}_within_10mm_rate": metrics.within_10mm_rate,
        f"{prefix}_wrong_key_nearest_count": metrics.wrong_key_nearest_count,
        f"{prefix}_mean_center_error_m": metrics.mean_center_error_m,
        f"{prefix}_mean_surface_error_m": metrics.mean_surface_error_m,
    }


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
    parser.add_argument("--smooth-weight", type=float, default=0.0)
    parser.add_argument("--use-previous-pose-smoothing", action="store_true")
    parser.add_argument("--inactive-weight", type=float, default=0.02)
    parser.add_argument("--inactive-margin-m", type=float, default=0.018)
    parser.add_argument("--z-weight", type=float, default=0.25)
    parser.add_argument("--key-z-offset-m", type=float, default=0.0)
    parser.add_argument("--xy-tolerance-m", type=float, default=0.003)
    parser.add_argument("--z-tolerance-m", type=float, default=0.002)
    parser.add_argument("--worst-contact-weight", type=float, default=0.35)
    parser.add_argument("--mean-contact-weight", type=float, default=0.65)
    parser.add_argument("--disable-two-stage-refinement", action="store_true")
    parser.add_argument("--micro-max-iter", type=int, default=40)
    parser.add_argument("--micro-pose-reg-weight", type=float, default=0.003)
    parser.add_argument("--micro-z-weight", type=float, default=1.0)
    parser.add_argument("--assignment-mode", choices=["nearest", "ordered", "hand_constrained"], default="ordered")
    parser.add_argument("--crossing-penalty", type=float, default=0.05)
    parser.add_argument("--hand-side-penalty", type=float, default=0.03)
    parser.add_argument("--target-mode", choices=["key_center", "key_surface_box"], default="key_center")
    parser.add_argument("--white-key-half-width-m", type=float, default=0.010)
    parser.add_argument("--black-key-half-width-m", type=float, default=0.006)
    parser.add_argument("--key-front-back-tolerance-m", type=float, default=0.018)
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
        xy_tolerance_m=float(args.xy_tolerance_m),
        z_tolerance_m=float(args.z_tolerance_m),
        worst_contact_weight=float(args.worst_contact_weight),
        mean_contact_weight=float(args.mean_contact_weight),
        two_stage_refinement=not bool(args.disable_two_stage_refinement),
        micro_max_iter=int(args.micro_max_iter),
        micro_pose_reg_weight=float(args.micro_pose_reg_weight),
        micro_z_weight=float(args.micro_z_weight),
        assignment_mode=str(args.assignment_mode),
        crossing_penalty=float(args.crossing_penalty),
        hand_side_penalty=float(args.hand_side_penalty),
        target_mode=str(args.target_mode),
        white_key_half_width_m=float(args.white_key_half_width_m),
        black_key_half_width_m=float(args.black_key_half_width_m),
        key_front_back_tolerance_m=float(args.key_front_back_tolerance_m),
    )
    previous = None
    with ContactRefiner(output_dir=Path(args.output).with_suffix("").parent / "contact_refinement_env") as refiner:
        for idx in tqdm(range(count), desc=f"refine {args.model_type}"):
            previous_arg = previous if args.use_previous_pose_smoothing else None
            result = refiner.refine_pose(initial[idx], target_keys[idx], previous_pose=previous_arg, config=cfg)
            refined[idx] = result.refined_pose
            previous = refined[idx]
            row = {
                "index": idx,
                "active_keys": result.active_keys,
                "assigned_pairs": result.assigned_pairs,
                "assignment_mode": result.assignment_mode,
                "crossing_penalty_applied": result.crossing_penalty_applied,
                "hand_side_penalty_applied": result.hand_side_penalty_applied,
                "assigned_tip_indices": " ".join(str(value) for value in result.assigned_tip_indices),
                "assigned_key_indices": " ".join(str(value) for value in result.assigned_key_indices),
                "initial_loss": result.initial_loss,
                "final_loss": result.final_loss,
                "improvement": result.initial_loss - result.final_loss,
                "success": result.success,
                "message": result.message,
            }
            row.update(metric_row("initial", result.initial_metrics))
            row.update(metric_row("final", result.final_metrics))
            rows.append(row)

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
    metric_keys = [
        "initial_mean_error_m",
        "final_mean_error_m",
        "initial_median_error_m",
        "final_median_error_m",
        "initial_max_error_m",
        "final_max_error_m",
        "initial_p95_error_m",
        "final_p95_error_m",
        "initial_mean_xy_error_m",
        "final_mean_xy_error_m",
        "initial_mean_z_error_m",
        "final_mean_z_error_m",
        "initial_within_2mm_rate",
        "final_within_2mm_rate",
        "initial_within_5mm_rate",
        "final_within_5mm_rate",
        "initial_within_10mm_rate",
        "final_within_10mm_rate",
        "initial_wrong_key_nearest_count",
        "final_wrong_key_nearest_count",
        "initial_mean_center_error_m",
        "final_mean_center_error_m",
        "initial_mean_surface_error_m",
        "final_mean_surface_error_m",
    ]
    metric_summary = {
        f"mean_{key}": float(np.mean([row[key] for row in rows])) if rows else None
        for key in metric_keys
        if not key.endswith("_count")
    }
    metric_summary.update(
        {
            f"sum_{key}": int(np.sum([row[key] for row in rows])) if rows else None
            for key in metric_keys
            if key.endswith("_count")
        }
    )
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
            "contact_metrics": metric_summary,
            "contact_refinement_config": {
                "max_iter": int(args.max_iter),
                "pose_reg_weight": float(args.pose_reg_weight),
                "smooth_weight": float(args.smooth_weight),
                "use_previous_pose_smoothing": bool(args.use_previous_pose_smoothing),
                "inactive_weight": float(args.inactive_weight),
                "inactive_margin_m": float(args.inactive_margin_m),
                "z_weight": float(args.z_weight),
                "key_z_offset_m": float(args.key_z_offset_m),
                "xy_tolerance_m": float(args.xy_tolerance_m),
                "z_tolerance_m": float(args.z_tolerance_m),
                "worst_contact_weight": float(args.worst_contact_weight),
                "mean_contact_weight": float(args.mean_contact_weight),
                "two_stage_refinement": not bool(args.disable_two_stage_refinement),
                "micro_max_iter": int(args.micro_max_iter),
                "micro_pose_reg_weight": float(args.micro_pose_reg_weight),
                "micro_z_weight": float(args.micro_z_weight),
                "assignment_mode": str(args.assignment_mode),
                "crossing_penalty": float(args.crossing_penalty),
                "hand_side_penalty": float(args.hand_side_penalty),
                "target_mode": str(args.target_mode),
                "white_key_half_width_m": float(args.white_key_half_width_m),
                "black_key_half_width_m": float(args.black_key_half_width_m),
                "key_front_back_tolerance_m": float(args.key_front_back_tolerance_m),
            },
        },
    )
    print(f"Wrote contact-refined labels: {output}")


if __name__ == "__main__":
    main()
