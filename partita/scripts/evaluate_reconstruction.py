from __future__ import annotations

import sys
from pathlib import Path

PARTITA_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PARTITA_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse
import math
import numpy as np
import pandas as pd

from partita.evaluation.metrics import action_metrics, key_metrics
from partita.evaluation.pianoroll import plot_pianoroll_comparison
from partita.utils.config import experiment_name, load_config, output_root
from partita.utils.io import ensure_dir, load_json, save_json


def _entropy(counts) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log(p)).sum())


def _prefix_metrics(prefix: str, values: dict) -> dict[str, object]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def _load_rollout_validation(root: Path, exp: str) -> dict[str, object]:
    summary_path = root / "rollout" / exp / "rollout_summary.json"
    if not summary_path.exists():
        return {
            "scoring_validation_status": "rollout_missing",
            "scoring_validation_note": (
                "Run partita/scripts/simulate_rollout.py --which reconstructed through Slurm to score "
                "the learned primitive reconstruction from RoboPianist key presses."
            ),
        }
    summary = load_json(summary_path)
    results = [item for item in summary.get("results", []) if isinstance(item, dict)]
    by_label = {str(item.get("label")): item for item in results}
    reconstructed = by_label.get("reconstructed")
    original = by_label.get("original_target")
    if reconstructed is None:
        return {
            "scoring_validation_status": "reconstructed_rollout_missing",
            "scoring_validation_note": "Rollout summary exists, but it does not include the reconstructed action replay.",
        }

    metrics: dict[str, object] = {
        "scoring_validation_status": "rollout_scored",
        "scoring_validation_note": (
            "Primary key metrics come from RoboPianist piano activation produced by replayed reconstructed actions."
        ),
    }
    keep_keys = [
        "rollout_key_precision",
        "rollout_key_recall",
        "rollout_key_f1",
        "rollout_mispress_rate",
        "rollout_scored_steps",
        "rollout_scoring_source",
        "total_reward",
        "actions_executed",
        "terminated",
        "render_error",
        "video_path",
        "audio_warning",
        "audio_midi_note_event_count",
    ]
    metrics.update(_prefix_metrics("reconstructed", {key: reconstructed.get(key) for key in keep_keys if key in reconstructed}))
    if original is not None:
        metrics.update(_prefix_metrics("original_target", {key: original.get(key) for key in keep_keys if key in original}))
        recon_f1 = reconstructed.get("rollout_key_f1")
        orig_f1 = original.get("rollout_key_f1")
        if isinstance(recon_f1, (int, float)) and isinstance(orig_f1, (int, float)) and math.isfinite(float(orig_f1)):
            metrics["reconstructed_vs_original_rollout_key_f1_ratio"] = float(recon_f1) / max(float(orig_f1), 1e-12)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Partita reconstruction quality.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    exp = experiment_name(config)
    root = output_root(config)
    data_dir = root / "data" / exp
    prim_dir = root / "primitives" / exp
    recon_dir = root / "reconstruction" / exp
    eval_dir = ensure_dir(root / "evaluation" / exp)

    original = np.load(recon_dir / "original_actions.npy")
    reconstructed = np.load(recon_dir / "reconstructed_actions.npy")
    metrics = action_metrics(original, reconstructed)

    original_piano = None
    reconstructed_piano = None
    if (recon_dir / "original_piano_states.npy").exists() and (recon_dir / "reconstructed_piano_states.npy").exists():
        original_piano = np.load(recon_dir / "original_piano_states.npy")
        reconstructed_piano = np.load(recon_dir / "reconstructed_piano_states.npy")
        offline_keys = key_metrics(original_piano, reconstructed_piano, threshold=float(config.get("selection", {}).get("key_threshold", 0.5)))
        metrics.update(_prefix_metrics("offline_primitive_profile", offline_keys))
        metrics["offline_primitive_profile_note"] = (
            "Diagnostic only: reconstructed_piano_states.npy is assembled from learned primitive mean piano profiles, "
            "not from RoboPianist playback."
        )

    selection = load_json(data_dir / "selection.json")
    target_assignments = pd.read_csv(recon_dir / "target_primitive_assignments.csv")
    primitive_summary = pd.read_csv(prim_dir / "primitive_summary.csv")
    counts = target_assignments["primitive_id"].value_counts().sort_index().values
    metrics.update({
        "num_training_trajectories": int(selection.get("num_training_trajectories", 0)),
        "num_target_segments": int(len(target_assignments)),
        "num_primitives": int(len(primitive_summary)),
        "primitive_entropy": _entropy(counts),
        "mean_primitive_coverage_across_trajectories": float(primitive_summary["trajectory_coverage_fraction"].mean()) if len(primitive_summary) else 0.0,
        "max_primitive_usage_fraction": float(counts.max() / max(counts.sum(), 1)) if len(counts) else 0.0,
    })
    metrics.update(_load_rollout_validation(root, exp))
    save_json(eval_dir / "metrics.json", metrics)

    goal = None
    target_npz = np.load(data_dir / "target_trajectory.npz")
    if "goals" in target_npz.files:
        goal = target_npz["goals"]
    plot_pianoroll_comparison(
        goal,
        original_piano,
        reconstructed_piano,
        eval_dir / "pianoroll_comparison.png",
        threshold=float(config.get("selection", {}).get("key_threshold", 0.5)),
    )
    lines = [f"{k}: {v}" for k, v in sorted(metrics.items())]
    (eval_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved evaluation metrics to {eval_dir / 'metrics.json'}")
    print("Key metrics:")
    for key in [
        "action_mse",
        "action_l1",
        "reconstructed_rollout_key_f1",
        "offline_primitive_profile_key_f1",
        "scoring_validation_status",
        "primitive_entropy",
        "mean_primitive_coverage_across_trajectories",
        "max_primitive_usage_fraction",
    ]:
        if key in metrics:
            print(f"  {key}: {metrics[key]}")


if __name__ == "__main__":
    main()
