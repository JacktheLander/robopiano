#!/usr/bin/env python3
"""Merge Variations evaluation CSVs (local vs diffusion / latent MDN runs) and save comparison plots."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

VARIATIONS_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = VARIATIONS_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from variations.utils.io import ensure_dir


def _float(row: dict[str, str], key: str) -> float:
    raw = row.get(key)
    if raw in {None, ""}:
        return float("nan")
    return float(raw)


def load_metric_rows(path: Path, suite_label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            norm_mse = _float(row, "normalized_pose_mse")
            if norm_mse != norm_mse:  # nan
                norm_mse = _float(row, "test_loss")
            joint = _float(row, "joint_mse")
            if joint != joint:
                joint = _float(row, "denormalized_pose_mse")
            if joint != joint:
                joint = _float(row, "pose_mse")
            ms = _float(row, "inference_time_ms_per_sample")
            params = int(float(row.get("num_parameters", "nan"))) if row.get("num_parameters") else 0
            raw_name = str(row.get("model_name", "unknown")).strip()
            pretty = {
                "mlp_baseline": "MLP",
                "diffusion": "Diffusion",
                "latent_mdn_best_component": "Latent MDN",
            }.get(raw_name, raw_name)
            rows.append(
                {
                    "suite": suite_label,
                    "model_key": raw_name,
                    "label": f"{pretty}\n({suite_label})",
                    "pretty_model": pretty,
                    "joint_mse": joint,
                    "normalized_pose_mse": norm_mse,
                    "inference_time_ms_per_sample": ms,
                    "num_parameters": params,
                }
            )
    return rows


def plot_comparison(rows: list[dict[str, Any]], out_png: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    if not rows:
        raise SystemExit("No rows to plot.")
    labels = [r["label"] for r in rows]
    x = np.arange(len(labels))
    width = 0.42

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5))
    fig.suptitle("Press-pose models (validation split)")

    ax = axes[0, 0]
    vals = [r["joint_mse"] for r in rows]
    colors = [_suite_color(r["suite"]) for r in rows]
    ax.bar(x, vals, width, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Joint MSE (denormalized)")
    ax.set_title("Lower is better")

    ax = axes[0, 1]
    vals = [r["normalized_pose_mse"] for r in rows]
    ax.bar(x, vals, width, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Normalized joint MSE")

    ax = axes[1, 0]
    vals = [max(r["inference_time_ms_per_sample"], 1e-6) for r in rows]
    ax.bar(x, vals, width, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Inference ms / sample")
    ax.set_yscale("log")

    ax = axes[1, 1]
    vals = [max(r["num_parameters"], 1) for r in rows]
    ax.bar(x, vals, width, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Trainable parameters")
    ax.set_yscale("log")

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=_suite_color(s), edgecolor="black")
        for s in sorted({r["suite"] for r in rows}, key=str)
    ]
    labs = sorted({r["suite"] for r in rows}, key=str)
    fig.legend(handles, labs, loc="lower center", ncol=min(4, len(labs)), frameon=False)

    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    ensure_dir(out_png.parent)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _suite_color(suite: str) -> str:
    if "debug" in suite.lower() or "local" in suite.lower():
        return "#6baed6"
    if "full" in suite.lower() or "rp1m" in suite.lower():
        return "#74c476"
    return "#bdbdbd"


def write_merged_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "suite",
        "pretty_model",
        "joint_mse",
        "normalized_pose_mse",
        "inference_time_ms_per_sample",
        "num_parameters",
        "source_csv",
    ]
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot merged Variations press-pose evaluation CSVs.")
    parser.add_argument(
        "--csv-suite",
        action="append",
        nargs=2,
        metavar=("CSV_PATH", "SUITE_LABEL"),
        required=True,
        help="Repeat for each metrics CSV and its legend suite label (e.g. debug_local RP1M_full_val).",
    )
    parser.add_argument(
        "--output-png",
        default=str(VARIATIONS_ROOT / "outputs" / "comparisons" / "press_pose_comparison.png"),
        help="Where to save the figure.",
    )
    parser.add_argument(
        "--merged-csv",
        default=None,
        help="Optional path to write a merged long-form CSV.",
    )
    args = parser.parse_args()

    all_rows: list[dict[str, Any]] = []
    for csv_path_str, suite in args.csv_suite:
        path = Path(csv_path_str).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        loaded = load_metric_rows(path, suite.strip())
        for r in loaded:
            r["source_csv"] = str(path)
        all_rows.extend(loaded)

    out_png = Path(args.output_png).expanduser().resolve()
    plot_comparison(all_rows, out_png)
    print(f"Saved plot: {out_png}")

    if args.merged_csv:
        mp = Path(args.merged_csv).expanduser().resolve()
        write_merged_csv(all_rows, mp)
        print(f"Saved merged CSV: {mp}")


if __name__ == "__main__":
    main()
