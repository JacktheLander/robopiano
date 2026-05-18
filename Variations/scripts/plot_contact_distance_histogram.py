#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

VARIATIONS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = VARIATIONS_ROOT.parent
for path in (
    VARIATIONS_ROOT / "src",
    VARIATIONS_ROOT,
    REPO_ROOT / "Intermezzo" / "src",
    REPO_ROOT / "partita" / "src",
    REPO_ROOT,
):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from variations.contact_refinement import ContactRefinementConfig, ContactRefiner  # noqa: E402
from variations.utils.io import ensure_dir, save_json  # noqa: E402


def _float_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            "count": 0,
            "mean_m": float("nan"),
            "median_m": float("nan"),
            "p90_m": float("nan"),
            "p95_m": float("nan"),
            "p99_m": float("nan"),
            "max_m": float("nan"),
            "success_at_0p005": float("nan"),
            "success_at_0p01": float("nan"),
            "success_at_0p02": float("nan"),
            "success_at_0p05": float("nan"),
        }
    return {
        "count": int(values.size),
        "mean_m": float(np.mean(values)),
        "median_m": float(np.median(values)),
        "p90_m": float(np.percentile(values, 90)),
        "p95_m": float(np.percentile(values, 95)),
        "p99_m": float(np.percentile(values, 99)),
        "max_m": float(np.max(values)),
        "success_at_0p005": float(np.mean(values <= 0.005)),
        "success_at_0p01": float(np.mean(values <= 0.010)),
        "success_at_0p02": float(np.mean(values <= 0.020)),
        "success_at_0p05": float(np.mean(values <= 0.050)),
    }


def _surface_xy_errors(
    refiner: ContactRefiner,
    tip_positions: np.ndarray,
    key_positions: np.ndarray,
    key_indices: np.ndarray,
    cfg: ContactRefinementConfig,
) -> np.ndarray:
    return refiner._surface_xy_errors(tip_positions, key_positions, key_indices, cfg)  # noqa: SLF001


def collect_distances(
    refiner: ContactRefiner,
    poses: np.ndarray,
    target_keys: np.ndarray,
    *,
    label: str,
    cfg: ContactRefinementConfig,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    rows: list[dict[str, Any]] = []
    center_distances: list[float] = []
    surface_distances: list[float] = []
    xy_distances: list[float] = []
    z_distances: list[float] = []
    width_center_distances: list[float] = []
    width_surface_distances: list[float] = []

    for sample_index, (pose, keys) in enumerate(zip(poses, target_keys, strict=False)):
        active_keys = np.flatnonzero(np.asarray(keys, dtype=np.float32).reshape(-1)[:88] > 0.5)
        if active_keys.size == 0:
            continue
        tips = refiner.fingertip_positions(pose)
        assignment = refiner._assignment_info(tips, active_keys, cfg)  # noqa: SLF001
        if assignment.tip_positions.size == 0:
            continue
        diff = assignment.tip_positions - assignment.key_positions
        center = np.linalg.norm(diff, axis=1)
        xy = np.linalg.norm(diff[:, :2], axis=1)
        z = np.abs(diff[:, 2])
        width_center = np.abs(diff[:, 0])
        width_surface = refiner._surface_width_errors(  # noqa: SLF001
            assignment.tip_positions,
            assignment.key_positions,
            assignment.key_indices,
            cfg,
        )
        surface_xy = _surface_xy_errors(refiner, assignment.tip_positions, assignment.key_positions, assignment.key_indices, cfg)
        surface = np.sqrt(surface_xy**2 + diff[:, 2] ** 2)
        for pair_index, (tip_idx, key_idx, c, s, x, zz, wc, ws) in enumerate(
            zip(assignment.tip_indices, assignment.key_indices, center, surface, xy, z, width_center, width_surface, strict=False)
        ):
            rows.append(
                {
                    "label": label,
                    "sample_index": int(sample_index),
                    "pair_index": int(pair_index),
                    "tip_index": int(tip_idx),
                    "key_index": int(key_idx),
                    "center_distance_m": float(c),
                    "surface_distance_m": float(s),
                    "xy_distance_m": float(x),
                    "z_distance_m": float(zz),
                    "width_center_distance_m": float(wc),
                    "width_surface_distance_m": float(ws),
                }
            )
        center_distances.extend(float(v) for v in center)
        surface_distances.extend(float(v) for v in surface)
        xy_distances.extend(float(v) for v in xy)
        z_distances.extend(float(v) for v in z)
        width_center_distances.extend(float(v) for v in width_center)
        width_surface_distances.extend(float(v) for v in width_surface)

    return rows, {
        "center": np.asarray(center_distances, dtype=np.float64),
        "surface": np.asarray(surface_distances, dtype=np.float64),
        "xy": np.asarray(xy_distances, dtype=np.float64),
        "z": np.asarray(z_distances, dtype=np.float64),
        "width_center": np.asarray(width_center_distances, dtype=np.float64),
        "width_surface": np.asarray(width_surface_distances, dtype=np.float64),
    }


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "label",
        "sample_index",
        "pair_index",
        "tip_index",
        "key_index",
        "center_distance_m",
        "surface_distance_m",
        "xy_distance_m",
        "z_distance_m",
        "width_center_distance_m",
        "width_surface_distance_m",
    ]
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def plot_histograms(path: Path, series: dict[str, dict[str, np.ndarray]], title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_dir(path.parent)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    bins = np.linspace(0.0, 0.30, 61)
    colors = {"initial": "#808080", "refined": "#2f6f9f"}
    for ax, metric in zip(axes, ("center", "surface"), strict=True):
        for label, values_by_metric in series.items():
            values = values_by_metric[metric]
            if values.size == 0:
                continue
            ax.hist(
                values,
                bins=bins,
                density=True,
                alpha=0.48,
                color=colors.get(label, None),
                edgecolor="black",
                linewidth=0.25,
                label=f"{label} (n={values.size})",
            )
        ax.axvline(0.005, color="#666666", linestyle="--", linewidth=1.0)
        ax.axvline(0.010, color="#666666", linestyle=":", linewidth=1.0)
        ax.axvline(0.020, color="#666666", linestyle="-.", linewidth=1.0)
        ax.set_title(f"{metric.capitalize()} distance")
        ax.set_xlabel("Selected fingertip distance to target key (m)")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot selected fingertip-to-target-key distance histograms for contact labels.")
    parser.add_argument("--labels-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix", default="contact_distance_histogram")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--assignment-mode", choices=["nearest", "ordered", "hand_constrained"], default="ordered")
    parser.add_argument("--target-mode", choices=["key_center", "key_surface_box"], default="key_center")
    parser.add_argument("--key-z-offset-m", type=float, default=0.0)
    parser.add_argument("--white-key-half-width-m", type=float, default=0.010)
    parser.add_argument("--black-key-half-width-m", type=float, default=0.006)
    parser.add_argument("--key-front-back-tolerance-m", type=float, default=0.018)
    args = parser.parse_args()

    labels_path = Path(args.labels_npz).expanduser().resolve()
    output_dir = ensure_dir(Path(args.output_dir).expanduser().resolve())
    data = np.load(labels_path, allow_pickle=False)
    target_keys = np.asarray(data["target_keys"], dtype=np.float32)
    refined = np.asarray(data["refined_hand_state"], dtype=np.float32)
    initial = np.asarray(data["initial_hand_state"], dtype=np.float32) if "initial_hand_state" in data.files else None
    if args.max_samples is not None:
        n = int(args.max_samples)
        target_keys = target_keys[:n]
        refined = refined[:n]
        if initial is not None:
            initial = initial[:n]

    cfg = ContactRefinementConfig(
        assignment_mode=str(args.assignment_mode),
        target_mode=str(args.target_mode),
        key_z_offset_m=float(args.key_z_offset_m),
        white_key_half_width_m=float(args.white_key_half_width_m),
        black_key_half_width_m=float(args.black_key_half_width_m),
        key_front_back_tolerance_m=float(args.key_front_back_tolerance_m),
    )

    all_rows: list[dict[str, Any]] = []
    series: dict[str, dict[str, np.ndarray]] = {}
    with ContactRefiner(output_dir=output_dir / "env") as refiner:
        if initial is not None:
            rows, distances = collect_distances(refiner, initial, target_keys, label="initial", cfg=cfg)
            all_rows.extend(rows)
            series["initial"] = distances
        rows, distances = collect_distances(refiner, refined, target_keys, label="refined", cfg=cfg)
        all_rows.extend(rows)
        series["refined"] = distances

    summary = {
        "labels_npz": str(labels_path),
        "examples": int(target_keys.shape[0]),
        "assignment_mode": str(args.assignment_mode),
        "target_mode": str(args.target_mode),
        "series": {
            label: {
                metric: _float_summary(values)
                for metric, values in values_by_metric.items()
            }
            for label, values_by_metric in series.items()
        },
    }
    png_path = output_dir / f"{args.prefix}.png"
    csv_path = output_dir / f"{args.prefix}.csv"
    json_path = output_dir / f"{args.prefix}.json"
    plot_histograms(png_path, series, title=f"Selected fingertip distances: {labels_path.name}")
    write_rows_csv(csv_path, all_rows)
    save_json(json_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Saved histogram: {png_path}")
    print(f"Saved distances CSV: {csv_path}")
    print(f"Saved summary JSON: {json_path}")


if __name__ == "__main__":
    main()
