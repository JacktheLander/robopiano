#!/usr/bin/env python3
"""Standalone diagnostic: primitive dropout / filtering vs Stage 1 frequencies and online metrics.

Read-only w.r.t. Stage 1 primitives and any training process. Writes outputs under
<transformer-run-root>/artifacts/primitive_dropout_eval/ by default.

Run from the Sonata repo root with PYTHONPATH including src, e.g.:
  python scripts/evaluate_primitive_dropout.py \\
    --primitive-root /path/to/outputs/primitives/medium \\
    --transformer-run-root /path/to/outputs/transformer/<run>
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

_SONATA_ROOT_SCRIPT = Path(__file__).resolve().parents[1]
_MPL_FALLBACK = _SONATA_ROOT_SCRIPT / ".mpl_config_eval_primitive_dropout"
try:
    _MPL_FALLBACK.mkdir(parents=True, exist_ok=True)
except OSError:
    _MPL_FALLBACK = Path(tempfile.gettempdir()) / "sonata_mpl_eval_primitive_dropout"
    _MPL_FALLBACK.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_FALLBACK))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sonata.transformer.dataset import PlannerMetadata, PrimitiveSequenceDataset, load_transformer_inputs
from sonata.transformer.primitive_filter import (
    build_primitive_filter,
    canonical_alignment,
    coerce_primitive_table_id,
    load_primitive_quality_table,
    vocab_indices_for_bad_canonical,
)
from sonata.utils.io import read_json, read_table

_LOGGER = logging.getLogger("evaluate_primitive_dropout")


def _artifacts_dir(run_root: Path) -> Path:
    return (run_root / "artifacts").resolve()


def _resolve_default_paths(
    primitive_root: Path,
    transformer_run_root: Path,
    *,
    quality_metrics_path: Path | None,
    tokens_path: Path | None,
    output_dir: Path | None,
    valid_ids_path: Path | None,
    bad_ids_path: Path | None,
) -> tuple[Path | None, Path, Path, Path | None, Path | None]:
    art = _artifacts_dir(transformer_run_root)
    resolved_quality = quality_metrics_path.resolve() if quality_metrics_path else None
    if tokens_path is not None:
        tok = tokens_path.resolve()
    else:
        tok = (primitive_root / "tokens" / "primitive_tokens").resolve()
    out = (output_dir.resolve() if output_dir else art / "primitive_dropout_eval").resolve()
    vpath = valid_ids_path.resolve() if valid_ids_path else (art / "primitive_filter_valid_ids.txt")
    bpath = bad_ids_path.resolve() if bad_ids_path else (art / "primitive_filter_bad_ids.txt")
    if not vpath.exists():
        vpath = None  # type: ignore[assignment]
    if not bpath.exists():
        bpath = None  # type: ignore[assignment]
    return resolved_quality, tok, out, vpath, bpath


def _load_id_file(path: Path | None) -> set[int] | None:
    if path is None or not path.exists():
        return None
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return set()
    ids: set[int] = set()
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ids.add(int(line.split()[0]))
    return ids


def _load_primitive_filter_summary(run_root: Path) -> dict[str, Any] | None:
    p = _artifacts_dir(run_root) / "primitive_filter_summary.json"
    if not p.exists():
        return None
    return read_json(p)


def _load_training_config(run_root: Path) -> dict[str, Any] | None:
    p = _artifacts_dir(run_root) / "config.json"
    if not p.exists():
        return None
    return read_json(p)


def _build_rebuild_transformer_config(config: dict[str, Any] | None, summary: dict[str, Any] | None) -> dict[str, Any]:
    """configs/transformer YAML + snapshots; overlay summary.Filter fields so diagnostics match trainer logs."""

    rebuilt: dict[str, Any] = dict(config or {})
    pf_src = dict(rebuilt.get("primitive_filter") or {})
    summary_keys = (
        "enabled",
        "mode",
        "min_onset_f1",
        "strong_onset_f1",
        "require_key_press",
        "require_target_hit",
        "require_strong_f1",
        "bad_primitive_weight",
        "borderline_weight",
        "quality_metrics_path",
        "save_filter_artifacts",
        "remap_invalid_targets",
    )
    if summary:
        for key in summary_keys:
            if summary.get(key) is not None:
                pf_src[key] = summary[key]
    rebuilt["primitive_filter"] = pf_src
    if rebuilt.get("context_length") is None:
        rebuilt["context_length"] = int(config.get("context_length", 48)) if config else 48
    return rebuilt


def _row_canonical_counts_train(token_df: pd.DataFrame) -> dict[int, int]:
    """Canonical primitive id -> count of train-split token rows (Stage 1 assignments)."""
    if "split" in token_df.columns:
        frame = token_df[token_df["split"] == "train"]
    else:
        frame = token_df
    counts: dict[int, int] = {}
    for raw in frame["primitive_id"].astype(str):
        c = coerce_primitive_table_id(raw)
        if c is None:
            continue
        cid = int(c)
        counts[cid] = counts.get(cid, 0) + 1
    return counts


def _merge_quality_with_rows(
    quality_df: pd.DataFrame | None,
    row_counts: dict[int, int],
    all_primitive_ids: list[int],
) -> pd.DataFrame:
    rows_out: list[dict[str, Any]] = []
    q_index: dict[int, pd.Series] = {}
    if quality_df is not None and not quality_df.empty:
        for _, r in quality_df.iterrows():
            q_index[int(r["primitive_id"])] = r
    total_rows = sum(row_counts.values())
    for pid in sorted(all_primitive_ids):
        cnt = int(row_counts.get(pid, 0))
        frac = (cnt / total_rows) if total_rows > 0 else 0.0
        qrow = q_index.get(pid)
        onset = float(qrow["onset_f1"]) if qrow is not None and pd.notna(qrow.get("onset_f1")) else float("nan")
        kp = None
        th = None
        sf = float("nan")
        qstat = "missing_quality"
        if qrow is not None:
            kp = qrow.get("key_press")
            th = qrow.get("target_hit")
            if pd.notna(qrow.get("strong_f1")):
                sf = float(qrow["strong_f1"])  # type: ignore[arg-type]
            qstat = str(qrow.get("quality_status", "ok"))
        rows_out.append(
            {
                "primitive_id": pid,
                "onset_f1": onset,
                "key_press": kp,
                "target_hit": th,
                "strong_f1": sf,
                "frequency_count": cnt,
                "frequency_fraction": frac,
                "quality_status": qstat,
            }
        )
    return pd.DataFrame(rows_out)


def _weighted_mean_onset(freq: dict[int, int], onset: dict[int, float], id_set: set[int]) -> tuple[float, int]:
    num = 0.0
    den = 0
    for k in id_set:
        f = int(freq.get(k, 0))
        o = onset.get(k)
        if f <= 0 or o is None or not math.isfinite(o):
            continue
        num += o * f
        den += f
    if den == 0:
        return float("nan"), 0
    return num / den, den


def _unweighted_mean_onset(onset: dict[int, float], id_set: set[int]) -> float:
    vals = [onset[k] for k in sorted(id_set) if k in onset and math.isfinite(onset[k])]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def _simulate_train_targets(
    token_df: pd.DataFrame,
    metadata: PlannerMetadata,
    context_length: int,
    *,
    bad_canonical: set[int],
    effective_mode: str,
) -> tuple[int, int, np.ndarray, np.ndarray]:
    """Return (before_len, after_len, baseline_targets, filtered_targets)."""
    baseline = PrimitiveSequenceDataset(
        token_df,
        metadata,
        context_length=context_length,
        split="train",
        vocab_bad_indices=None,
        primitive_filter_mode="none",
        primitive_vocab_weights=None,
    )
    bt = baseline.target_primitive
    mode = str(effective_mode or "none").lower()
    bad_vocab = vocab_indices_for_bad_canonical(metadata, bad_canonical)
    if mode == "drop" and bad_vocab:
        filt = PrimitiveSequenceDataset(
            token_df,
            metadata,
            context_length=context_length,
            split="train",
            vocab_bad_indices=bad_vocab,
            primitive_filter_mode="drop",
            primitive_vocab_weights=None,
        )
        ft = filt.target_primitive
        return len(bt), len(ft), bt, ft
    return len(bt), len(bt), bt, bt


def _annotate_outliers(ax: Any, xs: list[float], ys: list[float], labels: list[int], top_n: int = 3) -> None:
    if not xs:
        return
    arr = np.asarray(xs, dtype=np.float64)
    brr = np.asarray(ys, dtype=np.float64)
    # score: high frequency, low F1
    med_x = float(np.nanmedian(arr)) or 1.0
    med_y = float(np.nanmedian(brr))
    if not math.isfinite(med_y):
        med_y = 0.5
    scored: list[tuple[float, int]] = []
    for i, lab in enumerate(labels):
        xi, yi = arr[i], brr[i]
        if not (math.isfinite(xi) and math.isfinite(yi)):
            continue
        score = (xi / med_x) * max(0.0, (med_y - yi))
        scored.append((score, i))
    scored.sort(reverse=True)
    for _, idx in scored[:top_n]:
        ax.annotate(str(labels[idx]), (xs[idx], ys[idx]), textcoords="offset points", xytext=(4, 4), fontsize=8)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Evaluate primitive dropout vs Stage 1 frequency and online metrics.")
    parser.add_argument("--primitive-root", type=Path, required=True)
    parser.add_argument("--transformer-run-root", type=Path, required=True)
    parser.add_argument("--quality-metrics-path", type=Path, default=None)
    parser.add_argument("--tokens-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--min-onset-f1", type=float, default=None)
    parser.add_argument("--strong-onset-f1", type=float, default=None)
    parser.add_argument("--valid-ids-path", type=Path, default=None)
    parser.add_argument("--bad-ids-path", type=Path, default=None)
    args = parser.parse_args()

    primitive_root = args.primitive_root.resolve()
    transformer_run = args.transformer_run_root.resolve()
    resolved_quality, tokens_base, output_dir, valid_path, bad_path = _resolve_default_paths(
        primitive_root,
        transformer_run,
        quality_metrics_path=args.quality_metrics_path,
        tokens_path=args.tokens_path,
        output_dir=args.output_dir,
        valid_ids_path=args.valid_ids_path,
        bad_ids_path=args.bad_ids_path,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = _load_primitive_filter_summary(transformer_run)
    config = _load_training_config(transformer_run)
    if summary is None:
        _LOGGER.warning("primitive_filter_summary.json not found yet; proceeding with CLI thresholds only.")
    if config is None:
        _LOGGER.warning("artifacts/config.json missing; cannot simulate exact training target counts (context length).")

    min_f1 = args.min_onset_f1
    strong_f1 = args.strong_onset_f1
    if min_f1 is None:
        min_f1 = float(summary["min_onset_f1"]) if summary and summary.get("min_onset_f1") is not None else 0.40
        if args.min_onset_f1 is None and summary is None:
            _LOGGER.warning("--min-onset-f1 defaulting to 0.40 (no summary).")
    else:
        min_f1 = float(min_f1)

    if strong_f1 is None:
        strong_f1 = float(summary["strong_onset_f1"]) if summary and summary.get("strong_onset_f1") is not None else 0.60
    else:
        strong_f1 = float(strong_f1)

    explicit_q = str(resolved_quality) if resolved_quality else None
    quality_df, quality_resolved_path, qual_warn = load_primitive_quality_table(primitive_root, explicit_q)
    qual_path_str = str(quality_resolved_path) if quality_resolved_path else None
    if qual_warn:
        _LOGGER.warning("Quality table load: %s", qual_warn)

    token_df, metadata = load_transformer_inputs(primitive_root, family_mapping_mode="heuristic_stats")
    if args.tokens_path is not None:
        tp = args.tokens_path.expanduser().resolve()
        if tp.is_file() and tp.suffix.lower() in (".csv", ".parquet"):
            token_df = read_table(tp.with_suffix(""))
        else:
            token_df = read_table(tp)

    row_counts = _row_canonical_counts_train(token_df)
    all_ids_sorted = sorted(row_counts.keys())

    _, ordered_canon = canonical_alignment(metadata)
    universe_ids = sorted(set(all_ids_sorted) | set(ordered_canon))

    rebuild_config = _build_rebuild_transformer_config(config, summary)
    filter_bundle = build_primitive_filter(quality_df, rebuild_config, metadata)
    diag = dict(filter_bundle.get("diagnostics") or {})
    bad_rebuilt: set[int] = set(int(x) for x in (filter_bundle.get("bad_primitive_ids") or set()))
    valid_rebuilt: set[int] = set(int(x) for x in (filter_bundle.get("valid_primitive_ids") or set()))

    bad_file = _load_id_file(bad_path)
    valid_file = _load_id_file(valid_path)
    bad_ids = bad_file if bad_file is not None else bad_rebuilt
    valid_ids = valid_file if valid_file is not None else valid_rebuilt

    if bad_file is not None and bad_file != bad_rebuilt:
        _LOGGER.warning("primitive_filter_bad_ids.txt disagrees with rebuilt filter; using file + warning in summary.")
    if valid_file is not None and valid_file != valid_rebuilt:
        _LOGGER.warning("primitive_filter_valid_ids.txt disagrees with rebuilt filter; trusting file for reporting.")

    effective_mode = str(diag.get("effective_mode") or (summary or {}).get("effective_mode") or "none")
    config_mode = str((config or {}).get("primitive_filter", {}).get("mode", "none"))
    pf_file = (config or {}).get("primitive_filter") or {}
    config_enabled_flag = bool(pf_file.get("enabled", False))
    summary_enabled_flag = summary["enabled"] if summary is not None and "enabled" in summary else None
    raw_enabled = summary_enabled_flag if summary_enabled_flag is not None else config_enabled_flag

    onset_by_id: dict[int, float] = {}
    if quality_df is not None and not quality_df.empty:
        for _, row in quality_df.iterrows():
            pid = int(row["primitive_id"])
            if pd.notna(row.get("onset_f1")):
                onset_by_id[pid] = float(row["onset_f1"])

    merged = _merge_quality_with_rows(quality_df, row_counts, universe_ids)
    retained_mask = np.array([merged["primitive_id"].iloc[i] not in bad_ids for i in range(len(merged))], dtype=bool)
    merged["retained"] = retained_mask
    merged["dropped"] = ~retained_mask

    total_row = int(sum(row_counts.values()))
    bad_row = int(sum(c for k, c in row_counts.items() if k in bad_ids))
    retained_row = total_row - bad_row
    frac_ret_row = (retained_row / total_row) if total_row > 0 else 0.0
    frac_drop_row = (bad_row / total_row) if total_row > 0 else 0.0

    w_used, den_used = _weighted_mean_onset(row_counts, onset_by_id, valid_ids)
    w_all, _ = _weighted_mean_onset(row_counts, onset_by_id, set(universe_ids))
    w_drop, _ = _weighted_mean_onset(row_counts, onset_by_id, bad_ids)
    uw_used = _unweighted_mean_onset(onset_by_id, valid_ids)

    missing_quality = merged["quality_status"].eq("missing_quality").sum()
    summary_dropped_targets = summary.get("train_examples_dropped") if summary else None
    summary_before = summary.get("train_examples_before") if summary else None
    summary_after = summary.get("train_examples_after") if summary else None

    ctx = int(rebuild_config.get("context_length", 48))
    sim_before, sim_after, baseline_tgts, filt_tgts = _simulate_train_targets(
        token_df,
        metadata,
        ctx,
        bad_canonical=bad_ids,
        effective_mode=effective_mode,
    )
    sim_drop = max(0, sim_before - sim_after)
    bad_vocab = vocab_indices_for_bad_canonical(metadata, bad_ids)
    bad_in_baseline_vocab = False
    if baseline_tgts.size > 0 and bad_vocab:
        bad_in_baseline_vocab = bool(np.isin(baseline_tgts, np.array(list(bad_vocab), dtype=np.int64)).any())

    bad_in_filtered_targets = "unknown"
    if effective_mode == "drop" and bad_vocab and filt_tgts.size > 0:
        bad_in_filtered_targets = str(bool(np.isin(filt_tgts, np.array(list(bad_vocab), dtype=np.int64)).any()))
    elif effective_mode != "drop" or not bad_vocab:
        bad_in_filtered_targets = "n/a"

    bad_in_raw_rows = any(k in bad_ids for k in row_counts)

    warnings: list[str] = []
    if qual_warn:
        warnings.append(f"quality_load:{qual_warn}")
    if summary_dropped_targets is not None and sim_drop != int(summary_dropped_targets):
        diff = abs(sim_drop - int(summary_dropped_targets))
        if sim_before > 0 and diff / sim_before > 0.01:
            warnings.append(
                f"target_drop_mismatch simulated_dropped={sim_drop} summary_dropped={summary_dropped_targets} "
                f"(baseline_targets={sim_before})"
            )
    if missing_quality > 0:
        warnings.append(f"{int(missing_quality)} primitives missing quality onset_f1 in merged metrics table.")

    csv_out = merged.copy()
    csv_out["retained"] = merged["retained"]
    csv_out["dropped"] = merged["dropped"]
    # align column order with spec
    spec_cols = [
        "primitive_id",
        "onset_f1",
        "key_press",
        "target_hit",
        "strong_f1",
        "frequency_count",
        "frequency_fraction",
        "retained",
        "dropped",
        "quality_status",
    ]
    for c in spec_cols:
        if c not in csv_out.columns:
            csv_out[c] = np.nan if c in {"onset_f1", "strong_f1"} else None
    csv_out = csv_out[spec_cols]
    csv_path = output_dir / "primitive_dropout_eval_table.csv"
    csv_out.to_csv(csv_path, index=False)

    json_summary: dict[str, Any] = {
        "primitive_root": str(primitive_root),
        "transformer_run_root": str(transformer_run),
        "quality_metrics_path": qual_path_str,
        "tokens_path": str(tokens_base),
        "total_primitives": int(len(universe_ids)),
        "retained_primitive_count": int(len(valid_ids)),
        "dropped_primitive_count": int(len(bad_ids)),
        "missing_quality_count": int(missing_quality),
        # Spec naming: frequencies from Stage 1 train-split token rows
        "total_token_assignments": total_row,
        "retained_token_assignments": retained_row,
        "dropped_token_assignments": bad_row,
        "retained_token_fraction": float(frac_ret_row),
        "dropped_token_fraction": float(frac_drop_row),
        "total_training_targets_before_filter": sim_before,
        "retained_training_targets_after_filter": sim_after,
        "dropped_training_targets": sim_drop,
        "retained_training_target_fraction": (sim_after / sim_before) if sim_before > 0 else float("nan"),
        "dropped_training_target_fraction": (sim_drop / sim_before) if sim_before > 0 else float("nan"),
        "weighted_f1_used": float(w_used) if math.isfinite(w_used) else None,
        "unweighted_f1_used": float(uw_used) if math.isfinite(uw_used) else None,
        "weighted_f1_all": float(w_all) if math.isfinite(w_all) else None,
        "weighted_f1_dropped": float(w_drop) if math.isfinite(w_drop) else None,
        "weighted_f1_denominator_retained_finite_onset": int(den_used),
        "frequency_definition": "train_split_primitive_token_rows",
        "min_onset_f1": float(min_f1),
        "strong_onset_f1": float(strong_f1),
        "bad_ids": sorted(bad_ids),
        "valid_ids": sorted(valid_ids),
        "primitive_filter_summary_enabled": summary_enabled_flag,
        "primitive_filter_config_enabled_active": raw_enabled,
        "primitive_filter_config_enabled_config_json": config_enabled_flag,
        "effective_mode_reported": effective_mode,
        "effective_enabled": bool(diag.get("effective_enabled", summary.get("effective_enabled") if summary else False)),
        "summary_train_examples_before": summary_before,
        "summary_train_examples_after": summary_after,
        "summary_train_examples_dropped": summary_dropped_targets,
        "bad_ids_in_raw_stage1_train_rows": bad_in_raw_rows,
        "bad_vocab_indices_in_baseline_train_targets": bad_in_baseline_vocab,
        "bad_ids_in_filtered_stage2_train_targets": bad_in_filtered_targets,
        "context_length_used_for_simulation": ctx,
        "warnings": warnings,
    }
    (output_dir / "primitive_dropout_eval_summary.json").write_text(json.dumps(json_summary, indent=2), encoding="utf-8")

    # --- Plots (retained only for bar charts A/B; all IDs for diagnostic D) ---
    ret_df = merged[merged["primitive_id"].isin(valid_ids)].copy()
    ids_r = ret_df["primitive_id"].tolist()
    onsets_r = [float(x) if math.isfinite(x) else 0.0 for x in ret_df["onset_f1"].tolist()]
    freqs_r = ret_df["frequency_count"].astype(int).tolist()

    if ids_r:
        fig, ax = plt.subplots(figsize=(max(8, len(ids_r) * 0.25), 4))
        ax.bar(np.asarray(ids_r, dtype=np.float64), onsets_r, color="#2b8cbe")
        ax.set_xlabel("primitive_id")
        ax.set_ylabel("onset_f1")
        ax.set_title("Retained primitives: onset_f1")
        fig.tight_layout()
        fig.savefig(output_dir / "retained_primitive_accuracy_bar.png", dpi=150)
        plt.close(fig)

        fig2, ax2 = plt.subplots(figsize=(max(8, len(ids_r) * 0.25), 4))
        ax2.bar(np.asarray(ids_r, dtype=np.float64), freqs_r, color="#66c2a4")
        ax2.set_xlabel("primitive_id")
        ax2.set_ylabel("frequency_count (train token rows)")
        ax2.set_title("Retained primitives: Stage 1 frequency")
        fig2.tight_layout()
        fig2.savefig(output_dir / "retained_primitive_frequency_bar.png", dpi=150)
        plt.close(fig2)

        fig3, ax3 = plt.subplots(figsize=(7, 5))
        xs = [float(f) for f in ret_df["frequency_count"]]
        ys = ret_df["onset_f1"].astype(float).tolist()
        ax3.scatter(xs, ys, alpha=0.75, edgecolors="k", linewidths=0.3)
        ax3.set_xlabel("frequency_count (train token rows)")
        ax3.set_ylabel("onset_f1")
        ax3.set_title("Retained: accuracy vs frequency")
        labs = ids_r
        _annotate_outliers(ax3, xs, ys, labs)
        fig3.tight_layout()
        fig3.savefig(output_dir / "retained_accuracy_vs_frequency.png", dpi=150)
        plt.close(fig3)

    # Diagnostic: retained vs dropped by onset vs id
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    drops = merged[merged["dropped"]].copy()
    keeps = merged[merged["retained"]].copy()
    ax4.scatter(keeps["primitive_id"], keeps["onset_f1"], c="#1b9e77", label="retained", alpha=0.8)
    if not drops.empty:
        ax4.scatter(drops["primitive_id"], drops["onset_f1"], c="#d95f02", marker="x", label="dropped/bad", s=42)
    ax4.set_xlabel("primitive_id")
    ax4.set_ylabel("onset_f1")
    ax4.legend()
    ax4.set_title("Primitive dropout status (onset vs id)")
    fig4.tight_layout()
    fig4.savefig(output_dir / "primitive_dropout_status.png", dpi=150)
    plt.close(fig4)

    slim_keys = (
        "effective_enabled",
        "effective_mode",
        "skipped",
        "reason",
        "filter_disable_reason",
        "bad_count",
        "valid_count",
        "quality_rows",
        "missing_quality_count",
    )
    diag_compact = {k: diag.get(k) for k in slim_keys}
    if summary:
        for sk in ("train_examples_before", "train_examples_after", "train_examples_dropped", "drop_fraction"):
            if sk in summary:
                diag_compact[f"summary_{sk}"] = summary.get(sk)

    _print_stdout_summary(
        primitive_root,
        transformer_run,
        qual_path_str,
        str(tokens_base),
        universe_ids,
        valid_ids,
        bad_ids,
        total_row,
        retained_row,
        bad_row,
        frac_ret_row,
        frac_drop_row,
        w_used,
        uw_used,
        w_all,
        w_drop,
        bad_in_raw_rows,
        bad_in_filtered_targets,
        sim_drop,
        summary_dropped_targets,
        effective_mode,
        summary_enabled_flag,
        config_enabled_flag,
        config_mode,
        warnings,
        diag_compact,
    )
    _LOGGER.info("Wrote diagnostics under %s", output_dir)
    return 0


def _print_stdout_summary(
    primitive_root: Path,
    transformer_run: Path,
    qual_path: str | None,
    tokens_path: str,
    universe_ids: list[int],
    valid_ids: set[int],
    bad_ids: set[int],
    total_row: int,
    retained_row: int,
    bad_row: int,
    frac_ret_row: float,
    frac_drop_row: float,
    w_used: float,
    uw_used: float,
    w_all: float,
    w_drop: float,
    bad_in_raw_rows: bool,
    bad_filtered: str,
    sim_drop: int,
    summary_drop: Any,
    effective_mode: str,
    summary_enabled_flag: bool | None,
    config_enabled_flag: bool,
    config_mode: str,
    warnings: list[str],
    diag_compact: dict[str, Any],
) -> None:
    lines = [
        "Primitive dropout evaluation",
        "----------------------------",
        f"Primitive root: {primitive_root}",
        f"Transformer run: {transformer_run}",
        f"Quality metrics: {qual_path or '(none resolved)'}",
        f"Tokens table base: {tokens_path}",
        f"Total primitives (union vocabulary + rows): {len(universe_ids)}",
        f"Retained primitive IDs (count): {len(valid_ids)}",
        f"Dropped/bad primitive IDs (count): {len(bad_ids)}",
        "",
        "Stage 1 train-split token-row counts (primitive_frequency definition):",
        f"  Total token assignments (rows): {total_row}",
        f"  Retained row assignments (non-bad id): {retained_row} ({frac_ret_row:.4f})",
        f"  Dropped row assignments (bad id rows): {bad_row} ({frac_drop_row:.4f})",
        "",
        "Training target sequences (simulated via PrimitiveSequenceDataset, train split):",
        f"  See JSON for baseline/filtered lengths; simulated dropped targets: {sim_drop}",
        "",
        "Weighted/unweighted onset F1 (frequency weights = train token-row counts):",
        f"  Weighted F1 of retained (valid) primitives: {w_used}",
        f"  Unweighted mean onset_f1 retained:          {uw_used}",
        f"  Weighted F1 of all primitives w/ onset:      {w_all}",
        f"  Weighted F1 of dropped/bad primitives:      {w_drop}",
        "",
        "Run filter telemetry:",
        f"  summary primitive_filter.enabled:    {summary_enabled_flag}",
        f"  config.json primitive_filter.enabled: {config_enabled_flag}",
        f"  config primitive_filter.mode:         {config_mode}",
        f"  effective_mode (trainer):             {effective_mode}",
        f"  diagnostics (compact):                {diag_compact}",
        "",
        "Dropout verification:",
        f"  Bad primitives appear in raw Stage 1 train token rows: {bad_in_raw_rows}",
        f"  Bad primitives still in filtered Stage 2 train targets: {bad_filtered}",
        f"  Simulated dropped target count: {sim_drop}",
        f"  Summary train_examples_dropped (artifact): {summary_drop}",
        "",
    ]
    if warnings:
        lines.append("Warnings:")
        lines.extend([f"  - {w}" for w in warnings])
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
