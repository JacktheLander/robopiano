"""Stage 2 primitive quality CSV used by primitive_filter downweight paths.

Writes ``<primitive-root>/metrics/stage2_primitive_quality.csv`` so training does not rely
on raw ``primitive_library.csv`` lacking onset/F1 columns. Merges online rollout summaries when
present on disk.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_LOGGER = logging.getLogger(__name__)

DEFAULT_STAGE2_QUALITY_REL = Path("metrics") / "stage2_primitive_quality.csv"


def _coalesce_online_onset(mean_f1: pd.Series, median_f1: pd.Series) -> pd.Series:
    out_mean = pd.to_numeric(mean_f1, errors="coerce")
    out_med = pd.to_numeric(median_f1, errors="coerce")
    return out_mean.where(out_mean.notna(), out_med)


def discover_online_primitive_summaries(primitive_root: Path) -> list[Path]:
    """Candidate ``primitive_summary_metrics.csv`` locations near the primitive Stage 1 tree."""

    resolved = primitive_root.resolve()
    parents: list[Path] = []
    rp = resolved
    for _ in range(6):
        parents.append(rp)
        if rp.parent == rp:
            break
        rp = rp.parent

    names = (
        "evaluation/primitives_online/primitive_summary_metrics.csv",
        "primitives_online_evaluation/primitive_summary_metrics.csv",
    )

    paths: list[Path] = []
    seen: set[str] = set()
    for base in parents:
        for suffix in names:
            candidate = (base / suffix).resolve()
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            paths.append(candidate)
    return paths


def build_stage2_quality_table_df(*, primitive_root: Path, online_summary: Path | None) -> pd.DataFrame:
    lib_path = primitive_root / "library" / "primitive_library.csv"
    if not lib_path.is_file():
        raise FileNotFoundError(f"Missing Stage 1 library at {lib_path}")

    library = pd.read_csv(lib_path)
    if "primitive_id" not in library.columns:
        raise ValueError(f"{lib_path} has no primitive_id column.")

    cols = ["primitive_id", "assignment_confidence_mean", "weighted_strike_error"]
    missing_cols = [c for c in cols if c not in library.columns]
    if missing_cols:
        raise ValueError(f"{lib_path} missing columns needed for weak-primitive surrogate: {missing_cols}")

    merged = library[cols].copy()
    merged["primitive_id"] = merged["primitive_id"].astype(str)

    optional_cols: list[str] = []
    if online_summary is not None and online_summary.is_file():
        roll_raw = pd.read_csv(online_summary)
        if "primitive_id" not in roll_raw.columns:
            raise ValueError(f"{online_summary} has no primitive_id column.")
        roll_raw["primitive_id"] = roll_raw["primitive_id"].astype(str)
        if "mean_onset_f1" in roll_raw.columns:
            optional_cols.append("mean_onset_f1")
        if "median_onset_f1" in roll_raw.columns:
            optional_cols.append("median_onset_f1")
        if optional_cols:
            sub = roll_raw[["primitive_id", *optional_cols]].drop_duplicates(subset=["primitive_id"], keep="last")
            merged = merged.merge(sub, on="primitive_id", how="left")

    roll_mean = (
        pd.to_numeric(merged["mean_onset_f1"], errors="coerce")
        if "mean_onset_f1" in merged.columns
        else pd.Series(np.nan, index=merged.index)
    )
    roll_med = (
        pd.to_numeric(merged["median_onset_f1"], errors="coerce")
        if "median_onset_f1" in merged.columns
        else pd.Series(np.nan, index=merged.index)
    )

    merged["assignment_confidence_mean"] = merged["assignment_confidence_mean"].astype(np.float64)
    merged["weighted_strike_error"] = merged["weighted_strike_error"].astype(np.float64)

    strikes = merged["weighted_strike_error"].to_numpy(dtype=np.float64)
    smin = float(np.min(strikes))
    smax = float(np.max(strikes))
    denom = max(smax - smin, 1e-9)
    norm_strike = (strikes - smin) / denom
    confidence = merged["assignment_confidence_mean"].to_numpy(dtype=np.float64)

    heuristic = np.clip(confidence * (1.0 - 0.85 * norm_strike), 0.0, 1.0)

    coalesced = _coalesce_online_onset(roll_mean, roll_med)
    onset = coalesced.to_numpy(dtype=np.float64)

    blended = heuristic.copy()
    mask = np.isfinite(onset) & (~np.isnan(onset))
    blended[mask] = np.clip(onset[mask], 0.0, 1.0)

    return pd.DataFrame(
        {
            "primitive_id": merged["primitive_id"].astype(str).tolist(),
            "onset_f1": blended.astype(np.float64),
            "key_press": True,
            "target_hit": True,
        }
    )


def pick_online_primitive_summary(primitive_root: Path) -> Path | None:
    for candidate in discover_online_primitive_summaries(primitive_root):
        if candidate.is_file():
            _LOGGER.info("Stage 2 primitive quality merge: using online summary %s", candidate)
            return candidate
    _LOGGER.warning(
        "No primitive_summary_metrics.csv discovered near %s; building weak-primitive table from Stage 1 library stats only.",
        primitive_root.resolve(),
    )
    return None


def write_stage2_primitive_quality_csv(
    primitive_root: Path,
    *,
    online_summary: Path | None | str = "__auto__",
    output_path: Path | None = None,
) -> Path:
    root = primitive_root.resolve()
    if online_summary == "__auto__":
        online_summary = pick_online_primitive_summary(root)
    elif isinstance(online_summary, str):
        online_summary = Path(online_summary)

    metrics_dir = root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    destination = metrics_dir / "stage2_primitive_quality.csv" if output_path is None else Path(output_path)
    df = build_stage2_quality_table_df(primitive_root=root, online_summary=online_summary)
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination, index=False)
    _LOGGER.info("Wrote Stage 2 primitive quality table (%d primitives) → %s", len(df), destination)
    return destination.resolve()


def should_materialize_weak_primitive_table(primitive_root: Path, pf_cfg: dict[str, Any]) -> bool:
    """True when weak-primitive dropout (downweight default) expects the staged quality CSV."""

    if not bool(pf_cfg.get("enabled", False)):
        return False
    mode = str(pf_cfg.get("mode", "none")).lower().strip()
    if mode != "downweight":
        return False
    qpath = pf_cfg.get("quality_metrics_path")
    if qpath is None:
        return True
    text = str(qpath).strip()
    if not text:
        return True
    qp = Path(text)
    if qp.is_absolute():
        default_abs = (primitive_root.resolve() / DEFAULT_STAGE2_QUALITY_REL).resolve()
        return qp.resolve() == default_abs
    rel = qp.as_posix().replace("\\", "/")
    return rel == DEFAULT_STAGE2_QUALITY_REL.as_posix()


def prepare_weak_primitive_dropout_artifacts(primitive_root: Path, pf_cfg: dict[str, Any]) -> None:
    """Write ``metrics/stage2_primitive_quality.csv`` before loading quality into the planner."""

    if should_materialize_weak_primitive_table(primitive_root, pf_cfg):
        write_stage2_primitive_quality_csv(primitive_root.resolve(), online_summary="__auto__")
