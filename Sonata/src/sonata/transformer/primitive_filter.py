from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sonata.transformer.dataset import PlannerMetadata
from sonata.utils.io import read_table

_LOGGER = logging.getLogger(__name__)

_ID_COLUMNS = ["primitive_id", "primitive", "primitive_idx", "cluster_id", "token_id"]
_ONSET_F1_COLUMNS = ["onset_f1", "mean_onset_f1", "median_onset_f1", "online_onset_f1", "rollout_onset_f1", "f1", "strong_f1_score"]
_KEY_PRESS_COLUMNS = ["key_press", "has_key_press", "produces_key_press", "key_press_success"]
_TARGET_HIT_COLUMNS = ["target_hit", "hits_target", "target_success", "has_target_hit"]
_STRONG_F1_COLUMNS = ["strong_f1", "is_strong_f1", "strong"]

_QUALITY_NAME_HINTS = (
    "onset_f1",
    "online_eval",
    "primitive_online",
    "rollout",
    "primitive_rollout",
    "primitive_summary",
)


def _digits_suffix_int(value: str) -> int | None:
    match = re.search(r"(\d+)$", value.strip())
    if match:
        return int(match.group(1))
    return None


def coerce_primitive_table_id(value: object) -> int | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    s = str(value).strip()
    if not s:
        return None
    try:
        if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
            return int(s)
        if s.lower() in {"true", "false"}:
            return None
        as_float = float(s)
        if np.isfinite(as_float) and as_float == int(as_float):
            return int(as_float)
    except (ValueError, TypeError):
        pass
    return _digits_suffix_int(s)


def canonical_id_for_vocab_entry(primitive_id: str | int) -> int | None:
    return coerce_primitive_table_id(primitive_id)


def bool_or_null(series: pd.Series) -> pd.Series:
    def cell(v: object) -> bool | None:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        if isinstance(v, bool):
            return v
        if isinstance(v, (np.bool_,)):
            return bool(v)
        s = str(v).strip().lower()
        if s in {"", "nan", "none"}:
            return None
        if s in {"1", "true", "yes", "y"}:
            return True
        if s in {"0", "false", "no", "n"}:
            return False
        try:
            x = float(s)
            if x == 1.0:
                return True
            if x == 0.0:
                return False
        except ValueError:
            pass
        return None

    return series.map(cell)


def _pick_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in frame.columns}
    for name in candidates:
        if name in frame.columns:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def _table_quality_score(frame: pd.DataFrame) -> int:
    score = 0
    if _pick_column(frame, _ID_COLUMNS):
        score += 10
    if _pick_column(frame, _ONSET_F1_COLUMNS):
        score += 5
    for group in (_KEY_PRESS_COLUMNS, _TARGET_HIT_COLUMNS, _STRONG_F1_COLUMNS):
        if _pick_column(frame, group):
            score += 2
    path_lower = "".join(frame.columns).lower()
    for hint in _QUALITY_NAME_HINTS:
        if hint in path_lower:
            score += 0
            break
    return score


def _collect_candidate_paths(primitive_root: Path, explicit: Path | None) -> list[Path]:
    ordered: list[Path] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        key = str(path.resolve())
        if key in seen:
            return
        seen.add(key)
        if path.exists() and path.is_file():
            ordered.append(path)

    if explicit is not None:
        add(explicit)
        return ordered

    metrics_dir = primitive_root / "metrics"
    library_dir = primitive_root / "library"
    for name in (
        "online_primitive_eval.csv",
        "primitive_online_eval.csv",
        "primitive_rollout_metrics.csv",
        "online_eval_summary.csv",
    ):
        add(metrics_dir / name)
    for name in ("primitive_summary.csv", "primitive_library.csv"):
        add(library_dir / name)
    if metrics_dir.is_dir():
        for path in sorted(metrics_dir.glob("*.csv")):
            add(path)
    if library_dir.is_dir():
        for path in sorted(library_dir.glob("*.csv")):
            add(path)
    return ordered


def _read_csv_flexible(path: Path) -> pd.DataFrame | None:
    try:
        suffix = path.suffix.lower()
        if suffix in {".csv", ".parquet"}:
            base = path.parent / path.stem
            return read_table(base)
        return read_table(path)
    except Exception as exc:
        _LOGGER.debug("Skipping unreadable metrics file %s: %s", path, exc)
        return None


def _normalize_quality_frame(
    best_df: pd.DataFrame,
    *,
    id_col: str,
    onset_col: str | None,
    kp_col: str | None,
    th_col: str | None,
    sf_col: str | None,
) -> pd.DataFrame:
    out = pd.DataFrame()
    mapped_ids: list[int | None] = [coerce_primitive_table_id(v) for v in best_df[id_col].tolist()]
    out["primitive_id"] = pd.array(mapped_ids, dtype=object)

    if onset_col:
        onset = pd.to_numeric(best_df[onset_col], errors="coerce").astype(np.float64)
        out["onset_f1"] = onset.values
    else:
        out["onset_f1"] = np.nan

    for name, col in [("key_press", kp_col), ("target_hit", th_col), ("strong_f1", sf_col)]:
        if col:
            out[name] = bool_or_null(best_df[col]).tolist()
        else:
            out[name] = [None] * len(best_df)

    statuses = [row_quality_status(row) for row in out.itertuples(index=False)]
    out["quality_status"] = statuses

    out = out[out["primitive_id"].notna()].copy()
    out["primitive_id"] = out["primitive_id"].astype(int)
    out = out.drop_duplicates(subset=["primitive_id"], keep="last")
    return out


def load_primitive_quality_table(
    primitive_root: Path | str,
    quality_metrics_path: str | None = None,
) -> tuple[pd.DataFrame | None, Path | None, str | None]:
    """Load and normalize primitive quality metrics.

    Returns (normalized_df_or_none, resolved_path_or_none, warning_message_or_none).
    """
    root = Path(primitive_root).resolve()
    explicit: Path | None = None
    if quality_metrics_path:
        explicit = Path(quality_metrics_path).expanduser()
        if not explicit.is_absolute():
            explicit = root / explicit
        explicit = explicit.resolve()
        if not explicit.exists():
            raise FileNotFoundError(f"primitive_filter quality_metrics_path does not exist: {explicit}")
        raw = _read_csv_flexible(explicit)
        if raw is None or raw.empty:
            raise FileNotFoundError(f"primitive_filter quality_metrics_path is unreadable or empty: {explicit}")
        id_col = _pick_column(raw, _ID_COLUMNS)
        if id_col is None:
            raise ValueError(f"No recognizable primitive ID column {_ID_COLUMNS} in {explicit}")
        onset_col = _pick_column(raw, _ONSET_F1_COLUMNS)
        kp_col = _pick_column(raw, _KEY_PRESS_COLUMNS)
        th_col = _pick_column(raw, _TARGET_HIT_COLUMNS)
        sf_col = _pick_column(raw, _STRONG_F1_COLUMNS)
        if onset_col is None:
            _LOGGER.warning(
                "%s has no recognizable onset/F1 column; disabling primitive quality filtering.", explicit
            )
            return None, explicit, "no_onset_metric"
        framed = _normalize_quality_frame(raw, id_col=id_col, onset_col=onset_col, kp_col=kp_col, th_col=th_col, sf_col=sf_col)
        return framed, explicit, None

    candidates = _collect_candidate_paths(root, None)
    best_path: Path | None = None
    best_df: pd.DataFrame | None = None
    best_score = -1

    for path in candidates:
        df = _read_csv_flexible(path)
        if df is None or df.empty:
            continue
        score = _table_quality_score(df)
        if score > best_score:
            best_score = score
            best_path = path
            best_df = df.copy()

    if best_df is None or best_score < 10:
        return None, best_path, "no_metrics_file"

    id_col = _pick_column(best_df, _ID_COLUMNS)
    if id_col is None:
        return None, best_path, "no_metrics_file"

    onset_col = _pick_column(best_df, _ONSET_F1_COLUMNS)
    kp_col = _pick_column(best_df, _KEY_PRESS_COLUMNS)
    th_col = _pick_column(best_df, _TARGET_HIT_COLUMNS)
    sf_col = _pick_column(best_df, _STRONG_F1_COLUMNS)

    if onset_col is None:
        _LOGGER.warning(
            "primitive quality CSV at %s has no recognizable onset/F1 column; ignoring for filtering.",
            best_path,
        )
        return None, best_path, "no_onset_metric"

    out = _normalize_quality_frame(best_df, id_col=id_col, onset_col=onset_col, kp_col=kp_col, th_col=th_col, sf_col=sf_col)

    return out, best_path, None


def row_quality_status(row: Any) -> str:
    oid = getattr(row, "primitive_id", None)
    if oid is None or (isinstance(oid, float) and np.isnan(oid)):
        return "invalid_id"
    return "ok"


def build_primitive_row_flags(
    row: pd.Series,
    *,
    min_onset_f1: float,
    strong_onset_f1: float,
    require_key_press: bool,
    require_target_hit: bool,
    require_strong_f1: bool,
) -> tuple[bool, bool, bool, bool]:
    onset = float(row["onset_f1"]) if pd.notna(row.get("onset_f1")) else None
    key_press = row.get("key_press")
    target_hit = row.get("target_hit")
    strong_f1 = row.get("strong_f1")

    rules_bad = False
    if onset is not None and onset < min_onset_f1:
        rules_bad = True
    if require_key_press and key_press is False:
        rules_bad = True
    if require_target_hit and target_hit is False:
        rules_bad = True
    if require_strong_f1 and strong_f1 is False:
        rules_bad = True

    is_strong = onset is not None and onset >= strong_onset_f1
    is_borderline = (
        onset is not None and not rules_bad and (min_onset_f1 <= onset < strong_onset_f1)
    )
    return rules_bad, (not rules_bad), is_borderline if not rules_bad else False, is_strong if not rules_bad else False


def vocab_indices_for_bad_canonical(
    metadata: PlannerMetadata,
    bad_canonical: set[int],
) -> frozenset[int]:
    bad_vocab: set[int] = set()
    for idx, pid in enumerate(metadata.primitive_ids):
        c = canonical_id_for_vocab_entry(pid)
        if c is not None and c in bad_canonical:
            bad_vocab.add(idx)
    return frozenset(bad_vocab)


def finish_build_filter_after_vocab_mapping(
    *,
    canon_flags: dict[int, dict[str, Any]],
    valid_ids: set[int],
    bad_ids: set[int],
    borderline_ids: set[int],
    strong_ids: set[int],
    all_canonical_ids: list[int],
    metadata: PlannerMetadata,
    diagnostics: dict[str, Any],
    bad_w: float,
    border_w: float,
    mode: str,
) -> dict[str, Any]:
    vocab_weights: dict[int, float] = {}
    vocab_bad_indices: set[int] = set()
    bad_canonical = set(bad_ids)

    for idx, vocab_entry in enumerate(metadata.primitive_ids):
        c = canonical_id_for_vocab_entry(vocab_entry)
        if c is None:
            vocab_weights[idx] = 1.0
            continue
        if c in bad_canonical:
            vocab_bad_indices.add(idx)
            vocab_weights[idx] = bad_w if mode == "downweight" else 1.0
            continue
        entry = canon_flags.get(c, {})
        if entry.get("is_borderline"):
            vocab_weights[idx] = border_w
        elif entry.get("quality_status") == "missing_quality":
            vocab_weights[idx] = 1.0
        else:
            vocab_weights[idx] = 1.0

    diagnostics["vocab_bad_index_count"] = len(vocab_bad_indices)

    return {
        "valid_primitive_ids": set(valid_ids),
        "bad_primitive_ids": set(bad_ids),
        "borderline_primitive_ids": set(borderline_ids),
        "strong_primitive_ids": set(strong_ids),
        "primitive_vocab_weights": vocab_weights,
        "vocab_bad_indices": frozenset(vocab_bad_indices),
        "canon_flags": canon_flags,
        "all_canonical_ids": list(all_canonical_ids),
        "diagnostics": diagnostics,
    }


def build_primitive_filter(
    quality_df: pd.DataFrame | None,
    config: dict[str, Any],
    metadata: PlannerMetadata,
) -> dict[str, Any]:
    """Build filter structures using vocabulary order from PlannerMetadata."""

    pf = dict(config.get("primitive_filter") or {})
    enabled = bool(pf.get("enabled", False))
    mode = str(pf.get("mode", "none")).lower()
    diagnostics: dict[str, Any] = {
        "config_enabled_requested": enabled,
        "effective_mode": mode,
        "quality_rows": int(len(quality_df)) if quality_df is not None else 0,
    }

    if not enabled or mode in {"", "none"}:
        diagnostics["skipped"] = True
        diagnostics["reason"] = "disabled"
        diagnostics["effective_mode"] = "none"
        diagnostics["effective_enabled"] = False
        diagnostics["total_primitives"] = int(metadata.num_primitives)
        _, ordered = canonical_alignment(metadata)
        return finish_build_filter_after_vocab_mapping(
            canon_flags={},
            valid_ids=set(ordered),
            bad_ids=set(),
            borderline_ids=set(),
            strong_ids=set(),
            all_canonical_ids=ordered,
            metadata=metadata,
            diagnostics=diagnostics,
            bad_w=float(pf.get("bad_primitive_weight", 0.0)),
            border_w=float(pf.get("borderline_weight", 0.5)),
            mode="none",
        )

    if quality_df is None or quality_df.empty:
        _LOGGER.warning(
            "primitive_filter.enabled=true but no online primitive quality table was found; proceeding without filtering"
        )
        config.setdefault("primitive_filter", {})
        config["primitive_filter"]["enabled"] = False
        diagnostics["effective_enabled"] = False
        diagnostics["effective_mode"] = "none"
        diagnostics["filter_disable_reason"] = "no_quality_table"
        diagnostics["skipped"] = True
        diagnostics["reason"] = "no_quality_table_auto_disabled"
        diagnostics["total_primitives"] = int(metadata.num_primitives)
        _, ordered = canonical_alignment(metadata)
        return finish_build_filter_after_vocab_mapping(
            canon_flags={},
            valid_ids=set(ordered),
            bad_ids=set(),
            borderline_ids=set(),
            strong_ids=set(),
            all_canonical_ids=ordered,
            metadata=metadata,
            diagnostics=diagnostics,
            bad_w=float(pf.get("bad_primitive_weight", 0.0)),
            border_w=float(pf.get("borderline_weight", 0.5)),
            mode="none",
        )

    mode_norm_raw = mode if mode in {"drop", "downweight"} else "none"
    diagnostics["effective_mode"] = mode_norm_raw

    min_onset = float(pf.get("min_onset_f1", 0.4))
    strong_thr = float(pf.get("strong_onset_f1", 0.6))
    require_kp = bool(pf.get("require_key_press", False))
    require_th = bool(pf.get("require_target_hit", False))
    require_sf = bool(pf.get("require_strong_f1", False))
    bad_w = float(pf.get("bad_primitive_weight", 0.0))
    border_w = float(pf.get("borderline_weight", 0.5))

    per_vocab_canonical, ordered_canonical = canonical_alignment(metadata)
    canon_population = sorted({c for c in per_vocab_canonical if c is not None})

    missing_extra = sorted(int(x) for x in quality_df["primitive_id"].unique().tolist() if int(x) not in canon_population)
    if missing_extra:
        _LOGGER.warning(
            "primitive quality table contains %d IDs not present in planner vocabulary (e.g. %s); ignoring those rows.",
            len(missing_extra),
            missing_extra[:5],
        )

    ql = quality_df.set_index("primitive_id")

    valid_ids: set[int] = set()
    bad_ids: set[int] = set()
    borderline_ids: set[int] = set()
    strong_ids: set[int] = set()
    canon_flags: dict[int, dict[str, Any]] = {}
    missing_quality = 0
    unmappable = 0

    for vocab_idx, raw_vocab in enumerate(metadata.primitive_ids):
        c_int = per_vocab_canonical[vocab_idx]
        if c_int is None:
            unmappable += 1
            _LOGGER.warning(
                "Vocabulary primitive %r (index %d) cannot be matched to canonical integer IDs in the "
                "quality table; treating as conservative valid (training targets kept).",
                raw_vocab,
                vocab_idx,
            )
            continue

        pid = int(c_int)
        if pid not in ql.index:
            missing_quality += 1
            valid_ids.add(pid)
            canon_flags[pid] = {
                "onset_f1": np.nan,
                "key_press": None,
                "target_hit": None,
                "strong_f1": None,
                "quality_status": "missing_quality",
                "is_valid": True,
                "is_bad": False,
                "is_borderline": False,
                "is_strong": False,
            }
            continue

        row_series = ql.loc[pid]
        if isinstance(row_series, pd.DataFrame):
            row_series = row_series.iloc[-1]

        qb, _, border, strong = build_primitive_row_flags(
            row_series,
            min_onset_f1=min_onset,
            strong_onset_f1=strong_thr,
            require_key_press=require_kp,
            require_target_hit=require_th,
            require_strong_f1=require_sf,
        )

        if qb:
            bad_ids.add(pid)
            canon_flags[pid] = {
                "onset_f1": float(row_series["onset_f1"]) if pd.notna(row_series.get("onset_f1")) else np.nan,
                "key_press": row_series.get("key_press"),
                "target_hit": row_series.get("target_hit"),
                "strong_f1": row_series.get("strong_f1"),
                "quality_status": "bad",
                "is_valid": False,
                "is_bad": True,
                "is_borderline": False,
                "is_strong": False,
            }
            continue

        valid_ids.add(pid)
        canon_flags[pid] = {
            "onset_f1": float(row_series["onset_f1"]) if pd.notna(row_series.get("onset_f1")) else np.nan,
            "key_press": row_series.get("key_press"),
            "target_hit": row_series.get("target_hit"),
            "strong_f1": row_series.get("strong_f1"),
            "quality_status": "ok",
            "is_valid": True,
            "is_bad": False,
            "is_borderline": bool(border),
            "is_strong": bool(strong),
        }
        if border:
            borderline_ids.add(pid)
        if strong:
            strong_ids.add(pid)

    diagnostics.update(
        {
            "effective_enabled": True,
            "min_onset_f1": min_onset,
            "strong_onset_f1": strong_thr,
            "require_key_press": require_kp,
            "require_target_hit": require_th,
            "require_strong_f1": require_sf,
            "total_primitives": int(metadata.num_primitives),
            "valid_count": len(valid_ids),
            "bad_count": len(bad_ids),
            "borderline_count": len(borderline_ids),
            "strong_count": len(strong_ids),
            "missing_quality_count": missing_quality,
            "unmappable_vocab_ids": unmappable,
        }
    )

    mode_norm_final = mode_norm_raw if mode_norm_raw in {"drop", "downweight"} else "none"
    diagnostics["effective_mode"] = mode_norm_final

    return finish_build_filter_after_vocab_mapping(
        canon_flags=canon_flags,
        valid_ids=valid_ids,
        bad_ids=bad_ids,
        borderline_ids=borderline_ids,
        strong_ids=strong_ids,
        all_canonical_ids=ordered_canonical,
        metadata=metadata,
        diagnostics=diagnostics,
        bad_w=bad_w,
        border_w=border_w,
        mode=mode_norm_final,
    )


def canonical_alignment(metadata: PlannerMetadata) -> tuple[list[int | None], list[int]]:
    per_vocab: list[int | None] = []
    for pid in metadata.primitive_ids:
        per_vocab.append(canonical_id_for_vocab_entry(pid))
    ordered_canon: list[int] = []
    seen: set[int] = set()
    for candidate in per_vocab:
        if candidate is None:
            continue
        key = int(candidate)
        if key not in seen:
            seen.add(key)
            ordered_canon.append(key)
    return per_vocab, ordered_canon


def build_filter_summary_json(
    *,
    filter_cfg: dict[str, Any],
    resolved_metrics_path: str | None,
    diagnostics: dict[str, Any],
    train_before: int,
    train_after: int,
    artifacts_dir: str,
) -> dict[str, Any]:
    dropped = max(0, train_before - train_after)
    frac = (dropped / train_before) if train_before > 0 else 0.0
    enabled = bool(filter_cfg.get("enabled", False))

    summary = {
        "enabled": enabled,
        "effective_enabled": diagnostics.get("effective_enabled", enabled),
        "mode": str(filter_cfg.get("mode", "none")),
        "effective_mode": diagnostics.get("effective_mode", filter_cfg.get("mode")),
        "min_onset_f1": filter_cfg.get("min_onset_f1"),
        "strong_onset_f1": filter_cfg.get("strong_onset_f1"),
        "require_key_press": filter_cfg.get("require_key_press"),
        "require_target_hit": filter_cfg.get("require_target_hit"),
        "require_strong_f1": filter_cfg.get("require_strong_f1"),
        "total_primitives": diagnostics.get("total_primitives"),
        "valid_count": diagnostics.get("valid_count"),
        "bad_count": diagnostics.get("bad_count"),
        "borderline_count": diagnostics.get("borderline_count"),
        "strong_count": diagnostics.get("strong_count"),
        "missing_quality_count": diagnostics.get("missing_quality_count"),
        "train_examples_before": train_before,
        "train_examples_after": train_after,
        "train_examples_dropped": dropped,
        "drop_fraction": float(frac),
        "metrics_path": resolved_metrics_path,
        "artifacts_dir": artifacts_dir,
        "diagnostics": {k: v for k, v in diagnostics.items() if k not in {"effective_mode"}},
    }
    return summary


def write_primitive_filter_table_csv(
    path: Path,
    *,
    metadata: PlannerMetadata,
    canon_flags: dict[int, dict[str, Any]],
    ordered_canonical: list[int],
    count_before: dict[int, int],
    count_after: dict[int, int],
) -> None:
    rows: list[dict[str, Any]] = []
    for pid in ordered_canonical:
        flags = canon_flags.get(
            pid,
            {
                "onset_f1": np.nan,
                "key_press": None,
                "target_hit": None,
                "strong_f1": None,
                "quality_status": "missing_quality",
                "is_valid": True,
                "is_bad": False,
                "is_borderline": False,
                "is_strong": False,
            },
        )
        rows.append(
            {
                "primitive_id": int(pid),
                "onset_f1": flags.get("onset_f1", np.nan),
                "key_press": flags.get("key_press"),
                "target_hit": flags.get("target_hit"),
                "strong_f1": flags.get("strong_f1"),
                "quality_status": flags.get("quality_status"),
                "is_valid": flags.get("is_valid"),
                "is_bad": flags.get("is_bad"),
                "is_borderline": flags.get("is_borderline"),
                "is_strong": flags.get("is_strong"),
                "train_target_count_before": int(count_before.get(pid, 0)),
                "train_target_count_after": int(count_after.get(pid, 0)),
            }
        )
    frame = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def write_id_list(path: Path, ids: set[int] | frozenset[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(x) for x in sorted(ids)) + ("\n" if ids else ""), encoding="utf-8")


def write_family_overlay_csv(path: Path, *, metadata: PlannerMetadata, canon_flags: dict[int, dict[str, Any]]) -> None:
    rows = []
    for idx, primitive_id_vocab in enumerate(metadata.primitive_ids):
        c = canonical_id_for_vocab_entry(primitive_id_vocab)
        flags = canon_flags.get(c, {}) if c is not None else {}
        fname = metadata.primitive_family_names[metadata.primitive_to_family[idx]]
        rows.append(
            {
                "primitive_index": idx,
                "primitive_id_vocab": str(primitive_id_vocab),
                "canonical_primitive_id": c,
                "primitive_family_index": metadata.primitive_to_family[idx],
                "primitive_family_name": fname,
                "quality_status": flags.get("quality_status", "missing_quality"),
                "is_valid_training_target": not flags.get("is_bad", False),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)
