from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sonata.utils.io import read_json, read_table


TOKEN_FEATURES = (
    "duration_steps",
    "motion_energy",
    "chord_size",
    "key_center",
    "start_state_norm",
    "end_state_norm",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a weak-to-strong primitive remap for Sonata Stage 2.")
    parser.add_argument("--primitive-root", required=True)
    parser.add_argument("--metrics", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--min-frequency", type=int, default=50)
    parser.add_argument("--max-weak-f1", type=float, default=0.25)
    parser.add_argument("--same-family-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-similarity", type=float, default=0.70)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    primitive_root = Path(args.primitive_root).expanduser().resolve()
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output is not None
        else primitive_root / "artifacts" / "primitive_remap.json"
    )

    token_df = read_table(primitive_root / "tokens" / "primitive_tokens")
    vocabulary = read_json(primitive_root / "tokens" / "primitive_vocabulary.json")
    primitive_ids = [str(item) for item in vocabulary.get("primitive_ids", [])]
    if not primitive_ids:
        primitive_ids = sorted(token_df["primitive_id"].astype(str).unique().tolist())
    id_to_index = {primitive_id: index for index, primitive_id in enumerate(primitive_ids)}
    num_primitives = int(vocabulary.get("num_primitives", len(primitive_ids)))

    frequencies = token_df["primitive_index"].astype(int).value_counts().reindex(range(num_primitives), fill_value=0)
    metrics_df = _read_optional_table(Path(args.metrics).expanduser()) if args.metrics else None
    metric_scores = _metric_scores(metrics_df, primitive_ids, id_to_index) if metrics_df is not None else {}
    family_by_index = _family_by_index(token_df, primitive_ids, id_to_index)
    features = _primitive_features(primitive_root, token_df, primitive_ids)
    similarity = _cosine_similarity(_standardize(features))

    weak: set[int] = set()
    for index in range(num_primitives):
        f1 = metric_scores.get(index)
        if int(frequencies.get(index, 0)) < int(args.min_frequency) or (f1 is not None and float(f1) < float(args.max_weak_f1)):
            weak.add(index)
    strong = {
        index
        for index in range(num_primitives)
        if index not in weak and int(frequencies.get(index, 0)) >= int(args.min_frequency)
    }

    remap: dict[str, str] = {}
    reasons: dict[str, str] = {}
    report_rows: list[dict[str, Any]] = []
    for weak_index in sorted(weak):
        candidates = [index for index in sorted(strong) if index != weak_index]
        if bool(args.same_family_only) and family_by_index:
            same_family = [
                index
                for index in candidates
                if family_by_index.get(index) is not None and family_by_index.get(index) == family_by_index.get(weak_index)
            ]
            candidates = same_family
        best_index: int | None = None
        best_similarity = -1.0
        for candidate in candidates:
            score = float(similarity[weak_index, candidate])
            if score > best_similarity:
                best_similarity = score
                best_index = candidate
        mapped = best_index is not None and best_similarity >= float(args.min_similarity)
        weak_id = primitive_ids[weak_index] if weak_index < len(primitive_ids) else str(weak_index)
        strong_id = primitive_ids[best_index] if mapped and best_index is not None and best_index < len(primitive_ids) else ""
        if mapped and best_index is not None:
            remap[weak_id] = strong_id
            reasons[weak_id] = (
                f"freq={int(frequencies.get(weak_index, 0))};"
                f"metric={metric_scores.get(weak_index, 'na')};"
                f"similar_to={strong_id};similarity={best_similarity:.4f}"
            )
        report_rows.append(
            {
                "primitive_index": weak_index,
                "primitive_id": weak_id,
                "frequency": int(frequencies.get(weak_index, 0)),
                "metric_score": metric_scores.get(weak_index),
                "family": family_by_index.get(weak_index),
                "mapped": bool(mapped),
                "target_primitive_index": int(best_index) if mapped and best_index is not None else None,
                "target_primitive_id": strong_id if mapped else None,
                "similarity": best_similarity if best_index is not None else None,
            }
        )

    payload = {
        "enabled": True,
        "mode": "weak_to_strong",
        "same_family_only": bool(args.same_family_only),
        "min_similarity": float(args.min_similarity),
        "remap": remap,
        "reason": reasons,
    }

    report = pd.DataFrame(report_rows)
    report_path = output_path.with_name("primitive_remap_report.csv")
    if not args.dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        report.to_csv(report_path, index=False)

    print(f"num primitives: {num_primitives}")
    print(f"num weak: {len(weak)}")
    print(f"num mapped: {len(remap)}")
    print(f"num unmapped: {len(weak) - len(remap)}")
    if args.dry_run:
        print(f"dry run: would write {output_path} and {report_path}")
    else:
        print(f"wrote {output_path}")
        print(f"wrote {report_path}")


def _read_optional_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return read_table(path)


def _metric_scores(metrics_df: pd.DataFrame, primitive_ids: list[str], id_to_index: dict[str, int]) -> dict[int, float]:
    score_column = next((name for name in ("f1", "weighted_f1", "success") if name in metrics_df.columns), None)
    if score_column is None and "mispress" in metrics_df.columns:
        scores = 1.0 - metrics_df["mispress"].astype(float)
    elif score_column is not None:
        scores = metrics_df[score_column].astype(float)
    else:
        return {}
    output: dict[int, float] = {}
    for row_index, row in metrics_df.iterrows():
        primitive_index: int | None = None
        if "primitive_index" in metrics_df.columns and not pd.isna(row["primitive_index"]):
            primitive_index = int(row["primitive_index"])
        elif "primitive_id" in metrics_df.columns:
            primitive_index = id_to_index.get(str(row["primitive_id"]))
        elif row_index < len(primitive_ids):
            primitive_index = int(row_index)
        if primitive_index is not None:
            output[int(primitive_index)] = float(scores.loc[row_index])
    return output


def _family_by_index(token_df: pd.DataFrame, primitive_ids: list[str], id_to_index: dict[str, int]) -> dict[int, str]:
    family_column = next(
        (name for name in ("primitive_family", "primitive_family_index", "heuristic_family") if name in token_df.columns),
        None,
    )
    if family_column is None:
        return {}
    frame = token_df[["primitive_id", family_column]].copy()
    frame["primitive_id"] = frame["primitive_id"].astype(str)
    grouped = frame.groupby("primitive_id", sort=True)[family_column].agg(_dominant_value)
    return {
        int(id_to_index[primitive_id]): str(grouped.get(primitive_id))
        for primitive_id in primitive_ids
        if primitive_id in id_to_index and primitive_id in grouped
    }


def _primitive_features(primitive_root: Path, token_df: pd.DataFrame, primitive_ids: list[str]) -> np.ndarray:
    library_path = primitive_root / "library" / "primitive_library"
    library_df = None
    if library_path.with_suffix(".parquet").exists() or library_path.with_suffix(".csv").exists():
        library_df = read_table(library_path)
    if library_df is not None and not library_df.empty and "primitive_id" in library_df.columns:
        library = library_df.copy()
        library["primitive_id"] = library["primitive_id"].astype(str)
        numeric_columns = [
            column
            for column in library.select_dtypes(include=[np.number]).columns
            if column not in {"primitive_index", "index"}
        ]
        if numeric_columns:
            indexed = library.set_index("primitive_id")
            return indexed.reindex(primitive_ids)[numeric_columns].fillna(0.0).to_numpy(dtype=np.float32)

    available = [column for column in TOKEN_FEATURES if column in token_df.columns]
    if not available:
        return np.eye(len(primitive_ids), dtype=np.float32)
    frame = token_df.copy()
    frame["primitive_id"] = frame["primitive_id"].astype(str)
    grouped = frame.groupby("primitive_id", sort=True)[available].mean()
    return grouped.reindex(primitive_ids).fillna(0.0).to_numpy(dtype=np.float32)


def _standardize(features: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(features.astype(np.float32))
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    return (values - mean) / np.clip(std, 1e-6, None)


def _cosine_similarity(features: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    normalized = features / np.clip(norm, 1e-6, None)
    return np.matmul(normalized, normalized.T)


def _dominant_value(values: pd.Series) -> Any:
    counts = values.astype(str).value_counts()
    if counts.empty:
        return None
    max_count = int(counts.iloc[0])
    winners = sorted(item for item, count in counts.items() if int(count) == max_count)
    return winners[0] if winners else None


if __name__ == "__main__":
    main()
