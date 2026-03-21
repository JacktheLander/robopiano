from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd


def add_token_columns(assignments_df: pd.DataFrame, num_duration_buckets: int, num_dynamics_buckets: int) -> pd.DataFrame:
    frame = assignments_df.copy()
    train_mask = frame["split"] == "train"
    frame["duration_bucket"] = bucketize(frame["duration_steps"].to_numpy(dtype=np.float32), train_mask.to_numpy(), num_duration_buckets)
    frame["dynamics_bucket"] = bucketize(frame["motion_energy"].to_numpy(dtype=np.float32), train_mask.to_numpy(), num_dynamics_buckets)
    primitive_names = sorted(frame["primitive_id"].astype(str).unique().tolist())
    primitive_to_index = {name: idx for idx, name in enumerate(primitive_names)}
    frame["primitive_index"] = frame["primitive_id"].map(primitive_to_index).astype(int)
    frame["token_json"] = frame.apply(
        lambda row: json.dumps(
            {
                "primitive_id": row["primitive_id"],
                "primitive_index": int(row["primitive_index"]),
                "duration_bucket": int(row["duration_bucket"]),
                "dynamics_bucket": int(row["dynamics_bucket"]),
                "score_context": json.loads(row["score_context_json"]),
            },
            sort_keys=True,
        ),
        axis=1,
    )
    return frame


def bucketize(values: np.ndarray, train_mask: np.ndarray, num_buckets: int) -> np.ndarray:
    if num_buckets <= 1:
        return np.zeros((len(values),), dtype=np.int64)
    source = values[train_mask] if np.any(train_mask) else values
    quantiles = np.linspace(0.0, 1.0, num_buckets + 1)[1:-1]
    boundaries = np.quantile(source, quantiles).astype(np.float32)
    return np.digitize(values, boundaries, right=False).astype(np.int64)


def build_vocabulary_payload(token_df: pd.DataFrame) -> dict[str, Any]:
    primitive_ids = sorted(token_df["primitive_id"].astype(str).unique().tolist())
    duration_buckets = sorted(token_df["duration_bucket"].astype(int).unique().tolist())
    dynamics_buckets = sorted(token_df["dynamics_bucket"].astype(int).unique().tolist())
    return {
        "num_primitives": len(primitive_ids),
        "primitive_ids": primitive_ids,
        "duration_buckets": duration_buckets,
        "dynamics_buckets": dynamics_buckets,
        "num_duration_buckets": (max(duration_buckets) + 1) if duration_buckets else 1,
        "num_dynamics_buckets": (max(dynamics_buckets) + 1) if dynamics_buckets else 1,
    }
