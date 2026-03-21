from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sonata.transformer.dataset import PlannerMetadata, planner_collate_fn
from sonata.transformer.model import PrimitivePlannerTransformer


def _sample(history: list[int], duration: list[int], dynamics: list[int]) -> dict[str, object]:
    score_dim = 20
    return {
        "primitive_history": np.asarray(history, dtype=np.int64),
        "duration_history": np.asarray(duration, dtype=np.int64),
        "dynamics_history": np.asarray(dynamics, dtype=np.int64),
        "score_history": np.ones((len(history), score_dim), dtype=np.float32),
        "target_primitive": history[-1],
        "target_duration": duration[-1],
        "target_dynamics": dynamics[-1],
    }


def test_planner_collate_right_padding_keeps_finite_plan_embeddings() -> None:
    metadata = PlannerMetadata(
        num_primitives=8,
        num_duration_buckets=7,
        num_dynamics_buckets=6,
        score_dim=20,
        pad_primitive=8,
        pad_duration=7,
        pad_dynamics=6,
    )
    batch = [
        _sample([1, 2, 3, 4], [0, 1, 2, 3], [0, 1, 2, 3]),
        _sample([5, 6], [1, 2], [2, 3]),
    ]

    collated = planner_collate_fn(batch, metadata)

    assert collated["primitive_history"][1].tolist() == [5, 6, 8, 8]
    assert collated["attention_mask"][1].tolist() == [1.0, 1.0, 0.0, 0.0]

    model = PrimitivePlannerTransformer(
        num_primitives=metadata.num_primitives,
        num_duration_buckets=metadata.num_duration_buckets,
        num_dynamics_buckets=metadata.num_dynamics_buckets,
        score_dim=metadata.score_dim,
        d_model=16,
        nhead=4,
        num_layers=2,
        dim_feedforward=32,
        dropout=0.0,
        max_length=4,
    )

    with torch.no_grad():
        outputs = model(collated)

    assert torch.isfinite(outputs["plan_embedding"]).all()
