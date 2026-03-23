from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sonata.transformer.dataset import PlannerMetadata, planner_collate_fn
from sonata.transformer.families import derive_primitive_family_mapping
from sonata.transformer.model import PrimitivePlannerTransformer


def _metadata() -> PlannerMetadata:
    return PlannerMetadata(
        num_primitives=8,
        num_duration_buckets=7,
        num_dynamics_buckets=6,
        num_families=3,
        score_dim=17,
        history_dim=20,
        pad_primitive=8,
        pad_duration=7,
        pad_dynamics=6,
        pad_family=3,
        primitive_ids=[f"primitive_{index:03d}" for index in range(8)],
        primitive_family_names=["single_note", "stacked_onset", "chordal"],
        primitive_to_family=[0, 0, 1, 2, 0, 1, 0, 2],
        family_mapping_mode="heuristic_stats",
        continuous_param_names=["motion_energy", "chord_size"],
        continuous_param_mean=[0.0, 0.0],
        continuous_param_std=[1.0, 1.0],
        goal_context_features=[f"goal_{index}" for index in range(17)],
        history_context_features=[f"hist_{index}" for index in range(20)],
    )


def _sample(history: list[int], families: list[int], duration: list[int], dynamics: list[int]) -> dict[str, object]:
    return {
        "primitive_history": np.asarray(history, dtype=np.int64),
        "family_history": np.asarray(families, dtype=np.int64),
        "duration_history": np.asarray(duration, dtype=np.int64),
        "dynamics_history": np.asarray(dynamics, dtype=np.int64),
        "history_context": np.ones((len(history), 20), dtype=np.float32),
        "planner_context": np.ones((17,), dtype=np.float32),
        "target_primitive": history[-1],
        "target_family": families[-1],
        "target_duration": duration[-1],
        "target_dynamics": dynamics[-1],
        "target_params": np.asarray([0.0, 0.0], dtype=np.float32),
    }


def test_planner_collate_right_padding_keeps_finite_plan_embeddings() -> None:
    metadata = _metadata()
    batch = [
        _sample([1, 2, 3, 4], [0, 0, 1, 0], [0, 1, 2, 3], [0, 1, 2, 3]),
        _sample([5, 6], [1, 0], [1, 2], [2, 3]),
    ]

    collated = planner_collate_fn(batch, metadata)

    assert collated["primitive_history"][1].tolist() == [5, 6, 8, 8]
    assert collated["family_history"][1].tolist() == [1, 0, 3, 3]
    assert collated["attention_mask"][1].tolist() == [1.0, 1.0, 0.0, 0.0]

    model = PrimitivePlannerTransformer(
        num_primitives=metadata.num_primitives,
        num_duration_buckets=metadata.num_duration_buckets,
        num_dynamics_buckets=metadata.num_dynamics_buckets,
        num_families=metadata.num_families,
        primitive_to_family=metadata.primitive_to_family,
        history_context_dim=metadata.history_dim,
        goal_context_dim=metadata.score_dim,
        continuous_param_dim=metadata.continuous_param_dim,
        d_model=16,
        nhead=4,
        num_layers=2,
        dim_feedforward=32,
        dropout=0.0,
        max_length=4,
        plan_embedding_dim=12,
    )

    with torch.no_grad():
        outputs = model(collated)

    assert outputs["family_logits"].shape == (2, metadata.num_families)
    assert outputs["primitive_logits"].shape == (2, metadata.num_primitives)
    assert outputs["continuous_param_pred"].shape == (2, metadata.continuous_param_dim)
    assert outputs["plan_embedding"].shape == (2, 12)
    assert torch.isfinite(outputs["plan_embedding"]).all()


def test_family_mapping_is_deterministic_under_row_reordering() -> None:
    frame = pd.DataFrame(
        [
            {"primitive_id": "primitive_000", "split": "train", "heuristic_family": "single", "duration_steps": 20.0, "motion_energy": 5.0, "chord_size": 1.0},
            {"primitive_id": "primitive_000", "split": "train", "heuristic_family": "single", "duration_steps": 22.0, "motion_energy": 5.5, "chord_size": 1.0},
            {"primitive_id": "primitive_001", "split": "train", "heuristic_family": "stacked", "duration_steps": 18.0, "motion_energy": 4.0, "chord_size": 2.0},
            {"primitive_id": "primitive_002", "split": "train", "heuristic_family": "single", "duration_steps": 42.0, "motion_energy": 2.0, "chord_size": 1.0},
            {"primitive_id": "primitive_003", "split": "train", "heuristic_family": "chord", "duration_steps": 30.0, "motion_energy": 6.0, "chord_size": 4.0},
        ]
    )
    primitive_ids = sorted(frame["primitive_id"].unique().tolist())
    shuffled = frame.sample(frac=1.0, random_state=13).reset_index(drop=True)

    first = derive_primitive_family_mapping(frame, primitive_ids, mode="heuristic_stats")
    second = derive_primitive_family_mapping(shuffled, primitive_ids, mode="heuristic_stats")

    assert first.family_names == second.family_names
    assert first.primitive_to_family_name == second.primitive_to_family_name
    assert first.primitive_to_family_index == second.primitive_to_family_index
