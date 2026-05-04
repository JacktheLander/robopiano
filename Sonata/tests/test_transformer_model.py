from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sonata.transformer.dataset import PlannerMetadata
from sonata.transformer.model import PrimitivePlannerTransformer, PrimitiveSelectionMLP, build_planner_from_config


def _metadata() -> PlannerMetadata:
    return PlannerMetadata(
        num_primitives=4,
        num_duration_buckets=3,
        num_dynamics_buckets=2,
        num_families=2,
        score_dim=17,
        history_dim=20,
        pad_primitive=4,
        pad_duration=3,
        pad_dynamics=2,
        pad_family=2,
        primitive_ids=["primitive_000", "primitive_001", "primitive_002", "primitive_003"],
        primitive_family_names=["single_note", "stacked_onset"],
        primitive_to_family=[0, 0, 1, 1],
        family_mapping_mode="heuristic_stats",
        continuous_param_names=["motion_energy"],
        continuous_param_mean=[0.0],
        continuous_param_std=[1.0],
        goal_context_features=[f"goal_{index}" for index in range(17)],
        history_context_features=[f"hist_{index}" for index in range(20)],
    )


def _batch() -> dict[str, torch.Tensor]:
    return {
        "primitive_history": torch.tensor([[0, 1], [2, 3]], dtype=torch.long),
        "family_history": torch.tensor([[0, 0], [1, 1]], dtype=torch.long),
        "duration_history": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        "dynamics_history": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        "history_context": torch.randn(2, 2, 20),
        "planner_context": torch.randn(2, 17),
        "attention_mask": torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float32),
    }


def test_oracle_plan_embedding_uses_forced_tokens() -> None:
    model = build_planner_from_config(
        _metadata(),
        {
            "d_model": 16,
            "nhead": 4,
            "num_layers": 1,
            "dim_feedforward": 32,
            "dropout": 0.0,
            "context_length": 4,
            "plan_embedding_dim": 12,
        },
    )
    batch = _batch()

    default_outputs = model(batch)
    oracle_outputs = model.oracle_plan_embedding(
        batch,
        family_index=torch.tensor([0, 1], dtype=torch.long),
        primitive_index=torch.tensor([1, 2], dtype=torch.long),
        duration_bucket=torch.tensor([2, 0], dtype=torch.long),
        dynamics_bucket=torch.tensor([1, 1], dtype=torch.long),
    )

    assert oracle_outputs["plan_embedding"].shape == (2, 12)
    assert oracle_outputs["family_intent"].shape == (2, 16)
    assert not torch.allclose(default_outputs["plan_embedding"], oracle_outputs["plan_embedding"])


def test_build_planner_uses_mlp_primitive_selector() -> None:
    model = build_planner_from_config(
        _metadata(),
        {
            "d_model": 16,
            "nhead": 4,
            "num_layers": 1,
            "dim_feedforward": 32,
            "dropout": 0.0,
            "context_length": 4,
            "plan_embedding_dim": 12,
            "primitive_selector_type": "mlp",
            "primitive_selector_hidden_dim": 24,
            "primitive_selector_layers": 2,
            "primitive_selector_dropout": 0.1,
        },
    )

    outputs = model(_batch())

    assert isinstance(model.primitive_head, PrimitiveSelectionMLP)
    assert outputs["primitive_logits"].shape == (2, 4)


def test_build_planner_without_dynamics_prediction() -> None:
    model = PrimitivePlannerTransformer(
        num_primitives=4,
        num_duration_buckets=3,
        num_dynamics_buckets=2,
        num_families=2,
        primitive_to_family=[0, 0, 1, 1],
        history_context_dim=20,
        goal_context_dim=17,
        continuous_param_dim=1,
        d_model=16,
        nhead=4,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        max_length=4,
        plan_embedding_dim=12,
        primitive_selector_type="linear",
        predict_dynamics=False,
        use_dynamics_intent_in_plan=False,
    )
    assert model.dynamics_head is None
    assert model.dynamics_intent_embed is None
    batch = _batch()

    outputs = model(batch)
    assert outputs["dynamics_logits"].shape == (2, 2)
    assert torch.allclose(outputs["dynamics_logits"], torch.zeros_like(outputs["dynamics_logits"]))
    assert torch.allclose(outputs["dynamics_intent"], torch.zeros_like(outputs["dynamics_intent"]))
    assert outputs["plan_embedding"].shape == (2, 12)

    oracle = model.oracle_plan_embedding(
        batch,
        family_index=torch.tensor([0, 1], dtype=torch.long),
        primitive_index=torch.tensor([1, 2], dtype=torch.long),
        duration_bucket=torch.tensor([2, 0], dtype=torch.long),
        dynamics_bucket=torch.tensor([1, 0], dtype=torch.long),
    )
    assert oracle["dynamics_logits"].shape == (2, 2)
    assert torch.allclose(oracle["dynamics_logits"], torch.zeros_like(oracle["dynamics_logits"]))
    assert torch.allclose(oracle["dynamics_intent"], torch.zeros_like(oracle["dynamics_intent"]))
    assert oracle["plan_embedding"].shape == (2, 12)
