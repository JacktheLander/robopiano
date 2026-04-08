from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sonata.transformer.dataset import PlannerMetadata, family_mask_tensor
from sonata.transformer.trainer import compute_loss


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


def test_compute_loss_masks_primitive_predictions_by_family() -> None:
    metadata = _metadata()
    outputs = {
        "family_logits": torch.tensor([[8.0, -4.0]], dtype=torch.float32),
        "primitive_logits": torch.tensor([[0.2, 0.1, 9.0, 8.5]], dtype=torch.float32),
        "duration_logits": torch.tensor([[3.0, 1.0, -2.0]], dtype=torch.float32),
        "dynamics_logits": torch.tensor([[2.0, -1.0]], dtype=torch.float32),
        "continuous_param_pred": torch.tensor([[0.1]], dtype=torch.float32),
    }
    batch = {
        "target_family": torch.tensor([0], dtype=torch.long),
        "target_primitive": torch.tensor([0], dtype=torch.long),
        "target_duration": torch.tensor([0], dtype=torch.long),
        "target_dynamics": torch.tensor([0], dtype=torch.long),
        "target_params": torch.tensor([[0.0]], dtype=torch.float32),
    }
    loss_config = {
        "label_smoothing": 0.0,
        "focal_heads": set(),
        "focal_gamma": 0.0,
        "family_class_weights": None,
        "primitive_class_weights": None,
        "family_primitive_mask": family_mask_tensor(metadata),
        "loss_weights": {"family": 1.0, "primitive": 1.0, "duration": 1.0, "dynamics": 1.0, "params": 1.0},
        "normalize_loss_by_active_weights": True,
    }

    loss, metrics = compute_loss(outputs, batch, "factored_goal_conditioned", loss_config=loss_config)

    assert torch.isfinite(loss)
    assert metrics["family_accuracy"] == 1.0
    assert metrics["primitive_accuracy"] in {0.0, 1.0}
    assert metrics["primitive_loss"] > 0.0
