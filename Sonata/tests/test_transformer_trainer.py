from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sonata.transformer.dataset import PlannerMetadata, family_mask_tensor
from sonata.transformer.trainer import compute_loss, normalize_transformer_config, validate_transformer_config


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


def test_compute_loss_zeros_dynamics_when_use_dynamics_loss_false() -> None:
    metadata = _metadata()
    outputs = {
        "family_logits": torch.tensor([[8.0, -4.0]], dtype=torch.float32),
        "primitive_logits": torch.tensor([[0.2, 0.1, 9.0, 8.5]], dtype=torch.float32),
        "duration_logits": torch.tensor([[3.0, 1.0, -2.0]], dtype=torch.float32),
        "dynamics_logits": torch.tensor([[99.0, -99.0]], dtype=torch.float32),
        "continuous_param_pred": torch.tensor([[0.1]], dtype=torch.float32),
    }
    batch = {
        "target_family": torch.tensor([0], dtype=torch.long),
        "target_primitive": torch.tensor([0], dtype=torch.long),
        "target_duration": torch.tensor([0], dtype=torch.long),
        "target_dynamics": torch.tensor([1], dtype=torch.long),
        "target_params": torch.tensor([[0.0]], dtype=torch.float32),
    }
    on = {
        "label_smoothing": 0.0,
        "focal_heads": set(),
        "focal_gamma": 0.0,
        "family_class_weights": None,
        "primitive_class_weights": None,
        "family_primitive_mask": family_mask_tensor(metadata),
        "use_dynamics_loss": True,
        "log_dynamics_metrics": True,
        "loss_weights": {"family": 1.0, "primitive": 1.0, "duration": 1.0, "dynamics": 1.0, "params": 1.0},
        "normalize_loss_by_active_weights": False,
    }
    off = dict(on)
    off["use_dynamics_loss"] = False
    off["log_dynamics_metrics"] = False
    off["loss_weights"] = {**on["loss_weights"], "dynamics": 0.0}

    loss_on, metrics_on = compute_loss(outputs, batch, "factored_goal_conditioned", loss_config=on)
    loss_off, metrics_off = compute_loss(outputs, batch, "factored_goal_conditioned", loss_config=off)

    assert torch.isfinite(loss_on)
    assert torch.isfinite(loss_off)
    assert metrics_off["dynamics_loss"] == 0.0
    assert "dynamics_accuracy" not in metrics_off
    assert "dynamics_accuracy" in metrics_on
    assert loss_on.item() > loss_off.item()


def test_normalize_predict_dynamics_false_sets_dependent_defaults() -> None:
    cfg = normalize_transformer_config(
        {
            "model_variant": "factored_goal_conditioned",
            "d_model": 16,
            "context_length": 4,
            "plan_embedding_dim": 12,
            "eval_temperature": 1.0,
            "predict_dynamics": False,
        }
    )
    assert cfg["use_dynamics_loss"] is False
    assert cfg["use_dynamics_intent_in_plan"] is False
    assert cfg["log_dynamics_metrics"] is False
    validate_transformer_config(cfg)


def test_validate_rejects_predict_dynamics_false_with_loss_enabled() -> None:
    cfg = normalize_transformer_config(
        {
            "model_variant": "factored_goal_conditioned",
            "d_model": 16,
            "context_length": 4,
            "plan_embedding_dim": 12,
            "eval_temperature": 1.0,
            "predict_dynamics": False,
            "use_dynamics_loss": True,
        }
    )
    try:
        validate_transformer_config(cfg)
    except ValueError as exc:
        assert "predict_dynamics=False" in str(exc)
    else:
        raise AssertionError("Expected ValueError for incompatible dynamics flags.")


def test_validate_rejects_predict_dynamics_false_with_intent_enabled() -> None:
    cfg = normalize_transformer_config(
        {
            "model_variant": "factored_goal_conditioned",
            "d_model": 16,
            "context_length": 4,
            "plan_embedding_dim": 12,
            "eval_temperature": 1.0,
            "predict_dynamics": False,
        }
    )
    cfg["use_dynamics_intent_in_plan"] = True
    try:
        validate_transformer_config(cfg)
    except ValueError as exc:
        assert "use_dynamics_intent_in_plan" in str(exc)
    else:
        raise AssertionError("Expected ValueError for incompatible dynamics intent flag.")


def test_validate_allows_predict_dynamics_false_with_log_metrics_true() -> None:
    cfg = normalize_transformer_config(
        {
            "model_variant": "factored_goal_conditioned",
            "d_model": 16,
            "context_length": 4,
            "plan_embedding_dim": 12,
            "eval_temperature": 1.0,
            "predict_dynamics": False,
        }
    )
    cfg["log_dynamics_metrics"] = True
    validate_transformer_config(cfg)


def test_transformer_config_defaults_keep_linear_selector_and_remap_disabled() -> None:
    config = normalize_transformer_config(
        {
            "model_variant": "factored_goal_conditioned",
            "d_model": 16,
            "context_length": 4,
            "plan_embedding_dim": 12,
            "eval_temperature": 1.0,
        }
    )

    validate_transformer_config(config)

    assert config["primitive_selector_type"] == "linear"
    assert config["primitive_selector_hidden_dim"] == 32
    assert config["primitive_remap"] == {"enabled": False}
    assert config["predict_dynamics"] is True
    assert config["use_dynamics_loss"] is True
    assert config["use_dynamics_intent_in_plan"] is True
    assert config["log_dynamics_metrics"] is True
    assert config["use_dynamics_in_history"] is True


def test_normalize_use_dynamics_in_plan_embedding_alias() -> None:
    cfg = normalize_transformer_config(
        {
            "model_variant": "factored_goal_conditioned",
            "d_model": 16,
            "context_length": 4,
            "plan_embedding_dim": 12,
            "eval_temperature": 1.0,
            "use_dynamics_in_plan_embedding": False,
        }
    )
    assert cfg["use_dynamics_intent_in_plan"] is False
    validate_transformer_config(cfg)


def test_normalize_rejects_conflicting_plan_embedding_keys() -> None:
    try:
        normalize_transformer_config(
            {
                "model_variant": "factored_goal_conditioned",
                "d_model": 16,
                "context_length": 4,
                "plan_embedding_dim": 12,
                "eval_temperature": 1.0,
                "use_dynamics_in_plan_embedding": False,
                "use_dynamics_intent_in_plan": True,
            }
        )
    except ValueError as exc:
        assert "disagree" in str(exc)
    else:
        raise AssertionError("Expected ValueError for conflicting plan embedding keys.")


def test_transformer_config_rejects_invalid_primitive_selector() -> None:
    config = normalize_transformer_config(
        {
            "model_variant": "factored_goal_conditioned",
            "d_model": 16,
            "context_length": 4,
            "plan_embedding_dim": 12,
            "primitive_selector_type": "bad",
        }
    )

    try:
        validate_transformer_config(config)
    except ValueError as exc:
        assert "primitive_selector_type" in str(exc)
    else:
        raise AssertionError("Expected invalid primitive selector config to raise ValueError.")
