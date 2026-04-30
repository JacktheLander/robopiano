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

from sonata.evaluation.attribution import (
    bootstrap_mean,
    build_controlled_batch,
    confidence_score,
    summarize_attribution,
)


class _Metadata:
    primitive_ids = ["primitive_000", "primitive_001", "primitive_002"]


def test_build_controlled_batch_uses_oracle_prior_and_can_zero_it() -> None:
    batch = {
        "primitive_index": torch.tensor([0, 1], dtype=torch.long),
        "duration_bucket": torch.tensor([0, 1], dtype=torch.long),
        "dynamics_bucket": torch.tensor([0, 1], dtype=torch.long),
        "gmr_prior": torch.ones((2, 4, 3), dtype=torch.float32),
    }
    oracle_tokens = {
        "family_index": torch.tensor([0, 1], dtype=torch.long),
        "primitive_index": torch.tensor([1, 2], dtype=torch.long),
        "duration_bucket": torch.tensor([1, 0], dtype=torch.long),
        "dynamics_bucket": torch.tensor([1, 1], dtype=torch.long),
    }
    prior_lookup = {
        "primitive_000": np.full((4, 3), 1.0, dtype=np.float32),
        "primitive_001": np.full((4, 3), 2.0, dtype=np.float32),
        "primitive_002": np.full((4, 3), 3.0, dtype=np.float32),
    }

    controlled, oracle = build_controlled_batch(
        batch=batch,
        metadata=_Metadata(),
        prior_lookup=prior_lookup,
        control_mode="oracle_full",
        oracle_tokens=oracle_tokens,
    )
    zeroed, _ = build_controlled_batch(
        batch=batch,
        metadata=_Metadata(),
        prior_lookup=prior_lookup,
        control_mode="oracle_no_prior",
        oracle_tokens=oracle_tokens,
    )

    assert oracle is not None
    assert controlled["primitive_index"].tolist() == [1, 2]
    assert torch.allclose(controlled["gmr_prior"][0], torch.full((4, 3), 2.0))
    assert torch.allclose(controlled["gmr_prior"][1], torch.full((4, 3), 3.0))
    assert torch.count_nonzero(zeroed["gmr_prior"]) == 0


def test_bootstrap_mean_is_deterministic() -> None:
    values = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    left = bootstrap_mean(values, samples=128, seed=7)
    right = bootstrap_mean(values, samples=128, seed=7)

    assert left == right
    assert left["support"] == 4


def test_summarize_attribution_prefers_planner_when_oracle_full_is_better() -> None:
    segment_df = pd.DataFrame(
        [
            {
                "predicted_full/action_l1": 2.0,
                "oracle_full/action_l1": 0.5,
                "oracle_no_prior/action_l1": 0.6,
                "oracle_gmr_only/action_l1": 1.3,
                "predicted_full/action_mse": 3.0,
                "oracle_full/action_mse": 0.5,
                "oracle_no_prior/action_mse": 0.7,
                "oracle_gmr_only/action_mse": 1.5,
            }
        ]
    )
    episode_df = pd.DataFrame(
        [
            {"episode_id": "a", "control_mode": "predicted_full", "reward": 0.1, "f1": 0.2, "sustain_f1": 0.2},
            {"episode_id": "a", "control_mode": "oracle_full", "reward": 0.8, "f1": 0.9, "sustain_f1": 0.8},
            {"episode_id": "a", "control_mode": "oracle_no_prior", "reward": 0.7, "f1": 0.8, "sustain_f1": 0.7},
            {"episode_id": "a", "control_mode": "oracle_gmr_only", "reward": 0.5, "f1": 0.5, "sustain_f1": 0.4},
            {"episode_id": "b", "control_mode": "predicted_full", "reward": 0.2, "f1": 0.1, "sustain_f1": 0.1},
            {"episode_id": "b", "control_mode": "oracle_full", "reward": 0.9, "f1": 0.8, "sustain_f1": 0.9},
            {"episode_id": "b", "control_mode": "oracle_no_prior", "reward": 0.85, "f1": 0.7, "sustain_f1": 0.7},
            {"episode_id": "b", "control_mode": "oracle_gmr_only", "reward": 0.4, "f1": 0.4, "sustain_f1": 0.3},
        ]
    )

    summary, bootstrap_df = summarize_attribution(segment_df=segment_df, episode_df=episode_df, bootstrap_samples=256, seed=3)

    assert not bootstrap_df.empty
    assert summary["planner_impact/f1_mean"] > summary["prior_benefit/f1_mean"]
    assert summary["planner_impact/reward_mean"] > summary["diffusion_benefit/reward_mean"]
    assert summary["dominant_bottleneck"] == "planner_impact"
    assert confidence_score(summary, "planner_impact") > 0.0
