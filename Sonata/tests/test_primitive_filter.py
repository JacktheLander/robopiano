from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

pytest.importorskip("torch")

from sonata.transformer.dataset import PlannerMetadata, PrimitiveSequenceDataset
from sonata.transformer.primitive_filter import build_primitive_filter


def _meta_5() -> PlannerMetadata:
    return PlannerMetadata(
        num_primitives=5,
        num_duration_buckets=3,
        num_dynamics_buckets=2,
        num_families=2,
        score_dim=17,
        history_dim=20,
        pad_primitive=5,
        pad_duration=3,
        pad_dynamics=2,
        pad_family=2,
        primitive_ids=["0", "1", "2", "3", "4"],
        primitive_family_names=["a", "b"],
        primitive_to_family=[0, 0, 1, 1, 0],
        family_mapping_mode="heuristic_stats",
        continuous_param_names=[],
        continuous_param_mean=[],
        continuous_param_std=[],
        goal_context_features=[],
        history_context_features=[],
    )


def test_build_primitive_filter_sets_and_missing_quality() -> None:
    quality = pd.DataFrame(
        [
            {"primitive_id": 0, "onset_f1": 0.0, "key_press": False, "target_hit": False, "strong_f1": False},
            {"primitive_id": 1, "onset_f1": 0.35, "key_press": True, "target_hit": True, "strong_f1": False},
            {"primitive_id": 2, "onset_f1": 0.45, "key_press": True, "target_hit": True, "strong_f1": False},
            {"primitive_id": 3, "onset_f1": 0.75, "key_press": True, "target_hit": True, "strong_f1": True},
        ]
    )
    config = {
        "primitive_filter": {
            "enabled": True,
            "mode": "drop",
            "min_onset_f1": 0.40,
            "strong_onset_f1": 0.60,
            "require_key_press": True,
            "require_target_hit": True,
            "require_strong_f1": False,
            "bad_primitive_weight": 0.0,
            "borderline_weight": 0.5,
        }
    }
    bundle = build_primitive_filter(quality, config, _meta_5())
    assert bundle["bad_primitive_ids"] == {0, 1}
    assert bundle["borderline_primitive_ids"] == {2}
    assert bundle["strong_primitive_ids"] == {3}
    assert 4 in bundle["valid_primitive_ids"]
    assert bundle["canon_flags"][4]["quality_status"] == "missing_quality"
    assert bundle["vocab_bad_indices"] == frozenset({0, 1})


def test_drop_mode_skips_bad_target_rows() -> None:
    meta = _meta_5()
    score = '{"goal_histogram": [0.0]*12, "active_ratio": 0.0, "future_density": 0.0}'
    prim_seq = [2, 0, 2]
    rows = []
    for step, prim in enumerate(prim_seq):
        rows.append(
            {
                "split": "train",
                "episode_id": 1,
                "song_id": "s",
                "onset_step": step * 10,
                "end_step": 5,
                "primitive_id": str(prim),
                "primitive_index": prim,
                "primitive_family_index": meta.primitive_to_family[prim],
                "duration_bucket": 0,
                "dynamics_bucket": 0,
                "score_context_json": score,
            }
        )
    token_df = pd.DataFrame(rows)
    quality = pd.DataFrame(
        [
            {"primitive_id": 0, "onset_f1": 0.0, "key_press": False, "target_hit": False},
            {"primitive_id": 1, "onset_f1": 0.35, "key_press": True, "target_hit": True},
            {"primitive_id": 2, "onset_f1": 0.45, "key_press": True, "target_hit": True},
            {"primitive_id": 3, "onset_f1": 0.75, "key_press": True, "target_hit": True},
        ]
    )
    config = {
        "primitive_filter": {
            "enabled": True,
            "mode": "drop",
            "min_onset_f1": 0.40,
            "strong_onset_f1": 0.60,
            "require_key_press": True,
            "require_target_hit": True,
        }
    }
    bundle = build_primitive_filter(quality, config, meta)
    full = PrimitiveSequenceDataset(token_df, meta, context_length=4, split="train", primitive_filter_mode="none")
    filt = PrimitiveSequenceDataset(
        token_df,
        meta,
        context_length=4,
        split="train",
        vocab_bad_indices=bundle["vocab_bad_indices"],
        primitive_filter_mode="drop",
    )
    assert len(full.target_primitive) == 2
    assert len(filt.target_primitive) == 1
    assert int(filt.target_primitive[0]) == 2
