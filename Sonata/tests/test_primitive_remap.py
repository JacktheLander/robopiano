from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sonata.transformer.decode import collapse_weak_primitive_logits, decode_factored_outputs
from sonata.transformer.primitive_remap import apply_primitive_remap_to_token_df, build_remap_tensor


def test_apply_primitive_remap_preserves_vocabulary_and_original_columns() -> None:
    token_df = pd.DataFrame(
        {
            "primitive_id": ["primitive_000", "primitive_001", "primitive_002", "primitive_001"],
            "primitive_index": [0, 1, 2, 1],
            "split": ["train", "train", "val", "val"],
        }
    )
    vocabulary = {
        "num_primitives": 3,
        "primitive_ids": ["primitive_000", "primitive_001", "primitive_002"],
    }
    payload = {"enabled": True, "mode": "weak_to_strong", "remap": {"primitive_001": "primitive_002"}}

    remapped, summary = apply_primitive_remap_to_token_df(token_df, vocabulary, payload, apply_to_history=True)

    assert vocabulary["num_primitives"] == 3
    assert remapped["primitive_index"].tolist() == [0, 2, 2, 2]
    assert remapped["primitive_id"].tolist() == ["primitive_000", "primitive_002", "primitive_002", "primitive_002"]
    assert remapped["original_primitive_index"].tolist() == [0, 1, 2, 1]
    assert remapped["original_primitive_id"].tolist() == ["primitive_000", "primitive_001", "primitive_002", "primitive_001"]
    assert summary["enabled"] is True
    assert summary["num_remapped_primitives"] == 1
    assert summary["remap"] == {"primitive_001": "primitive_002"}
    assert summary["apply_to_history"] is True


def test_build_remap_tensor_accepts_stringified_indices() -> None:
    tensor = build_remap_tensor(
        4,
        {"enabled": True, "remap": {"1": "3"}},
        ["primitive_000", "primitive_001", "primitive_002", "primitive_003"],
    )

    assert tensor is not None
    assert tensor.tolist() == [0, 3, 2, 3]


def test_decode_without_remap_keeps_existing_api() -> None:
    outputs = {
        "family_logits": torch.tensor([[1.0, 0.0]]),
        "primitive_logits": torch.tensor([[0.0, 2.0, 4.0]]),
        "duration_logits": torch.tensor([[0.0, 1.0]]),
        "dynamics_logits": torch.tensor([[1.0, 0.0]]),
    }
    family_mask = torch.tensor([[True, True, False], [False, False, True]])

    decoded = decode_factored_outputs(outputs, family_mask=family_mask)

    assert decoded["predicted_family"].tolist() == [0]
    assert decoded["predicted_primitive"].tolist() == [1]


def test_collapse_weak_primitive_logits_moves_mass_to_canonical_target() -> None:
    logits = torch.tensor([[0.0, 4.0, 1.0]])
    collapsed = collapse_weak_primitive_logits(logits, torch.tensor([0, 2, 2]))

    assert collapsed[0, 1].item() == -1.0e4
    assert collapsed[0, 2].item() > 4.0
