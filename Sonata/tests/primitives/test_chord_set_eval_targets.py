from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sonata.evaluation.primitive_online_eval import infer_instance_key_events  # noqa: E402


def test_chord_set_eval_prefers_target_key_ids_json_over_goals() -> None:
    goals = np.zeros((12, 89), dtype=np.float32)
    goals[6:, 7] = 1.0

    intended, _ = infer_instance_key_events(
        goals=goals,
        piano_states=None,
        events_config={"use_goals": True, "use_piano_states": False},
        row={"target_key_ids_json": "[2, 3]", "target_key_signature": "7"},
        duration_steps=12,
    )

    assert intended.source == "target_key_ids_json"
    assert intended.unique_keys == (2, 3)


def test_chord_set_eval_does_not_use_piano_states_as_intended_by_default() -> None:
    piano_states = np.zeros((12, 89), dtype=np.float32)
    piano_states[6:, 9] = 1.0

    intended, realized = infer_instance_key_events(
        goals=None,
        piano_states=piano_states,
        events_config={"use_goals": True, "use_piano_states": True, "use_piano_states_as_intended": False},
        row={},
        duration_steps=12,
    )

    assert intended.events == []
    assert intended.source == "goals"
    assert realized.unique_keys == (9,)
