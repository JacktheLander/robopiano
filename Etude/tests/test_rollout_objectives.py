from __future__ import annotations

import numpy as np

from etude.training.dagger_hooks import DAggerHook, FailureWindowCollector
from etude.training.rollout_objectives import RolloutObjectiveComposer


def test_rollout_objective_composer_returns_weighted_sum_and_breakdown() -> None:
    composer = RolloutObjectiveComposer.from_config(
        {
            "reward": {
                "target_key_activation": 2.0,
                "missed_key": -3.0,
                "action_l2": -0.5,
            }
        }
    )
    state = {
        "predicted_keys": np.array([1.0, 0.2, 0.8], dtype=np.float32),
        "target_keys": np.array([1.0, 1.0, 0.0], dtype=np.float32),
        "action": np.array([1.0, -1.0], dtype=np.float32),
    }

    total, breakdown = composer.compute(state, return_breakdown=True)

    assert np.isclose(breakdown["target_key_activation"], 2.4)
    assert np.isclose(breakdown["missed_key"], -2.4)
    assert np.isclose(breakdown["action_l2"], -0.5)
    assert np.isclose(total, -0.5)


def test_failure_window_collector_and_hook_export_samples() -> None:
    collector = FailureWindowCollector(window_size=3)
    hook = DAggerHook(collector=collector)
    frames = [
        {"obs": {"x": np.array([0.0], dtype=np.float32)}, "is_failure": False},
        {"obs": {"x": np.array([1.0], dtype=np.float32)}, "is_failure": False},
        {"obs": {"x": np.array([2.0], dtype=np.float32)}, "is_failure": True},
    ]

    for frame in frames:
        hook.observe_frame(frame)

    exported = hook.export_in_memory_samples()

    assert len(exported) == 1
    assert exported[0]["start_index"] == 0
    assert exported[0]["end_index"] == 2
    assert len(exported[0]["frames"]) == 3
