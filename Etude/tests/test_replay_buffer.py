from __future__ import annotations

import numpy as np

from etude.training.replay_buffer import ReplayBuffer, Transition


def _transition(step: float, *, is_failure: bool = False) -> Transition:
    return Transition(
        obs={"state": np.array([step], dtype=np.float32)},
        action=np.array([step, step + 1.0], dtype=np.float32),
        reward=step,
        next_obs={"state": np.array([step + 1.0], dtype=np.float32)},
        done=False,
        is_failure=is_failure,
    )


def test_replay_buffer_caps_capacity_and_samples_failures() -> None:
    buffer = ReplayBuffer(capacity=3)
    buffer.append_episode(
        [
            _transition(0.0),
            _transition(1.0, is_failure=True),
            _transition(2.0),
            _transition(3.0, is_failure=True),
        ]
    )

    assert len(buffer) == 3
    failure_batch = buffer.sample_batch(2, rng=np.random.default_rng(0), failure_only=True)

    assert failure_batch["is_failure"].shape == (2,)
    assert failure_batch["obs.state"].shape == (2, 1)
    assert np.all(failure_batch["is_failure"])
    assert set(failure_batch["obs.state"].reshape(-1).tolist()) == {1.0, 3.0}


def test_replay_buffer_append_from_kwargs_and_export_dict() -> None:
    buffer = ReplayBuffer(capacity=2)
    buffer.append(
        obs={"state": np.array([1.0, 2.0], dtype=np.float32)},
        action=np.array([0.5], dtype=np.float32),
        reward=1.25,
        next_obs={"state": np.array([2.0, 3.0], dtype=np.float32)},
        done=True,
        is_failure=False,
        metadata={"episode": np.asarray(7, dtype=np.int32)},
    )

    exported = buffer.as_dict()

    assert exported["action"].shape == (1, 1)
    assert exported["done"].shape == (1,)
    assert np.isclose(exported["reward"][0], 1.25)
    assert exported["meta.episode"][0] == 7
