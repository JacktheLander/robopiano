from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tin.online_rl import TrainArgs, initialize_agent_and_replay, resolve_backend


class _DummyTimestep:
    def __init__(self, observation: np.ndarray) -> None:
        self.observation = observation


class _DummyActionSpec:
    def __init__(self) -> None:
        self.shape = (3,)
        self.minimum = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
        self.maximum = np.array([1.0, 1.0, 1.0], dtype=np.float32)


class _DummyEnv:
    def __init__(self) -> None:
        self._action_spec = _DummyActionSpec()

    def reset(self) -> _DummyTimestep:
        return _DummyTimestep(np.array([0.1, -0.2, 0.3, 0.4], dtype=np.float32))

    def action_spec(self) -> _DummyActionSpec:
        return self._action_spec


def test_resolve_backend_prefers_droq_on_cuda() -> None:
    assert resolve_backend("auto", "cuda") == "droq"
    assert resolve_backend("auto", "cpu") == "sac"
    assert resolve_backend("sac", "cuda") == "sac"


def test_initialize_agent_and_replay_supports_droq_backend_without_sac_runtime() -> None:
    args = TrainArgs(
        device="cpu",
        agent_backend="droq",
        batch_size=8,
        replay_capacity=32,
        compile_models=False,
    )

    spec, agent, replay_buffer, device_info = initialize_agent_and_replay(args, _DummyEnv())

    assert spec is None
    assert agent.backend == "droq"
    assert replay_buffer.batch_size == 8
    assert replay_buffer.capacity == 32
    assert device_info["resolved_backend"] == "droq"
    assert device_info["resolved_device"] == "cpu"
    assert device_info["observation_dim"] == 4
    assert device_info["action_dim"] == 3
