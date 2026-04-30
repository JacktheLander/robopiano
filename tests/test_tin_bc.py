from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tin.bc import BCPolicy, build_bc_observation, infer_obs_dim
from tin.online_rl import TrainArgs, initialize_agent_and_replay


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


def test_build_bc_observation_fills_lookahead_window() -> None:
    goals = np.zeros((3, 89), dtype=np.float32)
    goals[0, 0] = 1.0
    goals[1, 1] = 1.0
    goals[2, 2] = 1.0
    piano = np.zeros(89, dtype=np.float32)
    joints = np.arange(46, dtype=np.float32)

    observation = build_bc_observation(
        goals_trajectory=goals,
        piano_state_t=piano,
        hand_joints_t=joints,
        timestep=0,
        n_steps_lookahead=2,
    )

    assert observation.shape == (infer_obs_dim(2),)
    assert observation[0] == 1.0
    assert observation[89 + 1] == 1.0
    assert observation[(2 * 89) + 2] == 1.0


def test_initialize_agent_and_replay_loads_bc_checkpoint(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "bc.pt"
    model = BCPolicy(obs_dim=4, act_dim=3, hidden_dim=256)
    state_dict = model.state_dict()
    for tensor in state_dict.values():
        tensor.copy_(torch.full_like(tensor, 0.25))
    torch.save(
        {
            "actor": state_dict,
            "metadata": {"obs_dim": 4, "act_dim": 3, "n_steps_lookahead": 0, "hidden_dim": 256},
        },
        checkpoint_path,
    )

    args = TrainArgs(
        device="cpu",
        agent_backend="droq",
        batch_size=8,
        replay_capacity=32,
        compile_models=False,
        bc_checkpoint=checkpoint_path,
        droq_hidden_dim=256,
    )

    _, agent, _, device_info = initialize_agent_and_replay(args, _DummyEnv())
    loaded_state = agent.actor.state_dict()

    for name, tensor in state_dict.items():
        assert torch.allclose(loaded_state[name], tensor)
    assert device_info["bc_checkpoint"] == str(checkpoint_path)
    assert device_info["bc_metadata"]["obs_dim"] == 4
