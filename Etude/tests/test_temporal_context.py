from __future__ import annotations

import numpy as np
import torch

from etude.controllers.pd import PDController
from etude.controllers.temporal_context import TemporalContextBuffer
from etude.controllers.temporal_residual import TemporalResidualController
from etude.robopianist.state_mapping import StateMapping


class VectorResidualModel(torch.nn.Module):
    def __init__(self, action_dim: int, value: float) -> None:
        super().__init__()
        self.register_buffer("residual", torch.full((action_dim,), value, dtype=torch.float32))

    def forward(self, sequence: torch.Tensor, hidden: torch.Tensor | None = None) -> torch.Tensor:
        return self.residual


class BatchedTemporalResidualModel(torch.nn.Module):
    def __init__(self, action_dim: int, value: float) -> None:
        super().__init__()
        self.register_buffer("residual", torch.full((action_dim,), value, dtype=torch.float32))

    def forward(self, sequence: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, steps = sequence.shape[:2]
        output = self.residual.view(1, 1, -1).expand(batch, steps, -1)
        return {"residual": output}


def _mapping(action_dim: int = 4) -> StateMapping:
    return StateMapping(
        qpos_indices_46=list(range(46)),
        qvel_indices_46=list(range(46)),
        action_indices=list(range(action_dim)),
        fingertip_indices=None,
        key_state_indices=None,
        action_low=-np.ones(action_dim, dtype=np.float32),
        action_high=np.ones(action_dim, dtype=np.float32),
    )


def _obs() -> dict[str, np.ndarray]:
    return {
        "q": np.zeros(46, dtype=np.float32),
        "qdot": np.zeros(46, dtype=np.float32),
    }


def test_temporal_context_buffer_tracks_length_and_shapes() -> None:
    buffer = TemporalContextBuffer(history_steps=3, store_actions=True, store_residuals=True)

    for index in range(4):
        buffer.append(
            np.full(5, index, dtype=np.float32),
            action=np.full(2, index, dtype=np.float32),
            residual=np.full(2, -index, dtype=np.float32),
        )

    assert len(buffer) == 3
    assert buffer.feature_history().shape == (3, 5)
    assert buffer.action_history() is not None
    assert buffer.action_history().shape == (3, 2)
    assert buffer.residual_history() is not None
    assert np.allclose(buffer.feature_history()[-1], np.full(5, 3, dtype=np.float32))


def test_temporal_context_reset_clears_state() -> None:
    buffer = TemporalContextBuffer(history_steps=2)
    buffer.append(
        np.ones(4, dtype=np.float32),
        action=np.ones(2, dtype=np.float32),
        residual=np.ones(2, dtype=np.float32),
    )
    buffer.reset()

    assert len(buffer) == 0
    assert buffer.feature_history().shape == (0, 4)
    assert buffer.action_history() is not None
    assert buffer.action_history().shape == (0, 2)
    assert buffer.residual_history() is not None
    assert buffer.residual_history().shape == (0, 2)


def test_temporal_residual_controller_accepts_vector_output() -> None:
    mapping = _mapping()
    controller = TemporalResidualController(
        mapping=mapping,
        temporal_model=VectorResidualModel(action_dim=4, value=0.25),
        pd=PDController(mapping, kp=0.0, kd=0.0, lookahead=0),
        history_steps=4,
        residual_clip=1.0,
        device="cpu",
    )
    controller.reset(np.zeros((3, 46), dtype=np.float32), np.zeros((3, 46), dtype=np.float32))

    action = controller.act(_obs(), 0)

    assert action.shape == (4,)
    assert np.allclose(action, 0.25)
    assert len(controller.context) == 1


def test_temporal_residual_controller_accepts_dict_batched_output() -> None:
    mapping = _mapping()
    controller = TemporalResidualController(
        mapping=mapping,
        temporal_model=BatchedTemporalResidualModel(action_dim=4, value=0.1),
        pd=PDController(mapping, kp=0.0, kd=0.0, lookahead=0),
        history_steps=4,
        residual_clip=1.0,
        use_hidden_state=False,
        device="cpu",
    )
    controller.reset(np.zeros((3, 46), dtype=np.float32), np.zeros((3, 46), dtype=np.float32))

    action = controller.act(_obs(), 0)

    assert action.shape == (4,)
    assert np.allclose(action, 0.1)
