from __future__ import annotations

import numpy as np

from etude.controllers.pd import PDController
from etude.robopianist.state_mapping import StateMapping


def test_pd_controller_returns_action_spec_shape_and_clips() -> None:
    mapping = StateMapping(
        qpos_indices_46=list(range(46)),
        qvel_indices_46=list(range(46)),
        action_indices=list(range(4)),
        fingertip_indices=None,
        key_state_indices=None,
        action_low=-np.ones(4, dtype=np.float32),
        action_high=np.ones(4, dtype=np.float32),
    )
    controller = PDController(mapping, kp=100.0, kd=0.0, lookahead=0)
    controller.reset(np.ones((2, 46), dtype=np.float32), np.zeros((2, 46), dtype=np.float32))
    action = controller.act({"q": np.zeros(46, dtype=np.float32), "qdot": np.zeros(46, dtype=np.float32)}, 0)
    assert action.shape == (4,)
    assert np.allclose(action, 1.0)
    assert controller.diagnostics()["control/action_clip_rate"] == 1.0
