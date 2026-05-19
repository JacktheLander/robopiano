from __future__ import annotations

from typing import Any

import numpy as np

from etude.controllers.base import TrajectoryFollower
from etude.robopianist.observation import extract_tracking_observation
from etude.robopianist.state_mapping import StateMapping


def rollout_controller(
    env: Any,
    controller: TrajectoryFollower,
    mapping: StateMapping,
    q_ref: np.ndarray,
    qdot_ref: np.ndarray | None = None,
    max_steps: int | None = None,
) -> dict[str, np.ndarray]:
    """Run a controller in a dm_control-style environment."""
    time_step = env.reset()
    controller.reset(q_ref, qdot_ref)
    steps = min(q_ref.shape[0], max_steps or q_ref.shape[0])
    actions = []
    q = []
    qdot = []
    for t in range(steps):
        obs = extract_tracking_observation(time_step.observation, mapping)
        action = controller.act(obs, t)
        actions.append(action)
        q.append(obs["q"])
        qdot.append(obs["qdot"])
        time_step = env.step(action)
        if getattr(time_step, "last", lambda: False)():
            break
    return {
        "actions": np.asarray(actions, dtype=np.float32),
        "q": np.asarray(q, dtype=np.float32),
        "qdot": np.asarray(qdot, dtype=np.float32),
    }
