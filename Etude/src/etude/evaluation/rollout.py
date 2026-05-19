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
    metadata: dict[str, Any] | None = None,
    max_steps: int | None = None,
) -> dict[str, np.ndarray]:
    """Run a controller in a dm_control-style environment."""
    time_step = env.reset()
    rollout_metadata = dict(metadata or {})
    controller.reset(q_ref, qdot_ref, metadata=rollout_metadata)
    steps = min(q_ref.shape[0], max_steps or q_ref.shape[0])
    actions = []
    q = []
    qdot = []
    key_state = []
    for t in range(steps):
        obs = extract_tracking_observation(time_step.observation, mapping)
        for key in ("target_keys", "desired_fingertips", "fingertip_ref", "time_to_next_active_key"):
            if key in rollout_metadata:
                obs[key] = rollout_metadata[key]
        action = controller.act(obs, t)
        actions.append(action)
        q.append(obs["q"])
        qdot.append(obs["qdot"])
        if "key_state" in obs:
            key_state.append(np.asarray(obs["key_state"], dtype=np.float32))
        time_step = env.step(action)
        if getattr(time_step, "last", lambda: False)():
            break
    rollout = {
        "actions": np.asarray(actions, dtype=np.float32),
        "q": np.asarray(q, dtype=np.float32),
        "qdot": np.asarray(qdot, dtype=np.float32),
    }
    if key_state:
        rollout["key_state"] = np.asarray(key_state, dtype=np.float32)
    return rollout
