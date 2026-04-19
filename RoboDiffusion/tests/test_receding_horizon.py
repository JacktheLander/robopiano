from __future__ import annotations

import numpy as np

from robodiffusion.evaluation.rollout import RecedingHorizonController


class DummyPolicy:
    def __init__(self) -> None:
        self.metadata = type(
            "Metadata",
            (),
            {
                "obs_horizon": 3,
                "pred_horizon": 4,
                "action_execute_horizon": 2,
                "score_dim": 14,
                "state_dim": 184,
                "action_dim": 3,
            },
        )()
        self.calls = 0

    def sample_action_chunk(self, *, score_window, state_window, warm_start=None):
        del score_window, state_window, warm_start
        self.calls += 1
        chunk = np.full((1, 4, 3), fill_value=float(self.calls), dtype=np.float32)
        return chunk

    def build_warm_start(self, previous_chunk, executed_steps):
        del executed_steps
        return previous_chunk


def test_receding_horizon_controller_replans_every_execute_horizon() -> None:
    controller = RecedingHorizonController(DummyPolicy(), observation_spec={"use_goal": True, "use_piano_state": True, "use_sustain_state": True, "use_joint_velocities": True})
    obs = {
        "goal": np.zeros((89,), dtype=np.float32),
        "piano/state": np.zeros((88,), dtype=np.float32),
        "piano/sustain_state": np.zeros((1,), dtype=np.float32),
        "rh_shadow_hand/joints_pos": np.zeros((24,), dtype=np.float32),
        "lh_shadow_hand/joints_pos": np.zeros((24,), dtype=np.float32),
    }
    first = controller.action(obs)
    second = controller.action(obs)
    third = controller.action(obs)
    assert np.allclose(first, 1.0)
    assert np.allclose(second, 1.0)
    assert np.allclose(third, 2.0)
