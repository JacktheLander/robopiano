from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from etude.controllers.base import TrajectoryFollower
from etude.controllers.contact_policy import ContactPlanBundle, ContactPolicy, ContactPolicyOutput, IdentityContactPolicy
from etude.controllers.hierarchical import HierarchicalContactController


@dataclass
class FakeContactPolicy(ContactPolicy):
    time_offset: float = 0.0

    def __post_init__(self) -> None:
        self.reset_bundle: ContactPlanBundle | None = None
        self.calls: list[tuple[dict[str, np.ndarray], int]] = []

    def reset(self, plan_bundle: ContactPlanBundle) -> None:
        self.reset_bundle = plan_bundle

    def act(self, obs: dict[str, np.ndarray], t: int) -> ContactPolicyOutput:
        self.calls.append((obs, t))
        assert self.reset_bundle is not None
        return ContactPolicyOutput(
            corrected_fingertip_target=np.asarray(self.reset_bundle.fingertip_targets[t], dtype=np.float32),
            press_intent=np.asarray(self.reset_bundle.press_intent[t], dtype=np.float32),
            release_intent=np.asarray(self.reset_bundle.release_intent[t], dtype=np.float32),
            local_time_offset=self.time_offset,
            diagnostics={"contact_policy/test_calls": float(len(self.calls))},
        )


class FakeLowLevelController(TrajectoryFollower):
    def __init__(self) -> None:
        self.reset_calls: list[tuple[np.ndarray, np.ndarray | None, dict[str, object] | None]] = []
        self.act_calls: list[tuple[dict[str, np.ndarray], int]] = []

    def reset(
        self,
        q_ref: np.ndarray,
        qdot_ref: np.ndarray | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        self.reset_calls.append((q_ref, qdot_ref, metadata))

    def act(self, obs: dict[str, np.ndarray], t: int) -> np.ndarray:
        self.act_calls.append((obs, t))
        return np.full(4, float(t), dtype=np.float32)

    def diagnostics(self) -> dict[str, float]:
        return {"low_level/calls": float(len(self.act_calls))}


def test_hierarchical_controller_calls_both_components_and_merges_diagnostics() -> None:
    contact_policy = FakeContactPolicy(time_offset=1.0)
    low_level = FakeLowLevelController()
    controller = HierarchicalContactController(contact_policy=contact_policy, low_level_controller=low_level)

    q_ref = np.zeros((3, 46), dtype=np.float32)
    qdot_ref = np.zeros((3, 46), dtype=np.float32)
    metadata = {
        "fingertip_targets": np.arange(18, dtype=np.float32).reshape(3, 6),
        "press_intent": np.ones((3, 2), dtype=np.float32),
        "release_intent": np.zeros((3, 2), dtype=np.float32),
    }
    controller.reset(q_ref, qdot_ref, metadata=metadata)

    action = controller.act({"q": np.zeros(46, dtype=np.float32)}, 1)

    assert contact_policy.reset_bundle is not None
    assert len(contact_policy.calls) == 1
    assert len(low_level.reset_calls) == 1
    assert len(low_level.act_calls) == 1
    act_obs, act_t = low_level.act_calls[0]
    assert act_t == 2
    assert np.allclose(act_obs["corrected_fingertip_target"], metadata["fingertip_targets"][1])
    assert np.allclose(act_obs["press_intent"], metadata["press_intent"][1])
    assert np.allclose(act_obs["release_intent"], metadata["release_intent"][1])
    assert action.shape == (4,)
    assert np.allclose(action, np.full(4, 2.0, dtype=np.float32))

    diagnostics = controller.diagnostics()
    assert diagnostics["low_level/calls"] == 1.0
    assert diagnostics["contact_policy/test_calls"] == 1.0
    assert diagnostics["hierarchical/has_contact_policy"] == 1.0


def test_identity_contact_policy_preserves_targets_and_intents() -> None:
    policy = IdentityContactPolicy()
    bundle = ContactPlanBundle(
        q_ref=np.zeros((2, 46), dtype=np.float32),
        fingertip_targets=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        press_intent=np.array([[1.0], [0.0]], dtype=np.float32),
        release_intent=np.array([[0.0], [1.0]], dtype=np.float32),
        local_time_offset=np.array([0.0, 0.0], dtype=np.float32),
    )
    policy.reset(bundle)

    output = policy.act({}, 1)

    assert np.allclose(output.corrected_fingertip_target, bundle.fingertip_targets[1])
    assert np.allclose(output.press_intent, bundle.press_intent[1])
    assert np.allclose(output.release_intent, bundle.release_intent[1])
    assert float(output.local_time_offset) == 0.0
    assert output.diagnostics["contact_policy/is_identity"] == 1.0


def test_hierarchical_controller_accepts_low_level_factory() -> None:
    low_level = FakeLowLevelController()
    controller = HierarchicalContactController(
        contact_policy=IdentityContactPolicy(),
        low_level_controller=lambda plan_bundle: low_level,
    )
    controller.reset(np.zeros((1, 46), dtype=np.float32), np.zeros((1, 46), dtype=np.float32), metadata={})

    action = controller.act({"q": np.zeros(46, dtype=np.float32)}, 0)

    assert action.shape == (4,)
    assert len(low_level.reset_calls) == 1
