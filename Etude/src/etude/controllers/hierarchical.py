from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from etude.controllers.base import TrajectoryFollower
from etude.controllers.contact_policy import (
    ContactPlanBundle,
    ContactPolicy,
    ContactPolicyOutput,
)


LowLevelFactory = Callable[[ContactPlanBundle], TrajectoryFollower]


class HierarchicalContactController(TrajectoryFollower):
    """Wrapper that applies a contact policy before a low-level tracker."""

    def __init__(
        self,
        contact_policy: ContactPolicy,
        low_level_controller: TrajectoryFollower | LowLevelFactory,
    ) -> None:
        self.contact_policy = contact_policy
        self._low_level_input = low_level_controller
        self.low_level_controller: TrajectoryFollower | None = None
        self.plan_bundle: ContactPlanBundle | None = None
        self._latest_contact_output = ContactPolicyOutput()

    def reset(
        self,
        q_ref: np.ndarray,
        qdot_ref: np.ndarray | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        bundle = ContactPlanBundle(
            q_ref=np.asarray(q_ref, dtype=np.float32),
            qdot_ref=None if qdot_ref is None else np.asarray(qdot_ref, dtype=np.float32),
            fingertip_targets=_optional_float32(metadata, "fingertip_targets"),
            press_intent=_optional_float32(metadata, "press_intent"),
            release_intent=_optional_float32(metadata, "release_intent"),
            local_time_offset=_optional_float32(metadata, "local_time_offset"),
            metadata=dict(metadata or {}),
        )
        self.plan_bundle = bundle
        self.contact_policy.reset(bundle)
        self.low_level_controller = self._resolve_low_level_controller(bundle)
        low_level_metadata = dict(bundle.metadata)
        low_level_metadata["contact_plan_bundle"] = bundle
        self.low_level_controller.reset(bundle.q_ref, bundle.qdot_ref, metadata=low_level_metadata)
        self._latest_contact_output = ContactPolicyOutput()

    def act(self, obs: dict[str, np.ndarray], t: int) -> np.ndarray:
        if self.low_level_controller is None or self.plan_bundle is None:
            raise RuntimeError("HierarchicalContactController.reset must be called before act")

        contact_output = self.contact_policy.act(obs, t)
        self._latest_contact_output = contact_output
        merged_obs = dict(obs)
        if contact_output.corrected_fingertip_target is not None:
            merged_obs["corrected_fingertip_target"] = np.asarray(
                contact_output.corrected_fingertip_target,
                dtype=np.float32,
            )
        if contact_output.press_intent is not None:
            merged_obs["press_intent"] = np.asarray(contact_output.press_intent, dtype=np.float32)
        if contact_output.release_intent is not None:
            merged_obs["release_intent"] = np.asarray(contact_output.release_intent, dtype=np.float32)

        corrected_t = _apply_local_time_offset(t, contact_output.local_time_offset)
        return self.low_level_controller.act(merged_obs, corrected_t)

    def diagnostics(self) -> dict[str, float]:
        if self.low_level_controller is None:
            return dict(self._latest_contact_output.diagnostics)
        merged = {}
        low_level_diagnostics = getattr(self.low_level_controller, "diagnostics", None)
        if callable(low_level_diagnostics):
            merged.update(low_level_diagnostics())
        merged.update(self._latest_contact_output.diagnostics)
        merged["hierarchical/has_contact_policy"] = 1.0
        merged["hierarchical/has_low_level_controller"] = 1.0
        return merged

    def _resolve_low_level_controller(self, bundle: ContactPlanBundle) -> TrajectoryFollower:
        candidate = self._low_level_input(bundle) if callable(self._low_level_input) else self._low_level_input
        if not isinstance(candidate, TrajectoryFollower):
            raise TypeError("low_level_controller must be a TrajectoryFollower or factory returning one")
        return candidate


def _optional_float32(metadata: dict[str, Any] | None, key: str) -> np.ndarray | None:
    if metadata is None or key not in metadata or metadata[key] is None:
        return None
    return np.asarray(metadata[key], dtype=np.float32)


def _apply_local_time_offset(t: int, offset: np.ndarray | float | None) -> int:
    if offset is None:
        return int(t)
    value = np.asarray(offset, dtype=np.float32)
    scalar_offset = float(np.mean(value)) if value.ndim > 0 else float(value)
    return max(0, int(round(float(t) + scalar_offset)))
