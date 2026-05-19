from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class ContactPlanBundle:
    """Reference bundle consumed by contact-aware controller layers."""

    q_ref: np.ndarray
    qdot_ref: np.ndarray | None = None
    fingertip_targets: np.ndarray | None = None
    press_intent: np.ndarray | None = None
    release_intent: np.ndarray | None = None
    local_time_offset: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ContactPolicyOutput:
    """Mid-level policy outputs for one controller step."""

    corrected_fingertip_target: np.ndarray | None = None
    press_intent: np.ndarray | None = None
    release_intent: np.ndarray | None = None
    local_time_offset: np.ndarray | float | None = None
    diagnostics: dict[str, float] = field(default_factory=dict)


class ContactPolicy(ABC):
    """Interface for mid-level contact policies."""

    @abstractmethod
    def reset(self, plan_bundle: ContactPlanBundle) -> None:
        """Set the plan bundle used for future contact-policy decisions."""

    @abstractmethod
    def act(self, obs: dict[str, np.ndarray], t: int) -> ContactPolicyOutput:
        """Return contact-aware corrections for timestep ``t``."""


class IdentityContactPolicy(ContactPolicy):
    """Default no-op policy that forwards bundled targets and intents unchanged."""

    def __init__(self) -> None:
        self.plan_bundle: ContactPlanBundle | None = None

    def reset(self, plan_bundle: ContactPlanBundle) -> None:
        self.plan_bundle = plan_bundle

    def act(self, obs: dict[str, np.ndarray], t: int) -> ContactPolicyOutput:
        del obs
        if self.plan_bundle is None:
            raise RuntimeError("IdentityContactPolicy.reset must be called before act")

        target = _slice_optional(self.plan_bundle.fingertip_targets, t)
        press_intent = _slice_optional(self.plan_bundle.press_intent, t)
        release_intent = _slice_optional(self.plan_bundle.release_intent, t)
        time_offset = _slice_optional(self.plan_bundle.local_time_offset, t)
        diagnostics = {
            "contact_policy/is_identity": 1.0,
            "contact_policy/has_corrected_target": float(target is not None),
            "contact_policy/has_press_intent": float(press_intent is not None),
            "contact_policy/has_release_intent": float(release_intent is not None),
            "contact_policy/has_local_time_offset": float(time_offset is not None),
        }
        return ContactPolicyOutput(
            corrected_fingertip_target=target,
            press_intent=press_intent,
            release_intent=release_intent,
            local_time_offset=time_offset,
            diagnostics=diagnostics,
        )


def _slice_optional(value: np.ndarray | None, t: int) -> np.ndarray | float | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        return float(array)
    if array.shape[0] == 0:
        raise ValueError("Contact policy sequences must have at least one timestep")
    index = int(np.clip(t, 0, array.shape[0] - 1))
    item = array[index]
    if np.asarray(item).ndim == 0:
        return float(item)
    return np.asarray(item, dtype=np.float32)
