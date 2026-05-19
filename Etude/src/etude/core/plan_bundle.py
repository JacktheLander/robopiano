from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from etude.core.typing import MetadataDict


def _optional_array(value: Any, *, ndim: int | None = None, name: str) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got {array.ndim}")
    return array


@dataclass(slots=True)
class PlanBundle:
    q_ref: np.ndarray
    qdot_ref: np.ndarray | None = None
    dt: float = 0.005
    target_keys: np.ndarray | None = None
    key_state: np.ndarray | None = None
    fingertip_ref: np.ndarray | None = None
    fingertip_weights: np.ndarray | None = None
    assignments: Any | None = None
    phase: np.ndarray | None = None
    planner_confidence: np.ndarray | float | None = None
    metadata: MetadataDict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.q_ref = np.asarray(self.q_ref, dtype=np.float32)
        if self.q_ref.ndim != 2:
            raise ValueError(f"q_ref must have shape [T, D], got {self.q_ref.shape}")
        self.qdot_ref = _optional_array(self.qdot_ref, ndim=2, name="qdot_ref")
        if self.qdot_ref is not None and self.qdot_ref.shape != self.q_ref.shape:
            raise ValueError("qdot_ref must match q_ref shape")
        self.target_keys = _optional_array(self.target_keys, ndim=2, name="target_keys")
        self.key_state = _optional_array(self.key_state, ndim=2, name="key_state")
        self.fingertip_ref = _optional_array(self.fingertip_ref, name="fingertip_ref")
        self.fingertip_weights = _optional_array(self.fingertip_weights, name="fingertip_weights")
        self.phase = _optional_array(self.phase, name="phase")
        self.planner_confidence = _optional_array(self.planner_confidence, name="planner_confidence")
        self.dt = float(self.dt)
        if self.dt <= 0.0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        self.metadata = dict(self.metadata)

    @property
    def horizon(self) -> int:
        return int(self.q_ref.shape[0])

    @property
    def dof(self) -> int:
        return int(self.q_ref.shape[1])

    def validate_step_aligned(self) -> None:
        expected = self.horizon
        for name, value in (
            ("target_keys", self.target_keys),
            ("key_state", self.key_state),
            ("fingertip_ref", self.fingertip_ref),
            ("phase", self.phase),
            ("planner_confidence", self.planner_confidence),
        ):
            if value is not None and value.ndim > 0 and value.shape[0] != expected:
                raise ValueError(f"{name} must have leading dimension {expected}, got {value.shape}")

    @property
    def current_key_state(self) -> np.ndarray | None:
        return self.key_state

    @property
    def desired_keys(self) -> np.ndarray | None:
        return self.target_keys

    @property
    def goal_keys(self) -> np.ndarray | None:
        return self.target_keys

    def with_metadata(self, **metadata: Any) -> "PlanBundle":
        merged = dict(self.metadata)
        merged.update(metadata)
        return PlanBundle(
            q_ref=self.q_ref,
            qdot_ref=self.qdot_ref,
            dt=self.dt,
            target_keys=self.target_keys,
            key_state=self.key_state,
            fingertip_ref=self.fingertip_ref,
            fingertip_weights=self.fingertip_weights,
            assignments=self.assignments,
            phase=self.phase,
            planner_confidence=self.planner_confidence,
            metadata=merged,
        )
