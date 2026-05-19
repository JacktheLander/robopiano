from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _vector(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    return array


def _optional_vector(value: Any, *, name: str) -> np.ndarray | None:
    if value is None:
        return None
    return _vector(value, name=name)


@dataclass(slots=True)
class RuntimeObservation:
    q: np.ndarray
    qdot: np.ndarray
    fingertips: np.ndarray | None = None
    key_state: np.ndarray | None = None
    target_keys: np.ndarray | None = None
    previous_action: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.q = _vector(self.q, name="q")
        self.qdot = _vector(self.qdot, name="qdot")
        if self.q.shape != self.qdot.shape:
            raise ValueError(f"q and qdot must match, got {self.q.shape} and {self.qdot.shape}")
        self.fingertips = _optional_vector(self.fingertips, name="fingertips")
        self.key_state = _optional_vector(self.key_state, name="key_state")
        self.target_keys = _optional_vector(self.target_keys, name="target_keys")
        self.previous_action = _optional_vector(self.previous_action, name="previous_action")

    @classmethod
    def from_dict(cls, obs: dict[str, Any]) -> "RuntimeObservation":
        return cls(
            q=obs["q"],
            qdot=obs["qdot"],
            fingertips=obs.get("fingertips"),
            key_state=obs.get("key_state"),
            target_keys=obs.get("target_keys"),
            previous_action=obs.get("previous_action"),
        )

    def as_dict(self) -> dict[str, np.ndarray]:
        payload: dict[str, np.ndarray] = {
            "q": self.q,
            "qdot": self.qdot,
        }
        for name in ("fingertips", "key_state", "target_keys", "previous_action"):
            value = getattr(self, name)
            if value is not None:
                payload[name] = value
        return payload
