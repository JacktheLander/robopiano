from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class ControllerOutput:
    action: np.ndarray
    residual: np.ndarray | None = None
    press_intent: np.ndarray | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.action = np.asarray(self.action, dtype=np.float32).reshape(-1)
        self.residual = None if self.residual is None else np.asarray(self.residual, dtype=np.float32).reshape(-1)
        self.press_intent = (
            None if self.press_intent is None else np.asarray(self.press_intent, dtype=np.float32).reshape(-1)
        )
        self.diagnostics = dict(self.diagnostics)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"action": self.action}
        if self.residual is not None:
            payload["residual"] = self.residual
        if self.press_intent is not None:
            payload["press_intent"] = self.press_intent
        if self.diagnostics:
            payload["diagnostics"] = self.diagnostics
        return payload
