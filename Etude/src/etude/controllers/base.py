from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class TrajectoryFollower(ABC):
    """Interface for closed-loop controllers that follow a q_ref trajectory."""

    @abstractmethod
    def reset(
        self,
        q_ref: np.ndarray,
        qdot_ref: np.ndarray | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Set a new reference trajectory."""

    @abstractmethod
    def act(self, obs: dict[str, np.ndarray], t: int) -> np.ndarray:
        """Return an action accepted by the RoboPianist environment."""
