from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np


def _as_vector(value: np.ndarray | None, *, name: str) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    return array


def _stack_history(history: Iterable[np.ndarray], width: int | None) -> np.ndarray | None:
    items = [np.asarray(item, dtype=np.float32).reshape(-1) for item in history]
    if not items:
        if width is None:
            return None
        return np.zeros((0, width), dtype=np.float32)
    return np.stack(items, axis=0).astype(np.float32)


@dataclass(frozen=True, slots=True)
class TemporalContextSnapshot:
    feature_history: np.ndarray
    action_history: np.ndarray | None
    residual_history: np.ndarray | None


class TemporalContextBuffer:
    """Small NumPy-first rolling buffer for temporal controller context."""

    def __init__(
        self,
        history_steps: int,
        *,
        store_actions: bool = True,
        store_residuals: bool = True,
    ) -> None:
        if history_steps <= 0:
            raise ValueError("history_steps must be positive")
        self.history_steps = int(history_steps)
        self.store_actions = bool(store_actions)
        self.store_residuals = bool(store_residuals)
        self._feature_dim: int | None = None
        self._action_dim: int | None = None
        self._residual_dim: int | None = None
        self._features: deque[np.ndarray] = deque(maxlen=self.history_steps)
        self._actions: deque[np.ndarray] = deque(maxlen=self.history_steps)
        self._residuals: deque[np.ndarray] = deque(maxlen=self.history_steps)

    def reset(self) -> None:
        self._features.clear()
        self._actions.clear()
        self._residuals.clear()

    def __len__(self) -> int:
        return len(self._features)

    @property
    def is_empty(self) -> bool:
        return len(self) == 0

    def append(
        self,
        features: np.ndarray,
        *,
        action: np.ndarray | None = None,
        residual: np.ndarray | None = None,
    ) -> None:
        feature_vec = _as_vector(features, name="features")
        assert feature_vec is not None
        self._feature_dim = self._check_dim(feature_vec, self._feature_dim, name="features")
        self._features.append(feature_vec)

        if self.store_actions:
            action_vec = _as_vector(action, name="action")
            if action_vec is None:
                action_vec = self._zeros(self._action_dim)
            if action_vec is not None:
                self._action_dim = self._check_dim(action_vec, self._action_dim, name="action")
                self._actions.append(action_vec)

        if self.store_residuals:
            residual_vec = _as_vector(residual, name="residual")
            if residual_vec is None:
                residual_vec = self._zeros(self._residual_dim)
            if residual_vec is not None:
                self._residual_dim = self._check_dim(residual_vec, self._residual_dim, name="residual")
                self._residuals.append(residual_vec)

    def snapshot(self) -> TemporalContextSnapshot:
        return TemporalContextSnapshot(
            feature_history=self.feature_history(),
            action_history=self.action_history(),
            residual_history=self.residual_history(),
        )

    def feature_history(self) -> np.ndarray:
        history = _stack_history(self._features, self._feature_dim)
        if history is None:
            return np.zeros((0, 0), dtype=np.float32)
        return history

    def action_history(self) -> np.ndarray | None:
        if not self.store_actions:
            return None
        return _stack_history(self._actions, self._action_dim)

    def residual_history(self) -> np.ndarray | None:
        if not self.store_residuals:
            return None
        return _stack_history(self._residuals, self._residual_dim)

    def _check_dim(self, value: np.ndarray, current: int | None, *, name: str) -> int:
        width = int(value.shape[0])
        if current is not None and current != width:
            raise ValueError(f"{name} dimension changed from {current} to {width}")
        return width

    @staticmethod
    def _zeros(width: int | None) -> np.ndarray | None:
        if width is None:
            return None
        return np.zeros(width, dtype=np.float32)
