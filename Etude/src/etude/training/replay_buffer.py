from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class Transition:
    obs: dict
    action: np.ndarray
    reward: float
    next_obs: dict
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self._items: deque[Transition] = deque(maxlen=self.capacity)

    def append(self, transition: Transition) -> None:
        self._items.append(transition)

    def __len__(self) -> int:
        return len(self._items)

    def sample(self, batch_size: int, rng: np.random.Generator | None = None) -> list[Transition]:
        if batch_size > len(self._items):
            raise ValueError("batch_size exceeds replay buffer size")
        rng = rng or np.random.default_rng()
        indices = rng.choice(len(self._items), size=batch_size, replace=False)
        items = list(self._items)
        return [items[int(i)] for i in indices]
