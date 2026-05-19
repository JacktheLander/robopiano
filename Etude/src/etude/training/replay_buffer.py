from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np


ArrayDict = dict[str, np.ndarray]
TransitionPayload = dict[str, np.ndarray | float | bool | int]


def _copy_array_dict(values: dict[str, Any]) -> ArrayDict:
    return {key: np.array(value, copy=True) for key, value in values.items()}


def _stack_values(values: list[Any]) -> np.ndarray:
    arrays = [np.asarray(value) for value in values]
    return np.stack(arrays, axis=0)


@dataclass(slots=True)
class Transition:
    obs: ArrayDict
    action: np.ndarray
    reward: float
    next_obs: ArrayDict
    done: bool
    is_failure: bool = False
    metadata: TransitionPayload | None = None

    def to_payload(self) -> TransitionPayload:
        payload: TransitionPayload = {
            "action": np.array(self.action, copy=True),
            "reward": np.asarray(self.reward, dtype=np.float32),
            "done": np.asarray(self.done, dtype=np.bool_),
            "is_failure": np.asarray(self.is_failure, dtype=np.bool_),
        }
        for key, value in self.obs.items():
            payload[f"obs.{key}"] = np.array(value, copy=True)
        for key, value in self.next_obs.items():
            payload[f"next_obs.{key}"] = np.array(value, copy=True)
        if self.metadata:
            for key, value in self.metadata.items():
                payload[f"meta.{key}"] = np.array(value, copy=True)
        return payload


class ReplayBuffer:
    """In-memory CPU replay buffer for NumPy-backed transition dictionaries."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self._items: deque[Transition] = deque(maxlen=self.capacity)

    def append(
        self,
        transition: Transition | None = None,
        *,
        obs: dict[str, Any] | None = None,
        action: Any | None = None,
        reward: float | int = 0.0,
        next_obs: dict[str, Any] | None = None,
        done: bool = False,
        is_failure: bool = False,
        metadata: TransitionPayload | None = None,
    ) -> None:
        if transition is None:
            if obs is None or next_obs is None or action is None:
                raise ValueError("obs, action, and next_obs are required when transition is omitted")
            transition = Transition(
                obs=_copy_array_dict(obs),
                action=np.array(action, copy=True),
                reward=float(reward),
                next_obs=_copy_array_dict(next_obs),
                done=bool(done),
                is_failure=bool(is_failure),
                metadata=None if metadata is None else dict(metadata),
            )
        self._items.append(transition)

    def append_episode(
        self,
        episode: list[Transition | dict[str, Any]],
        *,
        mark_failure: bool | None = None,
    ) -> None:
        for item in episode:
            if isinstance(item, Transition):
                transition = item
            else:
                transition = Transition(
                    obs=_copy_array_dict(item["obs"]),
                    action=np.array(item["action"], copy=True),
                    reward=float(item.get("reward", 0.0)),
                    next_obs=_copy_array_dict(item["next_obs"]),
                    done=bool(item.get("done", False)),
                    is_failure=bool(item.get("is_failure", False)),
                    metadata=dict(item.get("metadata", {})) or None,
                )
            if mark_failure is not None:
                transition = Transition(
                    obs=transition.obs,
                    action=transition.action,
                    reward=transition.reward,
                    next_obs=transition.next_obs,
                    done=transition.done,
                    is_failure=bool(mark_failure),
                    metadata=transition.metadata,
                )
            self._items.append(transition)

    def sample(
        self,
        batch_size: int,
        *,
        rng: np.random.Generator | None = None,
        failure_only: bool = False,
    ) -> list[Transition]:
        items = self._eligible_items(failure_only=failure_only)
        if batch_size > len(items):
            raise ValueError("batch_size exceeds replay buffer size")
        rng = rng or np.random.default_rng()
        indices = rng.choice(len(items), size=batch_size, replace=False)
        return [items[int(i)] for i in indices]

    def sample_batch(
        self,
        batch_size: int,
        *,
        rng: np.random.Generator | None = None,
        failure_only: bool = False,
    ) -> TransitionPayload:
        samples = self.sample(batch_size, rng=rng, failure_only=failure_only)
        payloads = [sample.to_payload() for sample in samples]
        keys = payloads[0].keys()
        return {key: _stack_values([payload[key] for payload in payloads]) for key in keys}

    def as_dict(self, *, failure_only: bool = False) -> TransitionPayload:
        items = list(self._items)
        if failure_only:
            items = [item for item in items if item.is_failure]
        if not items:
            return {}
        payloads = [item.to_payload() for item in items]
        keys = payloads[0].keys()
        return {key: _stack_values([payload[key] for payload in payloads]) for key in keys}

    def _eligible_items(self, *, failure_only: bool) -> list[Transition]:
        items = list(self._items)
        if failure_only:
            items = [item for item in items if item.is_failure]
        if not items:
            message = "no failure transitions are available" if failure_only else "replay buffer is empty"
            raise ValueError(message)
        return items

    def __len__(self) -> int:
        return len(self._items)
