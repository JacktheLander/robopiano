from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from etude.training.replay_buffer import ReplayBuffer, Transition


FailurePredicate = Callable[[dict[str, Any]], bool]


def default_failure_predicate(frame: dict[str, Any]) -> bool:
    return bool(frame.get("is_failure", False))


@dataclass(slots=True)
class FailureWindow:
    frames: list[dict[str, Any]]
    start_index: int
    end_index: int


@dataclass(slots=True)
class FailureWindowCollector:
    window_size: int = 8
    predicate: FailurePredicate = default_failure_predicate
    _recent_frames: deque[dict[str, Any]] = field(init=False)
    _failure_windows: list[FailureWindow] = field(default_factory=list, init=False)
    _frame_index: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        self._recent_frames = deque(maxlen=self.window_size)

    def observe(self, frame: dict[str, Any]) -> bool:
        snapshot = _copy_frame(frame)
        self._recent_frames.append(snapshot)
        triggered = self.predicate(snapshot)
        if triggered:
            start = max(0, self._frame_index - len(self._recent_frames) + 1)
            self._failure_windows.append(
                FailureWindow(
                    frames=[_copy_frame(item) for item in self._recent_frames],
                    start_index=start,
                    end_index=self._frame_index,
                )
            )
        self._frame_index += 1
        return triggered

    def export_samples(self, max_windows: int | None = None) -> list[dict[str, Any]]:
        windows = self._failure_windows if max_windows is None else self._failure_windows[:max_windows]
        return [
            {
                "start_index": window.start_index,
                "end_index": window.end_index,
                "frames": [_copy_frame(frame) for frame in window.frames],
            }
            for window in windows
        ]

    def __len__(self) -> int:
        return len(self._failure_windows)


@dataclass(slots=True)
class DAggerHook:
    collector: FailureWindowCollector
    replay_buffer: ReplayBuffer | None = None

    def observe_frame(self, frame: dict[str, Any]) -> bool:
        return self.collector.observe(frame)

    def store_failure_window(self, window: list[dict[str, Any]]) -> None:
        if self.replay_buffer is None:
            return
        for index, frame in enumerate(window):
            obs = dict(frame.get("obs", {}))
            next_obs = dict(frame.get("next_obs", obs))
            action = frame.get("action", np.zeros(1, dtype=np.float32))
            transition = Transition(
                obs={key: np.array(value, copy=True) for key, value in obs.items()},
                action=np.array(action, copy=True),
                reward=float(frame.get("reward", 0.0)),
                next_obs={key: np.array(value, copy=True) for key, value in next_obs.items()},
                done=bool(frame.get("done", index == len(window) - 1)),
                is_failure=True,
                metadata={"window_index": np.asarray(index, dtype=np.int64)},
            )
            self.replay_buffer.append(transition)

    def export_in_memory_samples(self, max_windows: int | None = None) -> list[dict[str, Any]]:
        return self.collector.export_samples(max_windows=max_windows)


def _copy_frame(frame: dict[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for key, value in frame.items():
        if isinstance(value, np.ndarray):
            copied[key] = np.array(value, copy=True)
        elif isinstance(value, dict):
            copied[key] = {
                child_key: np.array(child_value, copy=True)
                if isinstance(child_value, np.ndarray)
                else child_value
                for child_key, child_value in value.items()
            }
        else:
            copied[key] = value
    return copied
