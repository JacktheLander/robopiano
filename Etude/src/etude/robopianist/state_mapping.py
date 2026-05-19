from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _default_indices(n: int) -> list[int]:
    return list(range(n))


@dataclass
class StateMapping:
    qpos_indices_46: list[int]
    qvel_indices_46: list[int]
    action_indices: list[int] | None
    fingertip_indices: list[int] | None
    key_state_indices: list[int] | None
    action_low: np.ndarray
    action_high: np.ndarray

    @classmethod
    def from_action_spec(
        cls,
        action_spec: Any,
        qpos_indices_46: list[int] | None = None,
        qvel_indices_46: list[int] | None = None,
        action_indices: list[int] | None = None,
        fingertip_indices: list[int] | None = None,
        key_state_indices: list[int] | None = None,
    ) -> "StateMapping":
        low = np.asarray(getattr(action_spec, "minimum", -np.ones(46)), dtype=np.float32).reshape(-1)
        high = np.asarray(getattr(action_spec, "maximum", np.ones_like(low)), dtype=np.float32).reshape(-1)
        if low.shape != high.shape:
            raise ValueError("action_spec minimum/maximum shapes differ")
        return cls(
            qpos_indices_46=qpos_indices_46 or _default_indices(46),
            qvel_indices_46=qvel_indices_46 or _default_indices(46),
            action_indices=action_indices,
            fingertip_indices=fingertip_indices,
            key_state_indices=key_state_indices,
            action_low=low,
            action_high=high,
        )

    @property
    def action_dim(self) -> int:
        return int(self.action_low.size)

    def extract_q(self, source: Any) -> np.ndarray:
        return self._extract_vector(source, ("q", "qpos", "position"), self.qpos_indices_46)

    def extract_qdot(self, source: Any) -> np.ndarray:
        return self._extract_vector(source, ("qdot", "qvel", "velocity"), self.qvel_indices_46)

    def extract_fingertips(self, source: Any) -> np.ndarray | None:
        if self.fingertip_indices is None:
            return None
        return self._extract_vector(source, ("fingertips", "fingertip_positions"), self.fingertip_indices)

    def extract_key_state(self, source: Any) -> np.ndarray | None:
        if self.key_state_indices is None:
            return None
        return self._extract_vector(source, ("key_state", "keys", "piano_keys"), self.key_state_indices)

    def action_from_joint_command(self, command_46: np.ndarray) -> np.ndarray:
        command_46 = np.asarray(command_46, dtype=np.float32).reshape(-1)
        if command_46.shape != (46,):
            raise ValueError(f"command_46 must have shape [46], got {command_46.shape}")
        if self.action_indices is None:
            if command_46.size == self.action_dim:
                action = command_46
            else:
                action = np.zeros(self.action_dim, dtype=np.float32)
                n = min(self.action_dim, command_46.size)
                action[:n] = command_46[:n]
        else:
            action = command_46[np.asarray(self.action_indices, dtype=np.int64)]
        return self.clip_action(action)

    def clip_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != self.action_low.shape:
            raise ValueError(f"action shape {action.shape} does not match {self.action_low.shape}")
        return np.clip(action, self.action_low, self.action_high).astype(np.float32)

    def zero_action(self) -> np.ndarray:
        low = np.where(np.isfinite(self.action_low), self.action_low, -1.0)
        high = np.where(np.isfinite(self.action_high), self.action_high, 1.0)
        return np.clip(np.zeros(self.action_dim, dtype=np.float32), low, high)

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["action_low"] = self.action_low.astype(float).tolist()
        data["action_high"] = self.action_high.astype(float).tolist()
        return data

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "StateMapping":
        return cls(
            qpos_indices_46=list(data["qpos_indices_46"]),
            qvel_indices_46=list(data["qvel_indices_46"]),
            action_indices=data.get("action_indices"),
            fingertip_indices=data.get("fingertip_indices"),
            key_state_indices=data.get("key_state_indices"),
            action_low=np.asarray(data["action_low"], dtype=np.float32),
            action_high=np.asarray(data["action_high"], dtype=np.float32),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json_dict(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "StateMapping":
        return cls.from_json_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @staticmethod
    def _extract_vector(source: Any, names: tuple[str, ...], indices: list[int]) -> np.ndarray:
        data = _find_array(source, names)
        if data is None:
            raise KeyError(f"Could not find any of {names} in observation/state source")
        flat = np.asarray(data, dtype=np.float32).reshape(-1)
        idx = np.asarray(indices, dtype=np.int64)
        if flat.size <= int(idx.max(initial=0)):
            raise ValueError(f"source vector has length {flat.size}, needs index {int(idx.max())}")
        return flat[idx].astype(np.float32)


def _find_array(source: Any, names: tuple[str, ...]) -> Any:
    if isinstance(source, dict):
        for name in names:
            if name in source:
                return source[name]
        for value in source.values():
            found = _find_array(value, names)
            if found is not None:
                return found
        return None
    for name in names:
        if hasattr(source, name):
            return getattr(source, name)
    if hasattr(source, "observation"):
        return _find_array(source.observation, names)
    return None


def resolve_mapping_from_env(env: Any, **kwargs: Any) -> StateMapping:
    """Build a conservative mapping from a RoboPianist-like environment."""
    action_spec = env.action_spec() if callable(getattr(env, "action_spec", None)) else env.action_spec
    return StateMapping.from_action_spec(action_spec, **kwargs)
