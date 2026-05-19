from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from etude.controllers.base import TrajectoryFollower
from etude.data.trajectory_io import finite_difference
from etude.robopianist.state_mapping import StateMapping


@dataclass
class PDStats:
    steps: int = 0
    clipped_steps: int = 0
    unclipped_l2: list[float] = field(default_factory=list)

    @property
    def clip_rate(self) -> float:
        return 0.0 if self.steps == 0 else self.clipped_steps / self.steps

    @property
    def mean_unclipped_l2(self) -> float:
        return float(np.mean(self.unclipped_l2)) if self.unclipped_l2 else 0.0


class PDController(TrajectoryFollower):
    """Proportional-derivative tracker over 46D references."""

    def __init__(
        self,
        mapping: StateMapping,
        kp: float | np.ndarray = 12.0,
        kd: float | np.ndarray = 0.6,
        lookahead: int = 1,
        clip: bool = True,
    ) -> None:
        self.mapping = mapping
        self.kp = _expand_gain(kp)
        self.kd = _expand_gain(kd)
        self.lookahead = int(lookahead)
        self.clip = bool(clip)
        self.q_ref: np.ndarray | None = None
        self.qdot_ref: np.ndarray | None = None
        self.metadata: dict[str, Any] = {}
        self.stats = PDStats()

    def reset(
        self,
        q_ref: np.ndarray,
        qdot_ref: np.ndarray | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        q_ref = np.asarray(q_ref, dtype=np.float32)
        if q_ref.ndim != 2 or q_ref.shape[1] != 46:
            raise ValueError(f"q_ref must have shape [T, 46], got {q_ref.shape}")
        self.q_ref = q_ref
        self.qdot_ref = (
            finite_difference(q_ref, float((metadata or {}).get("dt", 0.005)))
            if qdot_ref is None
            else np.asarray(qdot_ref, dtype=np.float32)
        )
        if self.qdot_ref.shape != q_ref.shape:
            raise ValueError("qdot_ref must match q_ref shape")
        self.metadata = metadata or {}
        self.stats = PDStats()

    def act(self, obs: dict[str, np.ndarray], t: int) -> np.ndarray:
        if self.q_ref is None or self.qdot_ref is None:
            raise RuntimeError("PDController.reset must be called before act")
        q = np.asarray(obs["q"], dtype=np.float32).reshape(-1)
        qdot = np.asarray(obs["qdot"], dtype=np.float32).reshape(-1)
        if q.shape != (46,) or qdot.shape != (46,):
            raise ValueError(f"obs q/qdot must have shape [46], got {q.shape}, {qdot.shape}")

        ref_idx = int(np.clip(t + self.lookahead, 0, self.q_ref.shape[0] - 1))
        joint_command = self.kp * (self.q_ref[ref_idx] - q) + self.kd * (self.qdot_ref[ref_idx] - qdot)
        action = self.mapping.action_from_joint_command(joint_command)
        unclipped = _map_without_clip(self.mapping, joint_command)
        clipped = self.mapping.clip_action(unclipped) if self.clip else unclipped.astype(np.float32)

        self.stats.steps += 1
        self.stats.unclipped_l2.append(float(np.linalg.norm(unclipped)))
        if not np.allclose(unclipped, clipped):
            self.stats.clipped_steps += 1
        return clipped

    def diagnostics(self) -> dict[str, float]:
        return {
            "control/action_clip_rate": self.stats.clip_rate,
            "control/unclipped_action_l2": self.stats.mean_unclipped_l2,
        }


def _expand_gain(gain: float | np.ndarray) -> np.ndarray:
    gain_array = np.asarray(gain, dtype=np.float32)
    if gain_array.ndim == 0:
        return np.full(46, float(gain_array), dtype=np.float32)
    if gain_array.shape != (46,):
        raise ValueError(f"gain must be scalar or shape [46], got {gain_array.shape}")
    return gain_array


def _map_without_clip(mapping: StateMapping, command_46: np.ndarray) -> np.ndarray:
    command_46 = np.asarray(command_46, dtype=np.float32).reshape(-1)
    if mapping.action_indices is None:
        action = np.zeros(mapping.action_dim, dtype=np.float32)
        n = min(mapping.action_dim, command_46.size)
        action[:n] = command_46[:n]
        return action
    return command_46[np.asarray(mapping.action_indices, dtype=np.int64)].astype(np.float32)
