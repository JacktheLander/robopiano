from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from etude.controllers.base import TrajectoryFollower
from etude.controllers.pd_gain_utils import (
    apply_phase_gain_scale,
    build_grouped_gains,
    canonicalize_phase_key,
    expand_gain,
    smooth_action,
)
from etude.data.trajectory_io import finite_difference
from etude.robopianist.state_mapping import StateMapping


DEFAULT_JOINT_GROUPS: dict[str, list[int]] = {
    "left_arm": list(range(0, 7)),
    "right_arm": list(range(7, 14)),
    "left_hand": list(range(14, 30)),
    "right_hand": list(range(30, 46)),
}


@dataclass
class PDStats:
    steps: int = 0
    clipped_steps: int = 0
    unclipped_l2: list[float] = field(default_factory=list)
    phase_counts: Counter[str] = field(default_factory=Counter)

    @property
    def clip_rate(self) -> float:
        return 0.0 if self.steps == 0 else self.clipped_steps / self.steps

    @property
    def mean_unclipped_l2(self) -> float:
        return float(np.mean(self.unclipped_l2)) if self.unclipped_l2 else 0.0


class ScheduledPDController(TrajectoryFollower):
    """Configurable PD tracker with gain scheduling and optional smoothing."""

    SUPPORTED_GAIN_MODES = ("scalar", "per_joint", "grouped", "phase_scheduled")

    def __init__(
        self,
        mapping: StateMapping,
        *,
        kp: float | Sequence[float] | np.ndarray = 12.0,
        kd: float | Sequence[float] | np.ndarray = 0.6,
        mode: str = "scalar",
        lookahead_steps: int = 1,
        action_clip: bool = True,
        joint_groups: Mapping[str, Sequence[int]] | None = None,
        kp_groups: Mapping[str, float | Sequence[float] | np.ndarray] | None = None,
        kd_groups: Mapping[str, float | Sequence[float] | np.ndarray] | None = None,
        phase_kp_scales: Mapping[str, float | Sequence[float] | np.ndarray] | None = None,
        phase_kd_scales: Mapping[str, float | Sequence[float] | np.ndarray] | None = None,
        action_smoothing: Mapping[str, Any] | None = None,
    ) -> None:
        self.mapping = mapping
        self.mode = str(mode)
        if self.mode not in self.SUPPORTED_GAIN_MODES:
            raise ValueError(f"Unsupported gain mode '{self.mode}'")

        self.lookahead_steps = int(lookahead_steps)
        self.action_clip = bool(action_clip)
        self.joint_groups = dict(DEFAULT_JOINT_GROUPS if joint_groups is None else joint_groups)
        self.phase_kp_scales = dict(phase_kp_scales or {})
        self.phase_kd_scales = dict(phase_kd_scales or {})

        smoothing_cfg = dict(action_smoothing or {})
        self.smoothing_enabled = bool(smoothing_cfg.get("enabled", False))
        self.smoothing_alpha = float(smoothing_cfg.get("alpha", 0.2))

        self.base_kp = self._build_gains(kp, kp_groups, name="kp")
        self.base_kd = self._build_gains(kd, kd_groups, name="kd")

        self.q_ref: np.ndarray | None = None
        self.qdot_ref: np.ndarray | None = None
        self.metadata: dict[str, Any] = {}
        self.phase_schedule: list[str] | None = None
        self.previous_action: np.ndarray | None = None
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

        metadata = metadata or {}
        self.q_ref = q_ref
        self.qdot_ref = (
            finite_difference(q_ref, float(metadata.get("dt", 0.005)))
            if qdot_ref is None
            else np.asarray(qdot_ref, dtype=np.float32)
        )
        if self.qdot_ref.shape != q_ref.shape:
            raise ValueError("qdot_ref must match q_ref shape")

        self.metadata = metadata
        self.phase_schedule = self._extract_phase_schedule(metadata, q_ref.shape[0])
        self.previous_action = None
        self.stats = PDStats()

    def act(self, obs: dict[str, np.ndarray], t: int) -> np.ndarray:
        if self.q_ref is None or self.qdot_ref is None:
            raise RuntimeError("ScheduledPDController.reset must be called before act")

        q = np.asarray(obs["q"], dtype=np.float32).reshape(-1)
        qdot = np.asarray(obs["qdot"], dtype=np.float32).reshape(-1)
        if q.shape != (46,) or qdot.shape != (46,):
            raise ValueError(f"obs q/qdot must have shape [46], got {q.shape}, {qdot.shape}")

        ref_idx = int(np.clip(t + self.lookahead_steps, 0, self.q_ref.shape[0] - 1))
        phase = self._phase_at(ref_idx, obs)
        kp = apply_phase_gain_scale(self.base_kp, phase, self.phase_kp_scales, name="kp")
        kd = apply_phase_gain_scale(self.base_kd, phase, self.phase_kd_scales, name="kd")

        joint_command = kp * (self.q_ref[ref_idx] - q) + kd * (self.qdot_ref[ref_idx] - qdot)
        unclipped_action = _map_without_clip(self.mapping, joint_command)
        if self.smoothing_enabled:
            unclipped_action = smooth_action(unclipped_action, self.previous_action, self.smoothing_alpha)
        action = self.mapping.clip_action(unclipped_action) if self.action_clip else unclipped_action.astype(np.float32)

        self.stats.steps += 1
        self.stats.unclipped_l2.append(float(np.linalg.norm(unclipped_action)))
        if phase is not None:
            self.stats.phase_counts[str(phase)] += 1
        if not np.allclose(unclipped_action, action):
            self.stats.clipped_steps += 1

        self.previous_action = action.astype(np.float32, copy=True)
        return action

    def diagnostics(self) -> dict[str, float]:
        metrics: dict[str, float] = {
            "control/action_clip_rate": self.stats.clip_rate,
            "control/unclipped_action_l2": self.stats.mean_unclipped_l2,
            "control/gain_mode": float(self.SUPPORTED_GAIN_MODES.index(self.mode)),
            "control/lookahead_steps": float(self.lookahead_steps),
            "control/smoothing_alpha": float(self.smoothing_alpha if self.smoothing_enabled else 0.0),
        }
        for phase, count in sorted(self.stats.phase_counts.items()):
            metrics[f"control/phase_count/{phase}"] = float(count)
        return metrics

    def _build_gains(
        self,
        base_gain: float | Sequence[float] | np.ndarray,
        grouped_gains: Mapping[str, float | Sequence[float] | np.ndarray] | None,
        *,
        name: str,
    ) -> np.ndarray:
        if self.mode in {"scalar", "per_joint", "phase_scheduled"}:
            return expand_gain(base_gain, name=name)
        if self.mode == "grouped":
            return build_grouped_gains(
                grouped_gains or {},
                self.joint_groups,
                default_gain=base_gain,
                name=name,
            )
        raise ValueError(f"Unsupported gain mode '{self.mode}'")

    @staticmethod
    def _extract_phase_schedule(metadata: Mapping[str, Any], horizon: int) -> list[str] | None:
        for key in ("phase_schedule", "phases"):
            value = metadata.get(key)
            if value is None:
                continue
            phases = np.asarray(value).reshape(-1)
            if phases.size != horizon:
                raise ValueError(f"{key} must have length {horizon}, got {phases.size}")
            return [canonicalize_phase_key(item) for item in phases.tolist()]
        return None

    def _phase_at(self, ref_idx: int, obs: Mapping[str, Any]) -> str | None:
        if self.phase_schedule is not None:
            return self.phase_schedule[ref_idx]
        phase = obs.get("phase", self.metadata.get("phase"))
        return None if phase is None else canonicalize_phase_key(phase)


def _map_without_clip(mapping: StateMapping, command_46: np.ndarray) -> np.ndarray:
    command_46 = np.asarray(command_46, dtype=np.float32).reshape(-1)
    if command_46.shape != (46,):
        raise ValueError(f"command_46 must have shape [46], got {command_46.shape}")
    if mapping.action_indices is None:
        action = np.zeros(mapping.action_dim, dtype=np.float32)
        n = min(mapping.action_dim, command_46.size)
        action[:n] = command_46[:n]
        return action
    return command_46[np.asarray(mapping.action_indices, dtype=np.int64)].astype(np.float32)
