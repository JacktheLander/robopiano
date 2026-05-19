from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PhaseGatingConfig:
    enabled: bool = False
    approach: float = 1.0
    pre_contact: float = 1.0
    contact: float = 1.0
    hold: float = 1.0
    release: float = 1.0

    def scale_for_phase(self, phase: Any) -> float:
        if not self.enabled:
            return 1.0
        name = canonicalize_phase_name(phase)
        if name == "approach":
            return float(self.approach)
        if name == "pre_contact":
            return float(self.pre_contact)
        if name == "contact":
            return float(self.contact)
        if name == "hold":
            return float(self.hold)
        if name == "release":
            return float(self.release)
        return 1.0


@dataclass(frozen=True)
class ResidualSafetyConfig:
    scale: float = 1.0
    clip_norm: float | None = None
    clip_per_dim: float | np.ndarray | None = None
    smoothing_alpha: float = 0.0
    phase_gating: PhaseGatingConfig = field(default_factory=PhaseGatingConfig)


def canonicalize_phase_name(phase: Any) -> str | None:
    if phase is None:
        return None
    if isinstance(phase, bytes):
        phase = phase.decode("utf-8")
    if isinstance(phase, np.ndarray):
        if phase.size == 0:
            return None
        phase = phase.reshape(-1)[0].item()
    if isinstance(phase, str):
        normalized = phase.strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "approach": "approach",
            "attack": "approach",
            "precontact": "pre_contact",
            "pre_contact": "pre_contact",
            "contact": "contact",
            "press": "contact",
            "hold": "hold",
            "sustain": "hold",
            "release": "release",
            "recovery": "recovery",
            "recover": "recovery",
            "idle": "unknown",
            "unknown": "unknown",
        }
        return aliases.get(normalized)
    if isinstance(phase, (int, float, np.integer, np.floating)):
        bins = (
            "approach",
            "pre_contact",
            "contact",
            "hold",
            "release",
            "recovery",
            "unknown",
        )
        value = float(phase)
        if 0.0 <= value <= 1.0:
            idx = min(int(value * len(bins)), len(bins) - 1)
            return bins[idx]
    return None


def resolve_phase(obs: dict[str, Any], metadata: dict[str, Any] | None = None, t: int | None = None) -> Any:
    metadata = metadata or {}
    for key in ("residual_phase", "controller_phase", "phase"):
        if key in obs:
            return obs[key]
    if "phase" in metadata:
        phase = metadata["phase"]
        if isinstance(phase, (list, tuple, np.ndarray)) and t is not None:
            return phase[min(max(t, 0), len(phase) - 1)]
        return phase
    return None


def scale_residual(residual: np.ndarray, scale: float) -> np.ndarray:
    return np.asarray(residual, dtype=np.float32) * np.float32(scale)


def clip_residual_per_dim(
    residual: np.ndarray, clip_per_dim: float | np.ndarray | None
) -> np.ndarray:
    residual = np.asarray(residual, dtype=np.float32)
    if clip_per_dim is None:
        return residual.copy()
    limit = np.asarray(clip_per_dim, dtype=np.float32)
    if limit.ndim == 0:
        limit = np.full(residual.shape, float(abs(limit)), dtype=np.float32)
    elif limit.shape != residual.shape:
        raise ValueError(f"clip_per_dim shape {limit.shape} does not match residual shape {residual.shape}")
    limit = np.abs(limit)
    return np.clip(residual, -limit, limit).astype(np.float32)


def clip_residual_norm(residual: np.ndarray, clip_norm: float | None) -> np.ndarray:
    residual = np.asarray(residual, dtype=np.float32)
    if clip_norm is None:
        return residual.copy()
    max_norm = float(clip_norm)
    if max_norm <= 0:
        return np.zeros_like(residual, dtype=np.float32)
    norm = float(np.linalg.norm(residual))
    if norm <= max_norm or norm == 0.0:
        return residual.copy()
    return (residual * np.float32(max_norm / norm)).astype(np.float32)


def low_pass_filter_residual(
    residual: np.ndarray, previous_residual: np.ndarray | None, alpha: float
) -> np.ndarray:
    residual = np.asarray(residual, dtype=np.float32)
    alpha = float(alpha)
    if previous_residual is None or alpha <= 0.0:
        return residual.copy()
    if alpha >= 1.0:
        return np.asarray(previous_residual, dtype=np.float32).copy()
    previous = np.asarray(previous_residual, dtype=np.float32)
    if previous.shape != residual.shape:
        raise ValueError(f"previous residual shape {previous.shape} does not match {residual.shape}")
    return (alpha * previous + (1.0 - alpha) * residual).astype(np.float32)


def phase_gate_residual(
    residual: np.ndarray,
    phase: Any,
    config: PhaseGatingConfig | None,
) -> tuple[np.ndarray, float]:
    config = config or PhaseGatingConfig()
    gate = config.scale_for_phase(phase)
    return (np.asarray(residual, dtype=np.float32) * np.float32(gate)).astype(np.float32), float(gate)


def residual_diagnostics(
    raw_residual: np.ndarray,
    pre_clip_residual: np.ndarray,
    processed_residual: np.ndarray,
    final_action_unclipped: np.ndarray,
    final_action: np.ndarray,
    phase_gate: float,
) -> dict[str, float]:
    raw = np.asarray(raw_residual, dtype=np.float32)
    pre_clip = np.asarray(pre_clip_residual, dtype=np.float32)
    processed = np.asarray(processed_residual, dtype=np.float32)
    action_unclipped = np.asarray(final_action_unclipped, dtype=np.float32)
    action = np.asarray(final_action, dtype=np.float32)
    if raw.shape != processed.shape or pre_clip.shape != processed.shape:
        raise ValueError("residual arrays must share the same shape")
    residual_changed = np.not_equal(pre_clip, processed)
    action_changed = np.not_equal(action_unclipped, action)
    return {
        "control/raw_residual_norm": float(np.linalg.norm(raw)),
        "control/clipped_residual_norm": float(np.linalg.norm(processed)),
        "control/residual_clip_fraction": float(np.mean(residual_changed.astype(np.float32))),
        "control/final_action_clip_fraction": float(np.mean(action_changed.astype(np.float32))),
        "control/phase_gate": float(phase_gate),
    }


class ResidualSafetyProcessor:
    def __init__(self, config: ResidualSafetyConfig | None = None) -> None:
        self.config = config or ResidualSafetyConfig()
        self.previous_residual: np.ndarray | None = None

    def reset(self) -> None:
        self.previous_residual = None

    def process(
        self,
        residual: np.ndarray,
        *,
        phase: Any = None,
    ) -> tuple[np.ndarray, dict[str, float]]:
        raw = np.asarray(residual, dtype=np.float32).reshape(-1)
        scaled = scale_residual(raw, self.config.scale)
        gated, gate = phase_gate_residual(scaled, phase, self.config.phase_gating)
        per_dim = clip_residual_per_dim(gated, self.config.clip_per_dim)
        clipped = clip_residual_norm(per_dim, self.config.clip_norm)
        smoothed = low_pass_filter_residual(clipped, self.previous_residual, self.config.smoothing_alpha)
        self.previous_residual = smoothed
        diagnostics = {
            "control/raw_residual_norm": float(np.linalg.norm(raw)),
            "control/clipped_residual_norm": float(np.linalg.norm(smoothed)),
            "control/residual_clip_fraction": float(
                np.mean(np.not_equal(gated, clipped).astype(np.float32))
            ),
            "control/phase_gate": float(gate),
        }
        return smoothed.astype(np.float32), diagnostics
