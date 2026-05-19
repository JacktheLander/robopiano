from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PlanCorruptionConfig:
    enabled: bool = False
    q_reference_noise_std: float = 0.0
    q_smooth_drift_std: float = 0.0
    fingertip_xy_noise_std: float = 0.0
    fingertip_z_noise_std: float = 0.0
    hover_height_bias: float = 0.0
    press_depth_bias: float = 0.0
    timing_jitter_frames: int = 0
    missing_waypoint_probability: float = 0.0
    lookahead_dropout_probability: float = 0.0
    curriculum_scale: float = 1.0

    def __post_init__(self) -> None:
        _validate_non_negative(self.q_reference_noise_std, name="q_reference_noise_std")
        _validate_non_negative(self.q_smooth_drift_std, name="q_smooth_drift_std")
        _validate_non_negative(self.fingertip_xy_noise_std, name="fingertip_xy_noise_std")
        _validate_non_negative(self.fingertip_z_noise_std, name="fingertip_z_noise_std")
        _validate_non_negative(self.press_depth_bias, name="press_depth_bias")
        _validate_non_negative(self.timing_jitter_frames, name="timing_jitter_frames")
        _validate_probability(self.missing_waypoint_probability, name="missing_waypoint_probability")
        _validate_probability(self.lookahead_dropout_probability, name="lookahead_dropout_probability")
        _validate_non_negative(self.curriculum_scale, name="curriculum_scale")

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PlanCorruptionConfig":
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise TypeError(f"Plan corruption config must be a mapping, got {type(data)!r}")
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def scaled_float(self, value: float) -> float:
        return float(value) * float(self.curriculum_scale)

    def scaled_probability(self, value: float) -> float:
        return float(min(1.0, max(0.0, self.scaled_float(value))))

    def scaled_frames(self, frames: int) -> int:
        return int(round(float(frames) * float(self.curriculum_scale)))


def _validate_non_negative(value: float | int, *, name: str) -> None:
    if float(value) < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def _validate_probability(value: float, *, name: str) -> None:
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")
