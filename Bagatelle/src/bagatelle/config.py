from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BagatelleConfig:
    control_timestep: float = 0.05
    threshold: float = 0.5
    environment_name: str = "RoboPianist-debug-TwinkleTwinkleLittleStar-v0"
    seed: int = 0
    reduced_action_space: bool = True

    # IK objective weights. Values are residual multipliers before least-squares.
    ik_fingertip_weight: float = 1.0
    ik_smoothness_weight: float = 0.05
    ik_neutral_weight: float = 0.01
    ik_max_nfev: int = 120
    ik_ftol: float = 1e-5
    ik_xtol: float = 1e-5
    ik_gtol: float = 1e-5
    residual_success_threshold: float = 0.02

    # RoboPianist's fingering reward uses this same key contact heuristic.
    key_target_front_offset: float = 0.35
    key_target_top_offset: float = 0.5
    key_press_depth: float = 0.008

    # Inter-waypoint planning parameters.
    clearance_height: float = 0.02
    lift_fraction: float = 0.20
    descent_fraction: float = 0.35
    vertical_min: float = 0.0
    vertical_max: float = 0.06

    # Evaluation-only settle steps after restoring each direct pose.
    settle_steps: int = 3

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
