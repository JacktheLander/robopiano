from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ImpromptuConfig:
    control_timestep: float = 0.05
    threshold: float = 0.5
    environment_name: str = "RoboPianist-debug-TwinkleTwinkleLittleStar-v0"
    seed: int = 0
    reduced_action_space: bool = True

    # Fingertip trajectory planning.
    interpolation_substeps: int = 4
    approach_s: float = 0.08
    hold_s: float = 0.02
    release_s: float = 0.06
    clearance_height: float = 0.02
    key_press_depth: float = 0.008

    # IK anchor selection.
    solve_contact_window_only: bool = True
    anchor_stride: int = 2
    include_midpoint_anchors: bool = True

    # IK objective.
    ik_fingertip_weight: float = 1.0
    ik_smoothness_weight: float = 0.05
    ik_neutral_weight: float = 0.01
    ik_max_nfev: int = 40
    ik_ftol: float = 1e-5
    ik_xtol: float = 1e-5
    ik_gtol: float = 1e-5
    residual_success_threshold: float = 0.02

    output_root: str = "/WAVE/datasets/ccoelho_lab-jlanders/Impromptu/runs"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
