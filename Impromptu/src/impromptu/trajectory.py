from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


REQUIRED_TRAJECTORY_FIELDS = (
    "target_keys",
    "waypoint_frames",
    "waypoint_target_keys",
    "assignments",
    "assignment_costs",
    "fingertip_targets",
    "waypoint_fingertips",
    "unassigned_keys",
    "fingertip_trajectory_targets",
    "fingertip_trajectory_weights",
    "fingertip_trajectory_dense_frames",
    "ik_anchor_frames_dense",
    "ik_anchor_frames_control",
    "ik_anchor_qpos",
    "ik_anchor_fingertips",
    "ik_anchor_metrics",
    "waypoint_hand_joints",
    "planned_hand_joints",
    "planned_hand_velocities",
    "planned_hand_joints_dense",
    "planned_hand_velocities_dense",
    "segment_ids",
    "segment_ids_dense",
)


@dataclass(frozen=True)
class ImpromptuTrajectory:
    target_keys: np.ndarray
    waypoint_frames: np.ndarray
    waypoint_target_keys: np.ndarray
    assignments: np.ndarray
    assignment_costs: np.ndarray
    fingertip_targets: np.ndarray
    waypoint_fingertips: np.ndarray
    unassigned_keys: np.ndarray
    fingertip_trajectory_targets: np.ndarray
    fingertip_trajectory_weights: np.ndarray
    fingertip_trajectory_dense_frames: np.ndarray
    ik_anchor_frames_dense: np.ndarray
    ik_anchor_frames_control: np.ndarray
    ik_anchor_qpos: np.ndarray
    ik_anchor_fingertips: np.ndarray
    ik_anchor_metrics: np.ndarray
    waypoint_hand_joints: np.ndarray
    planned_hand_joints: np.ndarray
    planned_hand_velocities: np.ndarray
    planned_hand_joints_dense: np.ndarray
    planned_hand_velocities_dense: np.ndarray
    segment_ids: np.ndarray
    segment_ids_dense: np.ndarray
    metadata: dict[str, Any]

    def npz_payload(self) -> dict[str, np.ndarray]:
        return {name: getattr(self, name) for name in REQUIRED_TRAJECTORY_FIELDS}
