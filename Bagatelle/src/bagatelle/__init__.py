"""Bagatelle: deterministic OT + IK press-pose planning for RoboPianist."""

from bagatelle.assignment import FingerAssignmentResult, assign_fingers_previous_pose
from bagatelle.config import BagatelleConfig
from bagatelle.kinematics import BagatelleKinematics, IKResult
from bagatelle.planner import BagatelleTrajectory, plan_target_keys

__all__ = [
    "BagatelleConfig",
    "BagatelleKinematics",
    "BagatelleTrajectory",
    "FingerAssignmentResult",
    "IKResult",
    "assign_fingers_previous_pose",
    "plan_target_keys",
]
