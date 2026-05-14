from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from bagatelle.assignment import NUM_FINGERS, assign_fingers_previous_pose
from bagatelle.config import BagatelleConfig
from bagatelle.kinematics import HAND_STATE_DIM, BagatelleKinematics, IKResult
from bagatelle.paths import ensure_repo_paths

ensure_repo_paths()
from intermezzo.constants import LEFT_FOREARM_TY_INDEX, RIGHT_FOREARM_TY_INDEX  # noqa: E402
from intermezzo.keys import extract_waypoint_frames, validate_target_keys  # noqa: E402
from intermezzo.planner import PlannerConfig, compute_hand_velocities, plan_between_waypoints  # noqa: E402


IK_METRIC_COLUMNS = (
    "success",
    "optimizer_success",
    "nfev",
    "optimizer_cost",
    "mean_assigned_distance",
    "max_assigned_distance",
    "residual_norm",
)


@dataclass(frozen=True)
class BagatelleTrajectory:
    target_keys: np.ndarray
    waypoint_frames: np.ndarray
    waypoint_target_keys: np.ndarray
    waypoint_hand_joints: np.ndarray
    planned_hand_joints: np.ndarray
    planned_hand_velocities: np.ndarray
    segment_ids: np.ndarray
    assignments: np.ndarray
    assignment_costs: np.ndarray
    fingertip_targets: np.ndarray
    waypoint_fingertips: np.ndarray
    unassigned_keys: np.ndarray
    ik_metrics: np.ndarray
    metadata: dict[str, Any]

    def npz_payload(self) -> dict[str, np.ndarray]:
        return {
            "target_keys": self.target_keys,
            "waypoint_frames": self.waypoint_frames,
            "waypoint_target_keys": self.waypoint_target_keys,
            "waypoint_hand_joints": self.waypoint_hand_joints,
            "planned_hand_joints": self.planned_hand_joints,
            "planned_hand_velocities": self.planned_hand_velocities,
            "segment_ids": self.segment_ids,
            "assignments": self.assignments,
            "assignment_costs": self.assignment_costs,
            "fingertip_targets": self.fingertip_targets,
            "waypoint_fingertips": self.waypoint_fingertips,
            "unassigned_keys": self.unassigned_keys,
            "ik_metrics": self.ik_metrics,
            "ik_metric_columns": np.asarray(IK_METRIC_COLUMNS),
        }


def _planner_config(config: BagatelleConfig) -> PlannerConfig:
    return PlannerConfig(
        control_timestep=float(config.control_timestep),
        threshold=float(config.threshold),
        clearance_height=float(config.clearance_height),
        lift_fraction=float(config.lift_fraction),
        descent_fraction=float(config.descent_fraction),
        vertical_min=float(config.vertical_min),
        vertical_max=float(config.vertical_max),
    )


def _empty_unassigned(rows: list[np.ndarray]) -> np.ndarray:
    width = max((int(row.size) for row in rows), default=0)
    out = np.full((len(rows), width), -1, dtype=np.int32)
    for index, row in enumerate(rows):
        if row.size:
            out[index, : row.size] = row.astype(np.int32)
    return out


def _ik_metric_row(result: IKResult) -> np.ndarray:
    mean_distance = float(np.mean(result.assigned_distances)) if result.assigned_distances.size else 0.0
    return np.asarray(
        [
            float(result.success),
            float(result.optimizer_success),
            float(result.nfev),
            float(result.optimizer_cost),
            mean_distance,
            float(result.max_residual),
            float(result.residual_norm),
        ],
        dtype=np.float32,
    )


def _metadata_from_results(
    *,
    config: BagatelleConfig,
    target_keys: np.ndarray,
    waypoint_frames: np.ndarray,
    ik_results: list[IKResult],
    kinematics: Any,
) -> dict[str, Any]:
    max_residuals = np.asarray([result.max_residual for result in ik_results], dtype=np.float32)
    unassigned_total = int(sum(int(result.unassigned_keys.size) for result in ik_results))
    return {
        "planner": "bagatelle_ot_ik_previous_pose",
        "config": config.to_dict(),
        "target_keys_shape": list(target_keys.shape),
        "num_waypoints": int(waypoint_frames.size),
        "waypoint_frames": waypoint_frames.astype(int).tolist(),
        "finger_order": "left_hand_sites_then_right_hand_sites",
        "joint_order": "right_hand_joints_then_left_hand_joints",
        "right_forearm_ty_index": int(RIGHT_FOREARM_TY_INDEX),
        "left_forearm_ty_index": int(LEFT_FOREARM_TY_INDEX),
        "ik_metric_columns": list(IK_METRIC_COLUMNS),
        "ik_success_count": int(sum(bool(result.success) for result in ik_results)),
        "ik_optimizer_success_count": int(sum(bool(result.optimizer_success) for result in ik_results)),
        "ik_unassigned_key_count": unassigned_total,
        "ik_max_residual_mean": float(max_residuals.mean()) if max_residuals.size else 0.0,
        "ik_max_residual_p95": float(np.percentile(max_residuals, 95)) if max_residuals.size else 0.0,
        "environment_name": str(getattr(kinematics, "environment_name", "")),
        "midi_proto_path": str(getattr(kinematics, "midi_proto_path", "")),
        "load_info": getattr(kinematics, "load_info", {}),
        "waypoint_results": [result.to_dict() for result in ik_results[:500]],
    }


def plan_target_keys(
    target_keys: np.ndarray,
    config: BagatelleConfig | None = None,
    *,
    kinematics: Any | None = None,
) -> BagatelleTrajectory:
    cfg = config or BagatelleConfig()
    keys = validate_target_keys(target_keys)
    waypoint_frames = extract_waypoint_frames(keys, threshold=float(cfg.threshold))
    waypoint_target_keys = keys[waypoint_frames] if waypoint_frames.size else np.zeros((0, 88), dtype=np.float32)

    owns_kinematics = kinematics is None
    kin = kinematics or BagatelleKinematics(cfg, target_keys=keys)
    try:
        previous_qpos = np.asarray(kin.neutral_qpos, dtype=np.float32).copy()
        neutral_qpos = np.asarray(kin.neutral_qpos, dtype=np.float32).copy()
        previous_fingertips = np.asarray(kin.fingertip_positions_for_qpos(previous_qpos), dtype=np.float32)

        waypoint_poses: list[np.ndarray] = []
        waypoint_fingertips: list[np.ndarray] = []
        assignment_rows: list[np.ndarray] = []
        assignment_cost_rows: list[np.ndarray] = []
        fingertip_target_rows: list[np.ndarray] = []
        unassigned_rows: list[np.ndarray] = []
        ik_metric_rows: list[np.ndarray] = []
        ik_results: list[IKResult] = []

        for target_row in waypoint_target_keys:
            active_keys = np.flatnonzero(target_row[:88] > float(cfg.threshold)).astype(np.int32)
            contact_targets = kin.key_contact_targets(active_keys)
            assignment = assign_fingers_previous_pose(active_keys, previous_fingertips, contact_targets, cfg)
            press_targets = kin.key_press_targets(active_keys)
            if assignment.count:
                assignment = replace(
                    assignment,
                    target_positions=press_targets[assignment.assigned_key_positions].astype(np.float32),
                )
            result = kin.solve_press_pose(assignment, previous_qpos, neutral_qpos=neutral_qpos, config=cfg)

            waypoint_poses.append(result.pose.astype(np.float32))
            waypoint_fingertips.append(result.fingertip_positions.astype(np.float32))
            assignment_rows.append(assignment.dense_key_by_finger())
            assignment_cost_rows.append(assignment.dense_cost_by_finger())
            fingertip_target_rows.append(assignment.dense_targets_by_finger())
            unassigned_rows.append(assignment.unassigned_keys.astype(np.int32))
            ik_metric_rows.append(_ik_metric_row(result))
            ik_results.append(result)

            previous_qpos = result.pose.astype(np.float32)
            previous_fingertips = result.fingertip_positions.astype(np.float32)

        if waypoint_poses:
            waypoint_hand_joints = np.stack(waypoint_poses, axis=0).astype(np.float32)
            fingertip_array = np.stack(waypoint_fingertips, axis=0).astype(np.float32)
            assignments = np.stack(assignment_rows, axis=0).astype(np.int32)
            assignment_costs = np.stack(assignment_cost_rows, axis=0).astype(np.float32)
            fingertip_targets = np.stack(fingertip_target_rows, axis=0).astype(np.float32)
            ik_metrics = np.stack(ik_metric_rows, axis=0).astype(np.float32)
            unassigned = _empty_unassigned(unassigned_rows)
            planned, velocities, segment_ids, sanitized = plan_between_waypoints(
                total_steps=int(keys.shape[0]),
                waypoint_frames=waypoint_frames,
                waypoint_target_keys=waypoint_target_keys,
                waypoint_hand_joints=waypoint_hand_joints,
                config=_planner_config(cfg),
            )
            waypoint_hand_joints = sanitized.astype(np.float32)
        else:
            waypoint_hand_joints = np.zeros((0, HAND_STATE_DIM), dtype=np.float32)
            assignments = np.zeros((0, NUM_FINGERS), dtype=np.int32)
            assignment_costs = np.zeros((0, NUM_FINGERS), dtype=np.float32)
            fingertip_targets = np.zeros((0, NUM_FINGERS, 3), dtype=np.float32)
            fingertip_array = np.zeros((0, NUM_FINGERS, 3), dtype=np.float32)
            unassigned = np.zeros((0, 0), dtype=np.int32)
            ik_metrics = np.zeros((0, len(IK_METRIC_COLUMNS)), dtype=np.float32)
            planned = np.tile(neutral_qpos.reshape(1, -1), (keys.shape[0], 1)).astype(np.float32)
            velocities = compute_hand_velocities(planned, control_timestep=float(cfg.control_timestep))
            segment_ids = np.full((keys.shape[0],), -1, dtype=np.int32)

        metadata = _metadata_from_results(
            config=cfg,
            target_keys=keys,
            waypoint_frames=waypoint_frames,
            ik_results=ik_results,
            kinematics=kin,
        )
        metadata["planned_hand_joints_shape"] = list(planned.shape)
        metadata["planned_hand_velocities_shape"] = list(velocities.shape)

        return BagatelleTrajectory(
            target_keys=keys,
            waypoint_frames=waypoint_frames,
            waypoint_target_keys=waypoint_target_keys,
            waypoint_hand_joints=waypoint_hand_joints,
            planned_hand_joints=planned.astype(np.float32),
            planned_hand_velocities=velocities.astype(np.float32),
            segment_ids=segment_ids.astype(np.int32),
            assignments=assignments,
            assignment_costs=assignment_costs,
            fingertip_targets=fingertip_targets,
            waypoint_fingertips=fingertip_array,
            unassigned_keys=unassigned,
            ik_metrics=ik_metrics,
            metadata=metadata,
        )
    finally:
        if owns_kinematics:
            kin.close()
