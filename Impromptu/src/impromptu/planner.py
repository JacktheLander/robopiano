from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from impromptu.anchor_selection import select_ik_anchor_frames
from impromptu.config import ImpromptuConfig
from impromptu.fingertip_trajectory import NUM_FINGERS, FingertipTrajectory, build_fingertip_trajectory
from impromptu.ik_solver import IK_ANCHOR_METRIC_COLUMNS, ik_metric_row, interpolate_anchor_qpos, solve_fingertip_trajectory_anchors
from impromptu.paths import ensure_repo_paths
from impromptu.trajectory import ImpromptuTrajectory

ensure_repo_paths()
from bagatelle.assignment import assign_fingers_previous_pose  # noqa: E402
from bagatelle.config import BagatelleConfig  # noqa: E402
from bagatelle.kinematics import HAND_STATE_DIM, BagatelleKinematics, IKResult  # noqa: E402
from intermezzo.keys import extract_waypoint_frames, validate_target_keys  # noqa: E402
from intermezzo.planner import compute_hand_velocities  # noqa: E402


def _bagatelle_config(config: ImpromptuConfig) -> BagatelleConfig:
    return BagatelleConfig(
        control_timestep=float(config.control_timestep),
        threshold=float(config.threshold),
        environment_name=str(config.environment_name),
        seed=int(config.seed),
        reduced_action_space=bool(config.reduced_action_space),
        ik_fingertip_weight=float(config.ik_fingertip_weight),
        ik_smoothness_weight=float(config.ik_smoothness_weight),
        ik_neutral_weight=float(config.ik_neutral_weight),
        ik_max_nfev=int(config.ik_max_nfev),
        ik_ftol=float(config.ik_ftol),
        ik_xtol=float(config.ik_xtol),
        ik_gtol=float(config.ik_gtol),
        residual_success_threshold=float(config.residual_success_threshold),
        key_press_depth=float(config.key_press_depth),
        clearance_height=float(config.clearance_height),
    )


def _empty_unassigned(rows: list[np.ndarray]) -> np.ndarray:
    width = max((int(row.size) for row in rows), default=0)
    out = np.full((len(rows), width), -1, dtype=np.int32)
    for index, row in enumerate(rows):
        if row.size:
            out[index, : row.size] = row.astype(np.int32)
    return out


def _active_keys(row: np.ndarray, threshold: float) -> np.ndarray:
    return np.flatnonzero(np.asarray(row, dtype=np.float32)[:88] > float(threshold)).astype(np.int32)


def _sparse_assignment_stage(
    *,
    kin: Any,
    waypoint_target_keys: np.ndarray,
    config: ImpromptuConfig,
    bagatelle_config: BagatelleConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[IKResult]]:
    previous_qpos = np.asarray(kin.neutral_qpos, dtype=np.float32).copy()
    neutral_qpos = np.asarray(kin.neutral_qpos, dtype=np.float32).copy()
    previous_fingertips = np.asarray(kin.fingertip_positions_for_qpos(previous_qpos), dtype=np.float32)

    waypoint_poses: list[np.ndarray] = []
    waypoint_fingertips: list[np.ndarray] = []
    assignment_rows: list[np.ndarray] = []
    assignment_cost_rows: list[np.ndarray] = []
    fingertip_target_rows: list[np.ndarray] = []
    unassigned_rows: list[np.ndarray] = []
    results: list[IKResult] = []

    for target_row in waypoint_target_keys:
        active_keys = _active_keys(target_row, float(config.threshold))
        contact_targets = kin.key_contact_targets(active_keys)
        assignment = assign_fingers_previous_pose(active_keys, previous_fingertips, contact_targets, bagatelle_config)
        press_targets = kin.key_press_targets(active_keys, press_depth=float(config.key_press_depth))
        if assignment.count:
            assignment = replace(
                assignment,
                target_positions=press_targets[assignment.assigned_key_positions].astype(np.float32),
            )
        result = kin.solve_press_pose(assignment, previous_qpos, neutral_qpos=neutral_qpos, config=bagatelle_config)

        waypoint_poses.append(result.pose.astype(np.float32))
        waypoint_fingertips.append(result.fingertip_positions.astype(np.float32))
        assignment_rows.append(assignment.dense_key_by_finger())
        assignment_cost_rows.append(assignment.dense_cost_by_finger())
        fingertip_target_rows.append(assignment.dense_targets_by_finger())
        unassigned_rows.append(assignment.unassigned_keys.astype(np.int32))
        results.append(result)

        previous_qpos = result.pose.astype(np.float32)
        previous_fingertips = result.fingertip_positions.astype(np.float32)

    if waypoint_poses:
        return (
            np.stack(waypoint_poses, axis=0).astype(np.float32),
            np.stack(waypoint_fingertips, axis=0).astype(np.float32),
            np.stack(assignment_rows, axis=0).astype(np.int32),
            np.stack(assignment_cost_rows, axis=0).astype(np.float32),
            np.stack(fingertip_target_rows, axis=0).astype(np.float32),
            _empty_unassigned(unassigned_rows),
            results,
        )
    return (
        np.zeros((0, HAND_STATE_DIM), dtype=np.float32),
        np.zeros((0, NUM_FINGERS, 3), dtype=np.float32),
        np.zeros((0, NUM_FINGERS), dtype=np.int32),
        np.zeros((0, NUM_FINGERS), dtype=np.float32),
        np.zeros((0, NUM_FINGERS, 3), dtype=np.float32),
        np.zeros((0, 0), dtype=np.int32),
        results,
    )


def _metadata(
    *,
    config: ImpromptuConfig,
    target_keys: np.ndarray,
    waypoint_frames: np.ndarray,
    dense_total: int,
    anchor_frames: np.ndarray,
    anchor_results: tuple[IKResult, ...],
    kin: Any,
) -> dict[str, Any]:
    nfev = np.asarray([result.nfev for result in anchor_results], dtype=np.float32)
    max_residuals = np.asarray([result.max_residual for result in anchor_results], dtype=np.float32)
    dense_dt = float(config.control_timestep) / float(max(int(config.interpolation_substeps), 1))
    return {
        "planner": "impromptu_fingertip_intermezzo_bagatelle_ik",
        "target_keys_shape": list(target_keys.shape),
        "num_waypoints": int(waypoint_frames.size),
        "waypoint_frames": waypoint_frames.astype(int).tolist(),
        "control_timestep": float(config.control_timestep),
        "dense_control_timestep": dense_dt,
        "interpolation_substeps": int(config.interpolation_substeps),
        "num_dense_frames": int(dense_total),
        "num_ik_anchor_frames": int(anchor_frames.size),
        "ik_anchor_fraction": float(anchor_frames.size / dense_total) if dense_total else 0.0,
        "ik_success_count": int(sum(bool(result.success) for result in anchor_results)),
        "ik_optimizer_success_count": int(sum(bool(result.optimizer_success) for result in anchor_results)),
        "ik_nfev_mean": float(nfev.mean()) if nfev.size else 0.0,
        "ik_nfev_p95": float(np.percentile(nfev, 95)) if nfev.size else 0.0,
        "ik_max_residual_mean": float(max_residuals.mean()) if max_residuals.size else 0.0,
        "ik_max_residual_p95": float(np.percentile(max_residuals, 95)) if max_residuals.size else 0.0,
        "environment_name": str(getattr(kin, "environment_name", config.environment_name)),
        "midi_proto_path": str(getattr(kin, "midi_proto_path", "")),
        "output_root": str(config.output_root),
        "config": config.to_dict(),
        "ik_anchor_metric_columns": list(IK_ANCHOR_METRIC_COLUMNS),
    }


def plan_target_keys(
    target_keys: np.ndarray,
    config: ImpromptuConfig | None = None,
    *,
    kinematics: BagatelleKinematics | None = None,
) -> ImpromptuTrajectory:
    cfg = config or ImpromptuConfig()
    keys = validate_target_keys(target_keys)
    waypoint_frames = extract_waypoint_frames(keys, threshold=float(cfg.threshold))
    waypoint_target_keys = keys[waypoint_frames] if waypoint_frames.size else np.zeros((0, 88), dtype=np.float32)
    bag_cfg = _bagatelle_config(cfg)

    owns_kinematics = kinematics is None
    kin = kinematics or BagatelleKinematics(bag_cfg, target_keys=keys)
    try:
        neutral_qpos = np.asarray(kin.neutral_qpos, dtype=np.float32).copy()
        (
            waypoint_hand_joints,
            waypoint_fingertips,
            assignments,
            assignment_costs,
            fingertip_targets,
            unassigned_keys,
            sparse_results,
        ) = _sparse_assignment_stage(
            kin=kin,
            waypoint_target_keys=waypoint_target_keys,
            config=cfg,
            bagatelle_config=bag_cfg,
        )

        substeps = max(int(cfg.interpolation_substeps), 1)
        if waypoint_frames.size:
            fingertip_traj = build_fingertip_trajectory(
                total_steps=int(keys.shape[0]),
                waypoint_frames=waypoint_frames,
                assignments=assignments,
                fingertip_targets=fingertip_targets,
                waypoint_fingertips=waypoint_fingertips,
                config=cfg,
            )
        else:
            dense_total_empty = int(keys.shape[0]) * substeps
            fingertip_traj = FingertipTrajectory(
                targets=np.full((dense_total_empty, NUM_FINGERS, 3), np.nan, dtype=np.float32),
                weights=np.zeros((dense_total_empty, NUM_FINGERS), dtype=np.float32),
                dense_frames=np.arange(dense_total_empty, dtype=np.int64),
            )

        dense_total = int(fingertip_traj.targets.shape[0])
        waypoint_frames_dense = np.clip(waypoint_frames.astype(np.int64) * substeps, 0, max(dense_total - 1, 0)).astype(np.int64)
        anchor_frames = select_ik_anchor_frames(
            fingertip_targets=fingertip_traj.targets,
            fingertip_weights=fingertip_traj.weights,
            waypoint_frames_dense=waypoint_frames_dense,
            config=cfg,
        )
        if dense_total == 0:
            anchor_solution_qpos = np.zeros((0, HAND_STATE_DIM), dtype=np.float32)
            anchor_solution_tips = np.zeros((0, NUM_FINGERS, 3), dtype=np.float32)
            anchor_metrics = np.zeros((0, len(IK_ANCHOR_METRIC_COLUMNS)), dtype=np.float32)
            anchor_results: tuple[IKResult, ...] = ()
        else:
            initial_qpos = waypoint_hand_joints[0] if waypoint_hand_joints.size else neutral_qpos
            solution = solve_fingertip_trajectory_anchors(
                kin=kin,
                fingertip_targets=fingertip_traj.targets,
                fingertip_weights=fingertip_traj.weights,
                anchor_frames=anchor_frames,
                initial_qpos=initial_qpos,
                neutral_qpos=neutral_qpos,
                config=cfg,
            )
            anchor_solution_qpos = solution.qpos
            anchor_solution_tips = solution.fingertips
            anchor_metrics = solution.metrics
            anchor_results = solution.results

        planned_dense, segment_ids_dense = interpolate_anchor_qpos(
            anchor_frames=anchor_frames,
            anchor_qpos=anchor_solution_qpos,
            dense_total=dense_total,
        )
        planned = planned_dense[::substeps][: keys.shape[0]].astype(np.float32, copy=True)
        if planned.shape[0] < keys.shape[0]:
            pad_source = planned[-1] if planned.size else neutral_qpos
            pad = np.tile(pad_source.reshape(1, -1), (keys.shape[0] - planned.shape[0], 1))
            planned = np.concatenate([planned, pad.astype(np.float32)], axis=0)
        segment_ids = segment_ids_dense[::substeps][: keys.shape[0]].astype(np.int32, copy=True)
        if segment_ids.shape[0] < keys.shape[0]:
            pad_value = int(segment_ids[-1]) if segment_ids.size else -1
            segment_ids = np.concatenate(
                [segment_ids, np.full((keys.shape[0] - segment_ids.shape[0],), pad_value, dtype=np.int32)],
                axis=0,
            )
        dense_dt = float(cfg.control_timestep) / float(substeps)
        planned_velocities = compute_hand_velocities(planned, control_timestep=float(cfg.control_timestep))
        dense_velocities = compute_hand_velocities(planned_dense, control_timestep=dense_dt)

        sparse_metrics = (
            np.stack([ik_metric_row(result) for result in sparse_results], axis=0).astype(np.float32)
            if sparse_results
            else np.zeros((0, len(IK_ANCHOR_METRIC_COLUMNS)), dtype=np.float32)
        )
        metadata = _metadata(
            config=cfg,
            target_keys=keys,
            waypoint_frames=waypoint_frames,
            dense_total=dense_total,
            anchor_frames=anchor_frames,
            anchor_results=anchor_results,
            kin=kin,
        )
        metadata["sparse_press_ik_metric_columns"] = list(IK_ANCHOR_METRIC_COLUMNS)
        metadata["sparse_press_ik_metrics_shape"] = list(sparse_metrics.shape)
        metadata["sparse_press_ik_success_count"] = int(sum(bool(result.success) for result in sparse_results))

        return ImpromptuTrajectory(
            target_keys=keys,
            waypoint_frames=waypoint_frames,
            waypoint_target_keys=waypoint_target_keys,
            assignments=assignments,
            assignment_costs=assignment_costs,
            fingertip_targets=fingertip_targets,
            waypoint_fingertips=waypoint_fingertips,
            unassigned_keys=unassigned_keys,
            fingertip_trajectory_targets=fingertip_traj.targets,
            fingertip_trajectory_weights=fingertip_traj.weights,
            fingertip_trajectory_dense_frames=fingertip_traj.dense_frames,
            ik_anchor_frames_dense=anchor_frames,
            ik_anchor_frames_control=(anchor_frames // substeps).astype(np.int64),
            ik_anchor_qpos=anchor_solution_qpos,
            ik_anchor_fingertips=anchor_solution_tips,
            ik_anchor_metrics=anchor_metrics,
            waypoint_hand_joints=waypoint_hand_joints,
            planned_hand_joints=planned,
            planned_hand_velocities=planned_velocities.astype(np.float32),
            planned_hand_joints_dense=planned_dense.astype(np.float32),
            planned_hand_velocities_dense=dense_velocities.astype(np.float32),
            segment_ids=segment_ids,
            segment_ids_dense=segment_ids_dense.astype(np.int32),
            metadata=metadata,
        )
    finally:
        if owns_kinematics:
            kin.close()
