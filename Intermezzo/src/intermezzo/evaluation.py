from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from intermezzo.constants import HAND_STATE_DIM, NUM_PIANO_KEYS
from intermezzo.fingertip_assignment import assign_active_fingertips
from intermezzo.kinematics import FakeHandKinematics, HandKinematics


def compute_interpolation_scale_metrics(
    *,
    planned_timestep_count: int,
    target_state_count: int,
    internal_planned_timestep_count: int | None = None,
) -> dict[str, Any]:
    """Quantify how much temporal upsampling Intermezzo performs.

    `target_state_count` is the number of sparse hand states supplied to the
    planner, usually one per keyset or onset waypoint. Planned timesteps are the
    dense frames emitted for RoboPianist playback.
    """
    planned_steps = max(int(planned_timestep_count), 0)
    target_states = max(int(target_state_count), 0)
    interpolated_steps = max(planned_steps - target_states, 0)
    metrics = {
        "intermezzo_planned_timestep_count": planned_steps,
        "intermezzo_target_state_count": target_states,
        "intermezzo_interpolated_timestep_count": interpolated_steps,
        "intermezzo_interpolated_timestep_fraction": float(interpolated_steps / planned_steps) if planned_steps else 0.0,
        "intermezzo_upsampling_factor": float(planned_steps / target_states) if target_states else 0.0,
    }
    if internal_planned_timestep_count is not None:
        internal_steps = max(int(internal_planned_timestep_count), 0)
        metrics.update(
            {
                "intermezzo_internal_planned_timestep_count": internal_steps,
                "intermezzo_internal_interpolated_timestep_count": max(internal_steps - target_states, 0),
                "intermezzo_internal_upsampling_factor": float(internal_steps / target_states) if target_states else 0.0,
            }
        )
    return metrics


def compute_two_state_metrics(
    *,
    planned_hand_joints: np.ndarray,
    endpoint_hand_joints: np.ndarray,
    final_target_keys: np.ndarray,
    key_xy: np.ndarray,
    kinematics: HandKinematics | None = None,
    robot_pressed_keys: np.ndarray | None = None,
    control_timestep: float = 0.05,
    threshold: float = 0.5,
) -> dict[str, Any]:
    planned = np.asarray(planned_hand_joints, dtype=np.float32)
    endpoints = np.asarray(endpoint_hand_joints, dtype=np.float32)
    if planned.ndim != 2 or planned.shape[1] != HAND_STATE_DIM:
        raise ValueError(f"planned_hand_joints must have shape [T, 46], got {planned.shape}")
    if endpoints.ndim != 2 or endpoints.shape[1] != HAND_STATE_DIM or endpoints.shape[0] < 2:
        raise ValueError(f"endpoint_hand_joints must have shape [2+, 46], got {endpoints.shape}")
    kin = kinematics or FakeHandKinematics()
    assignments = assign_active_fingertips(
        final_target_keys,
        endpoint_hand_state=planned[-1],
        key_xy=key_xy,
        kinematics=kin,
        threshold=threshold,
    )
    active_errors: list[np.ndarray] = []
    fingertip_positions: list[np.ndarray] = []
    for frame in planned:
        kin.set_hand_state(frame)
        xy = np.asarray(kin.fingertip_xy(), dtype=np.float32)[:10, :2]
        fingertip_positions.append(xy)
        if assignments:
            errors = [float(np.linalg.norm(xy[item.fingertip_index] - item.key_xy)) for item in assignments]
            active_errors.append(np.asarray(errors, dtype=np.float32))
    if active_errors:
        error_matrix = np.stack(active_errors, axis=0)
        mean_error = float(np.mean(error_matrix))
        final_error = float(np.mean(error_matrix[-1]))
        min_error = float(np.min(error_matrix))
    else:
        mean_error = final_error = min_error = 0.0

    positions = np.stack(fingertip_positions, axis=0) if fingertip_positions else np.zeros((0, 10, 2), dtype=np.float32)
    active_fingers = {item.fingertip_index for item in assignments}
    inactive = [idx for idx in range(10) if idx not in active_fingers]
    if positions.shape[0] and inactive:
        inactive_drift = float(np.mean(np.linalg.norm(positions[:, inactive, :] - positions[0:1, inactive, :], axis=2)))
    else:
        inactive_drift = 0.0

    diffs = np.diff(planned, axis=0) if planned.shape[0] > 1 else np.zeros((0, HAND_STATE_DIM), dtype=np.float32)
    step_norms = np.linalg.norm(diffs, axis=1) if diffs.size else np.zeros((0,), dtype=np.float32)
    dt = max(float(control_timestep), 1e-8)
    velocities = diffs / dt if diffs.size else np.zeros((0, HAND_STATE_DIM), dtype=np.float32)
    velocity_norms = np.linalg.norm(velocities, axis=1) if velocities.size else np.zeros((0,), dtype=np.float32)

    final_target = np.asarray(final_target_keys, dtype=np.float32).reshape(-1)[:NUM_PIANO_KEYS] > float(threshold)
    intended_count = int(np.count_nonzero(final_target))
    if robot_pressed_keys is None:
        pressed_any = np.zeros((NUM_PIANO_KEYS,), dtype=bool)
    else:
        pressed = np.asarray(robot_pressed_keys, dtype=np.float32)
        if pressed.ndim == 1:
            pressed_any = pressed[:NUM_PIANO_KEYS] > float(threshold)
        else:
            pressed_any = np.any(pressed[:, :NUM_PIANO_KEYS] > float(threshold), axis=0)
    unintended = int(np.count_nonzero(pressed_any & ~final_target))
    missed = int(np.count_nonzero(final_target & ~pressed_any))

    return {
        "mean_active_fingertip_xy_error": mean_error,
        "final_active_fingertip_xy_error": final_error,
        "min_active_fingertip_xy_error": min_error,
        "inactive_fingertip_drift": inactive_drift,
        "endpoint_joint_l2_delta": float(np.linalg.norm(planned[-1] - endpoints[-1])),
        "max_joint_step": float(np.max(step_norms)) if step_norms.size else 0.0,
        "max_joint_velocity": float(np.max(velocity_norms)) if velocity_norms.size else 0.0,
        "intended_key_press_count": intended_count,
        "unintended_key_press_count": unintended,
        "missed_key_count": missed,
        "active_assignment_count": int(len(assignments)),
    }


def save_metrics(path: str | Path, metrics: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return target
