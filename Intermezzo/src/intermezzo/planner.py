from __future__ import annotations

from dataclasses import asdict, dataclass
import inspect
from typing import Callable

import numpy as np

from intermezzo.constants import (
    FOREARM_TY_INDICES,
    FOREARM_TY_MAX,
    FOREARM_TY_MIN,
    HAND_STATE_DIM,
    LEFT_FOREARM_TY_INDEX,
    NUM_PIANO_KEYS,
    RIGHT_FOREARM_TY_INDEX,
)
from intermezzo.evaluation import compute_interpolation_scale_metrics
from intermezzo.fingertip_assignment import FingertipAssignment, assign_active_fingertips
from intermezzo.key_geometry import approximate_key_geometry
from intermezzo.kinematics import FakeHandKinematics, HandKinematics
from intermezzo.keys import extract_waypoint_frames, keyset_hand_sides, validate_target_keys
from intermezzo.magnetic_field import clip_norm, combined_magnetic_weight, frame_window_envelope


WaypointPredictor = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class PlannerConfig:
    control_timestep: float = 0.05
    threshold: float = 0.5
    interpolation_substeps: int = 10
    press_approach_s: float = 0.04
    press_hold_s: float = 0.01
    press_release_s: float = 0.04
    press_envelope_power: float = 2.0
    press_depth: float = 0.0
    clearance_height: float = 0.02
    lift_fraction: float = 0.20
    descent_fraction: float = 0.35
    vertical_min: float = FOREARM_TY_MIN
    vertical_max: float = FOREARM_TY_MAX
    enable_key_magnetism: bool = False
    magnet_radius: float = 0.08
    magnet_sigma: float = 0.03
    magnet_gain: float = 1.0
    magnet_max_xy_step: float = 0.015
    magnet_start_fraction: float = 0.55
    magnet_power: float = 2.0
    ik_damping: float = 1e-3
    ik_max_delta_q: float = 0.03
    ik_iterations_per_frame: int = 1
    preserve_waypoint_endpoints: bool = True
    magnet_only_final_keyset: bool = True


@dataclass(frozen=True)
class PlannedTrajectory:
    target_keys: np.ndarray
    waypoint_frames: np.ndarray
    waypoint_target_keys: np.ndarray
    waypoint_hand_joints: np.ndarray
    planned_hand_joints: np.ndarray
    planned_hand_velocities: np.ndarray
    segment_ids: np.ndarray
    planned_hand_joints_dense: np.ndarray
    planned_hand_velocities_dense: np.ndarray
    segment_ids_dense: np.ndarray
    metadata: dict[str, object]


def build_intermezzo_trajectory(
    target_keys: np.ndarray,
    *,
    predictor: WaypointPredictor,
    config: PlannerConfig | None = None,
    batch_size: int = 256,
    key_geometry: np.ndarray | None = None,
    kinematics: HandKinematics | None = None,
) -> PlannedTrajectory:
    cfg = config or PlannerConfig()
    keys = validate_target_keys(target_keys)
    waypoint_frames = extract_waypoint_frames(keys, threshold=cfg.threshold)
    waypoint_target_keys = keys[waypoint_frames] if waypoint_frames.size else np.zeros((0, NUM_PIANO_KEYS), dtype=np.float32)
    if waypoint_frames.size:
        predicted = np.asarray(_predict_waypoints(predictor, waypoint_target_keys, batch_size=batch_size), dtype=np.float32)
    else:
        predicted = np.zeros((0, HAND_STATE_DIM), dtype=np.float32)

    planned, velocities, segment_ids, sanitized_waypoints, dense, dense_velocities, dense_segment_ids = plan_between_waypoints(
        total_steps=int(keys.shape[0]),
        waypoint_frames=waypoint_frames,
        waypoint_target_keys=waypoint_target_keys,
        waypoint_hand_joints=predicted,
        config=cfg,
        key_geometry=key_geometry,
        kinematics=kinematics,
        return_dense=True,
    )
    metadata: dict[str, object] = {
        "planner": "intermezzo_lift_then_land" if not cfg.enable_key_magnetism else "intermezzo_magnetic_two_state_capable",
        "planner_config": asdict(cfg),
        "batch_size": int(batch_size),
        "target_keys_shape": list(keys.shape),
        "num_waypoints": int(waypoint_frames.size),
        "waypoint_frames": waypoint_frames.astype(int).tolist(),
        "planned_hand_joints_shape": list(planned.shape),
        "planned_hand_velocities_shape": list(velocities.shape),
        "planned_hand_joints_dense_shape": list(dense.shape),
        "planned_hand_velocities_dense_shape": list(dense_velocities.shape),
        "dense_control_timestep": _dense_control_timestep(cfg),
        **compute_interpolation_scale_metrics(
            planned_timestep_count=int(planned.shape[0]),
            target_state_count=int(waypoint_frames.size),
            internal_planned_timestep_count=int(planned.shape[0]) * _interpolation_substeps(cfg),
        ),
    }
    return PlannedTrajectory(
        target_keys=keys,
        waypoint_frames=waypoint_frames,
        waypoint_target_keys=waypoint_target_keys,
        waypoint_hand_joints=sanitized_waypoints,
        planned_hand_joints=planned,
        planned_hand_velocities=velocities,
        segment_ids=segment_ids,
        planned_hand_joints_dense=dense,
        planned_hand_velocities_dense=dense_velocities,
        segment_ids_dense=dense_segment_ids,
        metadata=metadata,
    )


def build_two_state_trajectory(
    keyset_a: np.ndarray,
    keyset_b: np.ndarray,
    *,
    endpoint_hand_joints: np.ndarray,
    num_steps: int,
    config: PlannerConfig | None = None,
    key_geometry: np.ndarray | None = None,
    kinematics: HandKinematics | None = None,
) -> PlannedTrajectory:
    cfg = config or PlannerConfig()
    steps = int(num_steps)
    if steps < 2:
        raise ValueError(f"num_steps must be at least 2, got {num_steps}")
    a = _validate_keyset(keyset_a, name="keyset_a")
    b = _validate_keyset(keyset_b, name="keyset_b")
    endpoints = _sanitize_hand_states(endpoint_hand_joints, cfg)
    if endpoints.shape != (2, HAND_STATE_DIM):
        raise ValueError(f"endpoint_hand_joints must have shape [2, 46], got {endpoints.shape}")
    target_keys = np.zeros((steps, NUM_PIANO_KEYS), dtype=np.float32)
    target_keys[0] = a
    target_keys[1:] = b
    frames = np.asarray([0, steps - 1], dtype=np.int64)
    waypoint_keys = np.stack([a, b], axis=0).astype(np.float32)
    planned, velocities, segment_ids, sanitized, dense, dense_velocities, dense_segment_ids = plan_between_waypoints(
        total_steps=steps,
        waypoint_frames=frames,
        waypoint_target_keys=waypoint_keys,
        waypoint_hand_joints=endpoints,
        config=cfg,
        key_geometry=key_geometry,
        kinematics=kinematics,
        return_dense=True,
    )
    metadata: dict[str, object] = {
        "planner": "intermezzo_two_state",
        "planner_config": asdict(cfg),
        "target_keys_shape": list(target_keys.shape),
        "num_waypoints": 2,
        "waypoint_frames": frames.astype(int).tolist(),
        "planned_hand_joints_shape": list(planned.shape),
        "planned_hand_velocities_shape": list(velocities.shape),
        "planned_hand_joints_dense_shape": list(dense.shape),
        "planned_hand_velocities_dense_shape": list(dense_velocities.shape),
        "dense_control_timestep": _dense_control_timestep(cfg),
        **compute_interpolation_scale_metrics(
            planned_timestep_count=int(planned.shape[0]),
            target_state_count=int(frames.size),
            internal_planned_timestep_count=int(planned.shape[0]) * _interpolation_substeps(cfg),
        ),
    }
    return PlannedTrajectory(
        target_keys=target_keys,
        waypoint_frames=frames,
        waypoint_target_keys=waypoint_keys,
        waypoint_hand_joints=sanitized,
        planned_hand_joints=planned,
        planned_hand_velocities=velocities,
        segment_ids=segment_ids,
        planned_hand_joints_dense=dense,
        planned_hand_velocities_dense=dense_velocities,
        segment_ids_dense=dense_segment_ids,
        metadata=metadata,
    )


def plan_between_waypoints(
    *,
    total_steps: int,
    waypoint_frames: np.ndarray,
    waypoint_target_keys: np.ndarray,
    waypoint_hand_joints: np.ndarray,
    config: PlannerConfig | None = None,
    key_geometry: np.ndarray | None = None,
    kinematics: HandKinematics | None = None,
    return_dense: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    cfg = config or PlannerConfig()
    _validate_config(cfg)
    total = int(total_steps)
    if total < 0:
        raise ValueError(f"total_steps must be non-negative, got {total_steps}")

    frames = np.asarray(waypoint_frames, dtype=np.int64).reshape(-1)
    keys = np.asarray(waypoint_target_keys, dtype=np.float32)
    waypoints = _sanitize_hand_states(waypoint_hand_joints, cfg)
    _validate_waypoints(total, frames, keys, waypoints)

    substeps = _interpolation_substeps(cfg)
    dense_total = total * substeps
    dense = np.zeros((dense_total, HAND_STATE_DIM), dtype=np.float32)
    dense_segment_ids = np.full((dense_total,), -1, dtype=np.int32)
    if total == 0:
        planned = np.zeros((0, HAND_STATE_DIM), dtype=np.float32)
        segment_ids = np.zeros((0,), dtype=np.int32)
        velocities = np.zeros_like(planned)
        if return_dense:
            dense_velocities = np.zeros_like(dense)
            return planned, velocities, segment_ids, waypoints, dense, dense_velocities, dense_segment_ids
        return planned, velocities, segment_ids, waypoints
    if frames.size == 0:
        planned = np.zeros((total, HAND_STATE_DIM), dtype=np.float32)
        segment_ids = np.full((total,), -1, dtype=np.int32)
        velocities = np.zeros_like(planned)
        if return_dense:
            dense_velocities = np.zeros_like(dense)
            return planned, velocities, segment_ids, waypoints, dense, dense_velocities, dense_segment_ids
        return planned, velocities, segment_ids, waypoints

    first_frame = int(frames[0])
    first_dense = first_frame * substeps
    dense[: first_dense + 1] = waypoints[0]
    dense_segment_ids[: first_dense + 1] = 0

    for segment_index in range(frames.size - 1):
        start = int(frames[segment_index]) * substeps
        end = int(frames[segment_index + 1]) * substeps
        q0 = waypoints[segment_index]
        q1 = waypoints[segment_index + 1]
        _interpolate_segment_dense(
            dense,
            dense_segment_ids,
            segment_index=segment_index,
            start=start,
            end=end,
            q0=q0,
            q1=q1,
        )

    last_frame = int(frames[-1])
    last_dense = last_frame * substeps
    dense[last_dense:] = waypoints[-1]
    dense_segment_ids[last_dense:] = int(max(frames.size - 1, 0))

    dense_frames = frames * substeps
    _apply_press_windows(
        dense,
        waypoint_frames_dense=dense_frames,
        waypoint_target_keys=keys,
        waypoint_hand_joints=waypoints,
        config=cfg,
    )
    if cfg.enable_key_magnetism:
        dense = _apply_magnetic_windows(
            dense,
            waypoint_frames_dense=dense_frames,
            waypoint_target_keys=keys,
            waypoint_hand_joints=waypoints,
            config=cfg,
            key_geometry=key_geometry,
            kinematics=kinematics,
        )
    planned = dense[::substeps][:total].astype(np.float32, copy=True)
    segment_ids = dense_segment_ids[::substeps][:total].astype(np.int32, copy=True)
    velocities = compute_hand_velocities(planned, control_timestep=cfg.control_timestep)
    if return_dense:
        dense_velocities = compute_hand_velocities(dense, control_timestep=_dense_control_timestep(cfg))
        return planned, velocities, segment_ids, waypoints, dense, dense_velocities, dense_segment_ids
    return planned, velocities, segment_ids, waypoints


def compute_hand_velocities(hand_joints: np.ndarray, *, control_timestep: float) -> np.ndarray:
    values = np.asarray(hand_joints, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"hand_joints must be 2D [T, 46], got {values.shape}")
    if values.shape[0] <= 1:
        return np.zeros_like(values, dtype=np.float32)
    dt = max(float(control_timestep), 1e-8)
    return np.gradient(values, dt, axis=0).astype(np.float32)


def _interpolate_segment_dense(
    planned: np.ndarray,
    segment_ids: np.ndarray,
    *,
    segment_index: int,
    start: int,
    end: int,
    q0: np.ndarray,
    q1: np.ndarray,
) -> None:
    span = int(end - start)
    if span <= 0:
        planned[start] = q1
        segment_ids[start] = int(segment_index)
        return
    for offset in range(span + 1):
        step = start + offset
        u = float(offset) / float(span)
        alpha = _smoothstep(u)
        frame = (q0 + alpha * (q1 - q0)).astype(np.float32)
        if offset == 0:
            frame = q0.astype(np.float32, copy=True)
        elif offset == span:
            frame = q1.astype(np.float32, copy=True)
        planned[step] = frame
        segment_ids[step] = int(segment_index)


def _apply_press_windows(
    dense: np.ndarray,
    *,
    waypoint_frames_dense: np.ndarray,
    waypoint_target_keys: np.ndarray,
    waypoint_hand_joints: np.ndarray,
    config: PlannerConfig,
) -> None:
    approach, hold, release = _press_window_frame_counts(config)
    if dense.shape[0] == 0 or waypoint_frames_dense.size == 0:
        return
    active_windows: dict[int, list[tuple[int, float]]] = {RIGHT_FOREARM_TY_INDEX: [], LEFT_FOREARM_TY_INDEX: []}
    for waypoint_index, center_value in enumerate(np.asarray(waypoint_frames_dense, dtype=np.int64).reshape(-1)):
        center = int(center_value)
        hands = keyset_hand_sides(waypoint_target_keys[waypoint_index], threshold=config.threshold)
        for side, vertical_index in (("right", RIGHT_FOREARM_TY_INDEX), ("left", LEFT_FOREARM_TY_INDEX)):
            if not hands.get(side, False):
                continue
            waypoint_pressed = float(waypoint_hand_joints[waypoint_index, vertical_index])
            pressed = float(np.clip(waypoint_pressed - float(config.press_depth), float(config.vertical_min), float(config.vertical_max)))
            lifted = min(pressed + float(config.clearance_height), float(config.vertical_max))
            active_windows[vertical_index].append((center, lifted))
            start = max(0, center - approach)
            end = min(dense.shape[0] - 1, center + hold + release)
            for step in range(start, end + 1):
                offset = step - center
                if offset < 0:
                    weight = float(frame_window_envelope(offset, approach, 0, 0, config.press_envelope_power))
                    dense[step, vertical_index] = float(dense[step, vertical_index]) * (1.0 - weight) + pressed * weight
                elif offset <= hold:
                    dense[step, vertical_index] = pressed
                else:
                    release_u = min(max((offset - hold) / float(max(release, 1)), 0.0), 1.0)
                    weight = float(np.power(release_u, max(float(config.press_envelope_power), 1e-8)))
                    dense[step, vertical_index] = pressed * (1.0 - weight) + lifted * weight
                dense[step, vertical_index] = float(
                    np.clip(dense[step, vertical_index], float(config.vertical_min), float(config.vertical_max))
                )
    for vertical_index, windows in active_windows.items():
        for (center, lifted), (next_center, _next_lifted) in zip(windows, windows[1:]):
            plateau_start = center + hold + release
            plateau_end = next_center - approach
            if plateau_start < plateau_end:
                start = max(0, plateau_start)
                end = min(dense.shape[0], plateau_end)
                dense[start:end, vertical_index] = float(
                    np.clip(lifted, float(config.vertical_min), float(config.vertical_max))
                )
    for waypoint_index, center_value in enumerate(np.asarray(waypoint_frames_dense, dtype=np.int64).reshape(-1)):
        center = int(center_value)
        dense[center] = waypoint_hand_joints[waypoint_index]
        hands = keyset_hand_sides(waypoint_target_keys[waypoint_index], threshold=config.threshold)
        for side, vertical_index in (("right", RIGHT_FOREARM_TY_INDEX), ("left", LEFT_FOREARM_TY_INDEX)):
            if hands.get(side, False):
                dense[center, vertical_index] = float(
                    np.clip(
                        float(waypoint_hand_joints[waypoint_index, vertical_index]) - float(config.press_depth),
                        float(config.vertical_min),
                        float(config.vertical_max),
                    )
                )


def _apply_magnetic_windows(
    planned: np.ndarray,
    *,
    waypoint_frames_dense: np.ndarray,
    waypoint_target_keys: np.ndarray,
    waypoint_hand_joints: np.ndarray,
    config: PlannerConfig,
    key_geometry: np.ndarray | None,
    kinematics: HandKinematics | None,
) -> np.ndarray:
    key_xy = approximate_key_geometry() if key_geometry is None else np.asarray(key_geometry, dtype=np.float32)
    if key_xy.shape != (NUM_PIANO_KEYS, 2):
        raise ValueError(f"key_geometry must have shape [88, 2], got {key_xy.shape}")
    kin = kinematics or FakeHandKinematics()
    corrected = planned.astype(np.float32, copy=True)
    approach, hold, _release = _press_window_frame_counts(config)
    for waypoint_index, center_value in enumerate(np.asarray(waypoint_frames_dense, dtype=np.int64).reshape(-1)):
        center = int(center_value)
        target_keyset = waypoint_target_keys[waypoint_index]
        assignments = assign_active_fingertips(
            target_keyset,
            endpoint_hand_state=waypoint_hand_joints[waypoint_index],
            key_xy=key_xy,
            kinematics=kin,
            threshold=config.threshold,
        )
        if not assignments:
            continue
        start = max(0, center - approach)
        end = min(corrected.shape[0] - 1, center + hold)
        for step in range(start, end + 1):
            if config.preserve_waypoint_endpoints and step == center:
                continue
            corrected[step] = _magnetic_correct_frame(
                corrected[step],
                assignments=assignments,
                frame_offset=float(step - center),
                approach_frames=approach,
                hold_frames=hold,
                config=config,
                kinematics=kin,
            )
    if config.preserve_waypoint_endpoints:
        for index, frame in enumerate(waypoint_frames_dense):
            corrected[int(frame)] = waypoint_hand_joints[index]
    return corrected.astype(np.float32)


def _magnetic_correct_frame(
    frame: np.ndarray,
    *,
    assignments: list[FingertipAssignment],
    frame_offset: float,
    approach_frames: int,
    hold_frames: int,
    config: PlannerConfig,
    kinematics: HandKinematics,
) -> np.ndarray:
    q = np.asarray(frame, dtype=np.float32).reshape(-1)[:HAND_STATE_DIM].copy()
    kinematics.set_hand_state(q)
    xy = np.asarray(kinematics.fingertip_xy(), dtype=np.float32)[:10, :2]
    fingertip_indices: list[int] = []
    targets: list[np.ndarray] = []
    weights: list[float] = []
    for item in assignments:
        finger = int(item.fingertip_index)
        vector = np.asarray(item.key_xy, dtype=np.float32) - xy[finger]
        distance = float(np.linalg.norm(vector))
        weight = combined_magnetic_weight(
            distance,
            frame_offset,
            config,
            approach_frames=approach_frames,
            hold_frames=hold_frames,
            release_frames=0,
        )
        if weight <= 0.0:
            continue
        xy_step = clip_norm(vector * weight, float(config.magnet_max_xy_step))
        fingertip_indices.append(finger)
        targets.append(xy[finger] + xy_step)
        weights.append(1.0)
    if not fingertip_indices:
        return q.astype(np.float32)
    return kinematics.solve_xy_correction(
        q,
        np.asarray(fingertip_indices, dtype=np.int64),
        np.stack(targets, axis=0).astype(np.float32),
        np.asarray(weights, dtype=np.float32),
        damping=float(config.ik_damping),
        max_delta_q=float(config.ik_max_delta_q),
        iterations=int(config.ik_iterations_per_frame),
    ).astype(np.float32)


def _apply_vertical_clearance(
    frame: np.ndarray,
    *,
    q0: np.ndarray,
    q1: np.ndarray,
    u: float,
    active_hands: dict[str, bool],
    config: PlannerConfig,
) -> None:
    envelope = _clearance_envelope(u, config)
    if envelope <= 0.0:
        return
    if active_hands.get("right", False):
        _lift_vertical_index(frame, q0=q0, q1=q1, index=RIGHT_FOREARM_TY_INDEX, envelope=envelope, config=config)
    if active_hands.get("left", False):
        _lift_vertical_index(frame, q0=q0, q1=q1, index=LEFT_FOREARM_TY_INDEX, envelope=envelope, config=config)


def _lift_vertical_index(
    frame: np.ndarray,
    *,
    q0: np.ndarray,
    q1: np.ndarray,
    index: int,
    envelope: float,
    config: PlannerConfig,
) -> None:
    lift_level = min(max(float(q0[index]), float(q1[index])) + float(config.clearance_height), float(config.vertical_max))
    frame[index] = float(frame[index]) + float(envelope) * (lift_level - float(frame[index]))
    frame[index] = float(np.clip(frame[index], float(config.vertical_min), float(config.vertical_max)))


def _clearance_envelope(u: float, config: PlannerConfig) -> float:
    if u <= 0.0 or u >= 1.0:
        return 0.0
    ascent = float(np.clip(config.lift_fraction, 1e-6, 0.95))
    descent = float(np.clip(config.descent_fraction, 1e-6, 0.95))
    if ascent + descent >= 0.98:
        return float(np.sin(np.pi * np.clip(u, 0.0, 1.0)))
    if u < ascent:
        return _smoothstep(u / ascent)
    if u > 1.0 - descent:
        return 1.0 - _smoothstep((u - (1.0 - descent)) / descent)
    return 1.0


def _press_window_frame_counts(config: PlannerConfig) -> tuple[int, int, int]:
    dense_dt = _dense_control_timestep(config)
    approach = int(round(float(config.press_approach_s) / dense_dt))
    hold = int(round(float(config.press_hold_s) / dense_dt))
    release = int(round(float(config.press_release_s) / dense_dt))
    return max(approach, 0), max(hold, 0), max(release, 0)


def _interpolation_substeps(config: PlannerConfig) -> int:
    return max(int(config.interpolation_substeps), 1)


def _dense_control_timestep(config: PlannerConfig) -> float:
    return float(config.control_timestep) / float(_interpolation_substeps(config))


def _smoothstep(x: float) -> float:
    y = float(np.clip(x, 0.0, 1.0))
    return y * y * (3.0 - 2.0 * y)


def _sanitize_hand_states(hand_states: np.ndarray, config: PlannerConfig) -> np.ndarray:
    values = np.asarray(hand_states, dtype=np.float32)
    if values.size == 0:
        return np.zeros((0, HAND_STATE_DIM), dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != HAND_STATE_DIM:
        raise ValueError(f"waypoint_hand_joints must have shape [N, 46], got {values.shape}")
    out = np.ascontiguousarray(values, dtype=np.float32).copy()
    for index in FOREARM_TY_INDICES:
        out[:, index] = np.clip(out[:, index], float(config.vertical_min), float(config.vertical_max))
    return out


def _validate_keyset(keyset: np.ndarray, *, name: str) -> np.ndarray:
    row = np.asarray(keyset, dtype=np.float32).reshape(-1)
    if row.size < NUM_PIANO_KEYS:
        raise ValueError(f"{name} must contain at least 88 values, got {row.size}")
    return np.ascontiguousarray(row[:NUM_PIANO_KEYS], dtype=np.float32)


def _validate_waypoints(total_steps: int, frames: np.ndarray, keys: np.ndarray, waypoints: np.ndarray) -> None:
    if frames.ndim != 1:
        raise ValueError(f"waypoint_frames must be 1D, got {frames.shape}")
    if frames.size and (np.any(frames < 0) or np.any(frames >= int(total_steps))):
        raise ValueError(f"waypoint_frames must fall inside [0, {total_steps}), got {frames}")
    if frames.size and not np.all(frames[:-1] < frames[1:]):
        raise ValueError(f"waypoint_frames must be strictly increasing, got {frames}")
    if keys.shape != (frames.size, NUM_PIANO_KEYS):
        raise ValueError(f"waypoint_target_keys must have shape [{frames.size}, 88], got {keys.shape}")
    if waypoints.shape != (frames.size, HAND_STATE_DIM):
        raise ValueError(f"waypoint_hand_joints must have shape [{frames.size}, 46], got {waypoints.shape}")


def _validate_config(config: PlannerConfig) -> None:
    if float(config.control_timestep) <= 0.0:
        raise ValueError("control_timestep must be positive")
    if int(config.interpolation_substeps) < 1:
        raise ValueError("interpolation_substeps must be at least 1")
    if float(config.press_approach_s) < 0.0:
        raise ValueError("press_approach_s must be non-negative")
    if float(config.press_hold_s) < 0.0:
        raise ValueError("press_hold_s must be non-negative")
    if float(config.press_release_s) < 0.0:
        raise ValueError("press_release_s must be non-negative")
    if float(config.press_envelope_power) <= 0.0:
        raise ValueError("press_envelope_power must be positive")
    if float(config.press_depth) < 0.0:
        raise ValueError("press_depth must be non-negative")
    if float(config.vertical_min) > float(config.vertical_max):
        raise ValueError("vertical_min cannot exceed vertical_max")
    if float(config.clearance_height) < 0.0:
        raise ValueError("clearance_height must be non-negative")
    if float(config.magnet_radius) < 0.0:
        raise ValueError("magnet_radius must be non-negative")
    if float(config.magnet_sigma) <= 0.0:
        raise ValueError("magnet_sigma must be positive")
    if float(config.magnet_max_xy_step) < 0.0:
        raise ValueError("magnet_max_xy_step must be non-negative")
    if int(config.ik_iterations_per_frame) < 1:
        raise ValueError("ik_iterations_per_frame must be at least 1")


def _predict_waypoints(predictor: WaypointPredictor, target_keys: np.ndarray, *, batch_size: int) -> np.ndarray:
    try:
        parameters = inspect.signature(predictor).parameters
    except (TypeError, ValueError):
        return predictor(target_keys)
    if "batch_size" in parameters:
        return predictor(target_keys, batch_size=int(batch_size))
    return predictor(target_keys)
