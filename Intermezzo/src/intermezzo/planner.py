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
from intermezzo.keys import active_hands_for_transition, extract_waypoint_frames, validate_target_keys


WaypointPredictor = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class PlannerConfig:
    control_timestep: float = 0.05
    threshold: float = 0.5
    clearance_height: float = 0.02
    lift_fraction: float = 0.20
    descent_fraction: float = 0.35
    vertical_min: float = FOREARM_TY_MIN
    vertical_max: float = FOREARM_TY_MAX


@dataclass(frozen=True)
class PlannedTrajectory:
    target_keys: np.ndarray
    waypoint_frames: np.ndarray
    waypoint_target_keys: np.ndarray
    waypoint_hand_joints: np.ndarray
    planned_hand_joints: np.ndarray
    planned_hand_velocities: np.ndarray
    segment_ids: np.ndarray
    metadata: dict[str, object]


def build_intermezzo_trajectory(
    target_keys: np.ndarray,
    *,
    predictor: WaypointPredictor,
    config: PlannerConfig | None = None,
    batch_size: int = 256,
) -> PlannedTrajectory:
    cfg = config or PlannerConfig()
    keys = validate_target_keys(target_keys)
    waypoint_frames = extract_waypoint_frames(keys, threshold=cfg.threshold)
    waypoint_target_keys = keys[waypoint_frames] if waypoint_frames.size else np.zeros((0, NUM_PIANO_KEYS), dtype=np.float32)
    if waypoint_frames.size:
        predicted = np.asarray(_predict_waypoints(predictor, waypoint_target_keys, batch_size=batch_size), dtype=np.float32)
    else:
        predicted = np.zeros((0, HAND_STATE_DIM), dtype=np.float32)

    planned, velocities, segment_ids, sanitized_waypoints = plan_between_waypoints(
        total_steps=int(keys.shape[0]),
        waypoint_frames=waypoint_frames,
        waypoint_target_keys=waypoint_target_keys,
        waypoint_hand_joints=predicted,
        config=cfg,
    )
    metadata: dict[str, object] = {
        "planner": "intermezzo_lift_then_land",
        "planner_config": asdict(cfg),
        "batch_size": int(batch_size),
        "target_keys_shape": list(keys.shape),
        "num_waypoints": int(waypoint_frames.size),
        "waypoint_frames": waypoint_frames.astype(int).tolist(),
        "planned_hand_joints_shape": list(planned.shape),
        "planned_hand_velocities_shape": list(velocities.shape),
    }
    return PlannedTrajectory(
        target_keys=keys,
        waypoint_frames=waypoint_frames,
        waypoint_target_keys=waypoint_target_keys,
        waypoint_hand_joints=sanitized_waypoints,
        planned_hand_joints=planned,
        planned_hand_velocities=velocities,
        segment_ids=segment_ids,
        metadata=metadata,
    )


def plan_between_waypoints(
    *,
    total_steps: int,
    waypoint_frames: np.ndarray,
    waypoint_target_keys: np.ndarray,
    waypoint_hand_joints: np.ndarray,
    config: PlannerConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cfg = config or PlannerConfig()
    _validate_config(cfg)
    total = int(total_steps)
    if total < 0:
        raise ValueError(f"total_steps must be non-negative, got {total_steps}")

    frames = np.asarray(waypoint_frames, dtype=np.int64).reshape(-1)
    keys = np.asarray(waypoint_target_keys, dtype=np.float32)
    waypoints = _sanitize_hand_states(waypoint_hand_joints, cfg)
    _validate_waypoints(total, frames, keys, waypoints)

    planned = np.zeros((total, HAND_STATE_DIM), dtype=np.float32)
    segment_ids = np.full((total,), -1, dtype=np.int32)
    if total == 0:
        velocities = np.zeros_like(planned)
        return planned, velocities, segment_ids, waypoints
    if frames.size == 0:
        velocities = np.zeros_like(planned)
        return planned, velocities, segment_ids, waypoints

    first_frame = int(frames[0])
    planned[: first_frame + 1] = waypoints[0]
    segment_ids[: first_frame + 1] = 0

    for segment_index in range(frames.size - 1):
        start = int(frames[segment_index])
        end = int(frames[segment_index + 1])
        q0 = waypoints[segment_index]
        q1 = waypoints[segment_index + 1]
        current_keys = keys[segment_index]
        next_keys = keys[segment_index + 1]
        hands = active_hands_for_transition(current_keys, next_keys, threshold=cfg.threshold)
        _fill_segment(
            planned,
            segment_ids,
            segment_index=segment_index,
            start=start,
            end=end,
            q0=q0,
            q1=q1,
            active_hands=hands,
            config=cfg,
        )

    last_frame = int(frames[-1])
    planned[last_frame:] = waypoints[-1]
    segment_ids[last_frame:] = int(max(frames.size - 1, 0))
    velocities = compute_hand_velocities(planned, control_timestep=cfg.control_timestep)
    return planned, velocities, segment_ids, waypoints


def compute_hand_velocities(hand_joints: np.ndarray, *, control_timestep: float) -> np.ndarray:
    values = np.asarray(hand_joints, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"hand_joints must be 2D [T, 46], got {values.shape}")
    if values.shape[0] <= 1:
        return np.zeros_like(values, dtype=np.float32)
    dt = max(float(control_timestep), 1e-8)
    return np.gradient(values, dt, axis=0).astype(np.float32)


def _fill_segment(
    planned: np.ndarray,
    segment_ids: np.ndarray,
    *,
    segment_index: int,
    start: int,
    end: int,
    q0: np.ndarray,
    q1: np.ndarray,
    active_hands: dict[str, bool],
    config: PlannerConfig,
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
        else:
            _apply_vertical_clearance(frame, q0=q0, q1=q1, u=u, active_hands=active_hands, config=config)
        planned[step] = frame
        segment_ids[step] = int(segment_index)


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
    if float(config.vertical_min) > float(config.vertical_max):
        raise ValueError("vertical_min cannot exceed vertical_max")
    if float(config.clearance_height) < 0.0:
        raise ValueError("clearance_height must be non-negative")


def _predict_waypoints(predictor: WaypointPredictor, target_keys: np.ndarray, *, batch_size: int) -> np.ndarray:
    try:
        parameters = inspect.signature(predictor).parameters
    except (TypeError, ValueError):
        return predictor(target_keys)
    if "batch_size" in parameters:
        return predictor(target_keys, batch_size=int(batch_size))
    return predictor(target_keys)
