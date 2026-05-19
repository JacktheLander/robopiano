from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from impromptu.config import ImpromptuConfig


NUM_FINGERS = 10


@dataclass(frozen=True)
class FingertipTrajectory:
    targets: np.ndarray
    weights: np.ndarray
    dense_frames: np.ndarray


def _frame_count(seconds: float, dense_dt: float) -> int:
    if seconds <= 0.0:
        return 0
    return max(int(np.ceil(float(seconds) / max(float(dense_dt), 1e-8))), 1)


def _smoothstep(value: float) -> float:
    u = float(np.clip(value, 0.0, 1.0))
    return u * u * (3.0 - 2.0 * u)


def _lift(target: np.ndarray, clearance: float) -> np.ndarray:
    out = np.asarray(target, dtype=np.float32).copy()
    out[2] += float(clearance)
    return out


def _write_frame(
    targets: np.ndarray,
    weights: np.ndarray,
    frame: int,
    finger: int,
    target: np.ndarray,
    weight: float,
    *,
    overwrite: bool = True,
) -> None:
    if frame < 0 or frame >= targets.shape[0]:
        return
    if overwrite or float(weight) >= float(weights[frame, finger]):
        targets[frame, finger] = np.asarray(target, dtype=np.float32)
        weights[frame, finger] = np.float32(weight)


def _write_interp(
    targets: np.ndarray,
    weights: np.ndarray,
    *,
    finger: int,
    start: int,
    end: int,
    p0: np.ndarray,
    p1: np.ndarray,
    w0: float,
    w1: float,
    min_z: float | None = None,
    overwrite: bool = True,
) -> None:
    if targets.shape[0] == 0:
        return
    lo = max(int(start), 0)
    hi = min(int(end), targets.shape[0] - 1)
    if hi < lo:
        return
    span = max(hi - lo, 1)
    for frame in range(lo, hi + 1):
        u = 0.0 if hi == lo else (frame - lo) / span
        alpha = _smoothstep(u)
        target = np.asarray(p0 + alpha * (p1 - p0), dtype=np.float32)
        if min_z is not None:
            target[2] = max(float(target[2]), float(min_z))
        weight = (1.0 - alpha) * float(w0) + alpha * float(w1)
        _write_frame(targets, weights, frame, finger, target, weight, overwrite=overwrite)


def _assigned_events(
    assignments: np.ndarray,
    fingertip_targets: np.ndarray,
    waypoint_frames_dense: np.ndarray,
    finger: int,
) -> list[tuple[int, int, np.ndarray]]:
    events: list[tuple[int, int, np.ndarray]] = []
    for index, frame in enumerate(waypoint_frames_dense.astype(np.int64)):
        key = int(assignments[index, finger])
        target = fingertip_targets[index, finger]
        if key >= 0 and np.isfinite(target).all():
            events.append((int(frame), key, np.asarray(target, dtype=np.float32)))
    return events


def build_fingertip_trajectory(
    *,
    total_steps: int,
    waypoint_frames: np.ndarray,
    assignments: np.ndarray,
    fingertip_targets: np.ndarray,
    waypoint_fingertips: np.ndarray,
    config: ImpromptuConfig,
) -> FingertipTrajectory:
    total = max(int(total_steps), 0)
    substeps = max(int(config.interpolation_substeps), 1)
    dense_total = total * substeps
    dense_frames = np.arange(dense_total, dtype=np.int64)
    targets = np.full((dense_total, NUM_FINGERS, 3), np.nan, dtype=np.float32)
    weights = np.zeros((dense_total, NUM_FINGERS), dtype=np.float32)
    if dense_total == 0:
        return FingertipTrajectory(targets=targets, weights=weights, dense_frames=dense_frames)

    frames = np.asarray(waypoint_frames, dtype=np.int64).reshape(-1)
    dense_waypoints = np.clip(frames * substeps, 0, max(dense_total - 1, 0)).astype(np.int64)
    assigns = np.asarray(assignments, dtype=np.int32)
    ft_targets = np.asarray(fingertip_targets, dtype=np.float32)
    wp_tips = np.asarray(waypoint_fingertips, dtype=np.float32)
    if assigns.shape != (frames.size, NUM_FINGERS):
        raise ValueError(f"assignments must have shape [N, 10], got {assigns.shape}")
    if ft_targets.shape != (frames.size, NUM_FINGERS, 3):
        raise ValueError(f"fingertip_targets must have shape [N, 10, 3], got {ft_targets.shape}")
    if wp_tips.shape != (frames.size, NUM_FINGERS, 3):
        raise ValueError(f"waypoint_fingertips must have shape [N, 10, 3], got {wp_tips.shape}")

    dense_dt = float(config.control_timestep) / float(substeps)
    approach = _frame_count(float(config.approach_s), dense_dt)
    hold = _frame_count(float(config.hold_s), dense_dt)
    release = _frame_count(float(config.release_s), dense_dt)
    clearance = float(config.clearance_height)

    for finger in range(NUM_FINGERS):
        events = _assigned_events(assigns, ft_targets, dense_waypoints, finger)
        if not events:
            continue
        previous_release_end: int | None = None
        previous_hover: np.ndarray | None = None
        previous_key: int | None = None
        previous_press_frame: int | None = None

        for event_index, (press_frame, key, press_target) in enumerate(events):
            next_key = events[event_index + 1][1] if event_index + 1 < len(events) else None
            hover_target = _lift(press_target, clearance)
            approach_start = press_frame - approach
            hold_end = min(press_frame + hold, dense_total - 1)
            same_as_previous = previous_key == key and previous_press_frame is not None

            if same_as_previous:
                _write_interp(
                    targets,
                    weights,
                    finger=finger,
                    start=previous_press_frame,
                    end=press_frame,
                    p0=press_target,
                    p1=press_target,
                    w0=1.0,
                    w1=1.0,
                    overwrite=True,
                )
            else:
                if previous_release_end is not None and previous_hover is not None:
                    travel_start = previous_release_end + 1
                    travel_end = approach_start - 1
                    min_z = max(float(previous_hover[2]), float(hover_target[2]))
                    _write_interp(
                        targets,
                        weights,
                        finger=finger,
                        start=travel_start,
                        end=travel_end,
                        p0=previous_hover,
                        p1=hover_target,
                        w0=0.2,
                        w1=0.35,
                        min_z=min_z,
                        overwrite=False,
                    )
                elif event_index == 0:
                    start_tip = wp_tips[0, finger]
                    p0 = _lift(press_target, clearance) if not np.isfinite(start_tip).all() else start_tip
                    if np.isfinite(p0).all():
                        p0 = np.asarray(p0, dtype=np.float32).copy()
                        p0[2] = max(float(p0[2]), float(hover_target[2]))
                    else:
                        p0 = hover_target
                    _write_interp(
                        targets,
                        weights,
                        finger=finger,
                        start=max(0, approach_start),
                        end=max(0, press_frame - 1),
                        p0=p0,
                        p1=hover_target,
                        w0=0.15,
                        w1=0.5,
                        min_z=float(hover_target[2]),
                        overwrite=False,
                    )

                _write_interp(
                    targets,
                    weights,
                    finger=finger,
                    start=approach_start,
                    end=press_frame,
                    p0=hover_target,
                    p1=press_target,
                    w0=0.5,
                    w1=1.0,
                    overwrite=True,
                )

            _write_interp(
                targets,
                weights,
                finger=finger,
                start=press_frame,
                end=hold_end,
                p0=press_target,
                p1=press_target,
                w0=1.0,
                w1=1.0,
                overwrite=True,
            )

            if next_key == key:
                previous_release_end = hold_end
                previous_hover = hover_target
            else:
                release_end = min(hold_end + release, dense_total - 1)
                _write_interp(
                    targets,
                    weights,
                    finger=finger,
                    start=hold_end,
                    end=release_end,
                    p0=press_target,
                    p1=hover_target,
                    w0=1.0,
                    w1=0.25,
                    overwrite=True,
                )
                previous_release_end = release_end
                previous_hover = hover_target
            previous_key = key
            previous_press_frame = press_frame

    return FingertipTrajectory(targets=targets, weights=weights, dense_frames=dense_frames)
