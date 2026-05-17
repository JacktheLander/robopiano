from __future__ import annotations

import numpy as np


def radial_weight(distance: float | np.ndarray, radius: float, sigma: float) -> float | np.ndarray:
    """Distance weight that is exactly zero outside ``radius``."""
    d = np.asarray(distance, dtype=np.float32)
    r = max(float(radius), 0.0)
    s = max(float(sigma), 1e-8)
    weight = np.exp(-d / s).astype(np.float32)
    weight = np.where((d >= 0.0) & (d <= r + 1e-7), weight, 0.0).astype(np.float32)
    if np.isscalar(distance):
        return float(weight)
    return weight


def landing_envelope(u: float | np.ndarray, start: float, power: float) -> float | np.ndarray:
    """Late-trajectory envelope in [0, 1]."""
    values = np.asarray(u, dtype=np.float32)
    start_value = float(np.clip(start, 0.0, 0.999999))
    exponent = max(float(power), 1e-8)
    normalized = np.clip((values - start_value) / max(1.0 - start_value, 1e-8), 0.0, 1.0)
    envelope = np.power(normalized, exponent).astype(np.float32)
    if np.isscalar(u):
        return float(envelope)
    return envelope


def frame_window_envelope(
    frame_offset: float | np.ndarray,
    approach: int,
    hold: int,
    release: int,
    power: float,
) -> float | np.ndarray:
    """Waypoint-centered press envelope in [0, 1]."""
    offsets = np.asarray(frame_offset, dtype=np.float32)
    approach_frames = max(int(approach), 0)
    hold_frames = max(int(hold), 0)
    release_frames = max(int(release), 0)
    exponent = max(float(power), 1e-8)
    envelope = np.zeros_like(offsets, dtype=np.float32)

    if approach_frames <= 0:
        envelope = np.where(offsets <= 0.0, 1.0, envelope).astype(np.float32)
    else:
        in_approach = (offsets >= -approach_frames) & (offsets <= 0.0)
        approach_u = np.clip((offsets + approach_frames) / float(approach_frames), 0.0, 1.0)
        envelope = np.where(in_approach, np.power(approach_u, exponent), envelope).astype(np.float32)

    in_hold = (offsets >= 0.0) & (offsets <= hold_frames)
    envelope = np.where(in_hold, 1.0, envelope).astype(np.float32)

    if release_frames > 0:
        release_start = float(hold_frames)
        in_release = (offsets > release_start) & (offsets <= release_start + release_frames)
        release_u = np.clip((offsets - release_start) / float(release_frames), 0.0, 1.0)
        envelope = np.where(in_release, 1.0 - np.power(release_u, exponent), envelope).astype(np.float32)

    if np.isscalar(frame_offset):
        return float(envelope)
    return envelope


def combined_magnetic_weight(
    distance: float,
    u: float,
    config: object,
    *,
    approach_frames: int | None = None,
    hold_frames: int = 0,
    release_frames: int = 0,
) -> float:
    """Combined radial and time-window magnetic gain."""
    distance_weight = radial_weight(distance, getattr(config, "magnet_radius"), getattr(config, "magnet_sigma"))
    if approach_frames is None:
        time_weight = landing_envelope(u, getattr(config, "magnet_start_fraction"), getattr(config, "magnet_power"))
    else:
        time_weight = frame_window_envelope(
            u,
            approach=int(approach_frames),
            hold=int(hold_frames),
            release=int(release_frames),
            power=getattr(config, "magnet_power"),
        )
    return float(getattr(config, "magnet_gain")) * float(distance_weight) * float(time_weight)


def clip_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
    """Clip a vector by Euclidean norm without changing its direction."""
    values = np.asarray(vector, dtype=np.float32)
    limit = float(max_norm)
    if limit < 0.0:
        raise ValueError("max_norm must be non-negative")
    norm = float(np.linalg.norm(values))
    if norm <= limit or norm <= 1e-12:
        return values.astype(np.float32, copy=True)
    return (values * (limit / norm)).astype(np.float32)
