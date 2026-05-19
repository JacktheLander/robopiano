from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from etude.evaluation.event_matching import extract_key_events, match_events


@dataclass(frozen=True)
class EventMetricsConfig:
    dt: float
    onset_tolerance_ms: float = 50.0
    offset_tolerance_ms: float | None = 80.0
    min_press_duration_ms: float = 20.0
    merge_gap_ms: float = 25.0
    key_activation_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("dt must be positive.")


def compute_event_metrics(
    predicted_keys: np.ndarray,
    target_keys: np.ndarray,
    config: EventMetricsConfig,
) -> dict[str, float]:
    predicted_events = extract_key_events(
        predicted_keys,
        activation_threshold=config.key_activation_threshold,
        min_press_duration_frames=_milliseconds_to_frames(config.min_press_duration_ms, config.dt, minimum=1),
        merge_gap_frames=_milliseconds_to_frames(config.merge_gap_ms, config.dt),
    )
    target_events = extract_key_events(
        target_keys,
        activation_threshold=config.key_activation_threshold,
        min_press_duration_frames=_milliseconds_to_frames(config.min_press_duration_ms, config.dt, minimum=1),
        merge_gap_frames=_milliseconds_to_frames(config.merge_gap_ms, config.dt),
    )

    offset_tolerance_frames = None
    if config.offset_tolerance_ms is not None:
        offset_tolerance_frames = _milliseconds_to_frames(config.offset_tolerance_ms, config.dt)

    matches, false_events, missed_events = match_events(
        predicted_events,
        target_events,
        onset_tolerance_frames=_milliseconds_to_frames(config.onset_tolerance_ms, config.dt),
        offset_tolerance_frames=offset_tolerance_frames,
    )

    matched_count = float(len(matches))
    false_count = float(len(false_events))
    missed_count = float(len(missed_events))
    precision = matched_count / max(matched_count + false_count, 1.0)
    recall = matched_count / max(matched_count + missed_count, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)

    onset_errors_s = np.asarray([abs(match.onset_error_frames) * config.dt for match in matches], dtype=np.float32)
    early_count = float(sum(match.onset_error_frames < 0 for match in matches))
    late_count = float(sum(match.onset_error_frames > 0 for match in matches))

    if offset_tolerance_frames is None:
        sticky_count = float(sum(match.offset_error_frames > 0 for match in matches))
    else:
        sticky_count = float(sum(match.offset_error_frames > offset_tolerance_frames for match in matches))

    return {
        "piano/event_precision": precision,
        "piano/event_recall": recall,
        "piano/event_f1": f1,
        "piano/matched_events": matched_count,
        "piano/missed_events": missed_count,
        "piano/false_events": false_count,
        "piano/timing_abs_error_mean_s": float(onset_errors_s.mean()) if onset_errors_s.size else 0.0,
        "piano/timing_abs_error_p95_s": float(np.percentile(onset_errors_s, 95)) if onset_errors_s.size else 0.0,
        "piano/early_press_rate": early_count / matched_count if matched_count else 0.0,
        "piano/late_press_rate": late_count / matched_count if matched_count else 0.0,
        "piano/sticky_release_rate": sticky_count / matched_count if matched_count else 0.0,
    }


def _milliseconds_to_frames(milliseconds: float, dt: float, *, minimum: int = 0) -> int:
    if milliseconds < 0.0:
        raise ValueError("milliseconds must be non-negative.")
    seconds = milliseconds / 1000.0
    frames = int(math.ceil(seconds / dt)) if seconds > 0.0 else 0
    return max(frames, minimum)
