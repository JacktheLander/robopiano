from __future__ import annotations

import math

import numpy as np

from etude.evaluation.event_matching import KeyEvent, extract_key_events, match_events
from etude.evaluation.event_metrics import EventMetricsConfig, compute_event_metrics
from etude.evaluation.metrics import note_metrics


def test_extract_key_events_merges_short_gaps_and_filters_short_presses() -> None:
    activations = np.zeros((8, 88), dtype=np.float32)
    activations[1, 10] = 1.0
    activations[3:5, 10] = 1.0
    activations[6, 11] = 1.0

    events = extract_key_events(
        activations,
        activation_threshold=0.5,
        min_press_duration_frames=2,
        merge_gap_frames=1,
    )

    assert events == [KeyEvent(key_id=10, onset_frame=1, offset_frame=5, duration_frames=4)]


def test_compute_event_metrics_exact_match() -> None:
    target = _roll(12, [(40, 2, 5), (43, 7, 10)])
    predicted = target.copy()

    metrics = compute_event_metrics(predicted, target, EventMetricsConfig(dt=0.01))

    assert metrics["piano/event_precision"] == 1.0
    assert metrics["piano/event_recall"] == 1.0
    assert metrics["piano/event_f1"] == 1.0
    assert metrics["piano/matched_events"] == 2.0
    assert metrics["piano/missed_events"] == 0.0
    assert metrics["piano/false_events"] == 0.0
    assert metrics["piano/timing_abs_error_mean_s"] == 0.0
    assert metrics["piano/early_press_rate"] == 0.0
    assert metrics["piano/late_press_rate"] == 0.0
    assert metrics["piano/sticky_release_rate"] == 0.0


def test_compute_event_metrics_missed_and_false_notes() -> None:
    target = _roll(12, [(40, 2, 5), (43, 7, 10)])
    predicted = _roll(12, [(40, 2, 5), (47, 7, 10)])

    metrics = compute_event_metrics(predicted, target, EventMetricsConfig(dt=0.01))

    assert metrics["piano/matched_events"] == 1.0
    assert metrics["piano/missed_events"] == 1.0
    assert metrics["piano/false_events"] == 1.0
    assert metrics["piano/event_precision"] == 0.5
    assert metrics["piano/event_recall"] == 0.5
    assert metrics["piano/event_f1"] == 0.5


def test_compute_event_metrics_early_and_late_notes() -> None:
    target = _roll(30, [(50, 10, 14), (51, 20, 24)])
    predicted = _roll(30, [(50, 9, 13), (51, 21, 25)])

    metrics = compute_event_metrics(
        predicted,
        target,
        EventMetricsConfig(dt=0.01, onset_tolerance_ms=15, min_press_duration_ms=10, merge_gap_ms=0),
    )

    assert metrics["piano/matched_events"] == 2.0
    assert math.isclose(metrics["piano/timing_abs_error_mean_s"], 0.01, rel_tol=0.0, abs_tol=1e-8)
    assert math.isclose(metrics["piano/timing_abs_error_p95_s"], 0.01, rel_tol=0.0, abs_tol=1e-8)
    assert metrics["piano/early_press_rate"] == 0.5
    assert metrics["piano/late_press_rate"] == 0.5


def test_match_events_repeated_same_key_uses_one_to_one_nearest_matches() -> None:
    target = extract_key_events(_roll(20, [(32, 4, 7), (32, 10, 13)]), min_press_duration_frames=1)
    predicted = extract_key_events(_roll(20, [(32, 5, 8), (32, 9, 12)]), min_press_duration_frames=1)

    matches, false_events, missed_events = match_events(
        predicted,
        target,
        onset_tolerance_frames=1,
        offset_tolerance_frames=2,
    )

    assert [(match.predicted_event.onset_frame, match.target_event.onset_frame) for match in matches] == [
        (5, 4),
        (9, 10),
    ]
    assert false_events == []
    assert missed_events == []


def test_compute_event_metrics_sticky_release_rate() -> None:
    target = _roll(16, [(60, 4, 7)])
    predicted = _roll(16, [(60, 4, 10)])

    metrics = compute_event_metrics(
        predicted,
        target,
        EventMetricsConfig(dt=0.01, offset_tolerance_ms=20, min_press_duration_ms=10, merge_gap_ms=0),
    )

    assert metrics["piano/matched_events"] == 1.0
    assert metrics["piano/sticky_release_rate"] == 1.0


def test_frame_metric_aliases_match_note_metrics() -> None:
    target = _roll(4, [(10, 1, 3)])
    predicted = _roll(4, [(10, 1, 2), (12, 0, 1)])

    metrics = note_metrics(predicted, target)

    assert metrics["piano/frame_precision"] == metrics["piano/note_precision"]
    assert metrics["piano/frame_recall"] == metrics["piano/note_recall"]
    assert metrics["piano/frame_f1"] == metrics["piano/note_f1"]


def _roll(num_frames: int, events: list[tuple[int, int, int]]) -> np.ndarray:
    roll = np.zeros((num_frames, 88), dtype=np.float32)
    for key_id, onset_frame, offset_frame in events:
        roll[onset_frame:offset_frame, key_id] = 1.0
    return roll
