from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KeyEvent:
    key_id: int
    onset_frame: int
    offset_frame: int
    duration_frames: int


@dataclass(frozen=True)
class MatchedEvent:
    predicted_event: KeyEvent
    target_event: KeyEvent
    onset_error_frames: int
    offset_error_frames: int


def extract_key_events(
    activations: np.ndarray,
    *,
    activation_threshold: float = 0.5,
    min_press_duration_frames: int = 1,
    merge_gap_frames: int = 0,
) -> list[KeyEvent]:
    values = np.asarray(activations, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected [T, K] activations, got shape {values.shape}.")
    if min_press_duration_frames < 1:
        raise ValueError("min_press_duration_frames must be at least 1.")
    if merge_gap_frames < 0:
        raise ValueError("merge_gap_frames must be non-negative.")

    binary = values >= activation_threshold
    num_frames, num_keys = binary.shape
    events: list[KeyEvent] = []

    for key_id in range(num_keys):
        key_active = _merge_short_gaps(binary[:, key_id], merge_gap_frames)
        onset: int | None = None
        for frame_idx in range(num_frames):
            if key_active[frame_idx]:
                if onset is None:
                    onset = frame_idx
                continue
            if onset is None:
                continue
            duration = frame_idx - onset
            if duration >= min_press_duration_frames:
                events.append(
                    KeyEvent(
                        key_id=key_id,
                        onset_frame=onset,
                        offset_frame=frame_idx,
                        duration_frames=duration,
                    )
                )
            onset = None
        if onset is not None:
            duration = num_frames - onset
            if duration >= min_press_duration_frames:
                events.append(
                    KeyEvent(
                        key_id=key_id,
                        onset_frame=onset,
                        offset_frame=num_frames,
                        duration_frames=duration,
                    )
                )

    return sorted(events, key=lambda event: (event.onset_frame, event.key_id, event.offset_frame))


def match_events(
    predicted_events: list[KeyEvent],
    target_events: list[KeyEvent],
    *,
    onset_tolerance_frames: int,
    offset_tolerance_frames: int | None = None,
    enforce_offset_tolerance: bool = False,
) -> tuple[list[MatchedEvent], list[KeyEvent], list[KeyEvent]]:
    if onset_tolerance_frames < 0:
        raise ValueError("onset_tolerance_frames must be non-negative.")
    if offset_tolerance_frames is not None and offset_tolerance_frames < 0:
        raise ValueError("offset_tolerance_frames must be non-negative when provided.")

    candidates: list[tuple[int, int, int, int]] = []
    for pred_idx, predicted in enumerate(predicted_events):
        for target_idx, target in enumerate(target_events):
            if predicted.key_id != target.key_id:
                continue
            onset_error = predicted.onset_frame - target.onset_frame
            onset_abs_error = abs(onset_error)
            if onset_abs_error > onset_tolerance_frames:
                continue
            offset_abs_error = abs(predicted.offset_frame - target.offset_frame)
            if enforce_offset_tolerance and offset_tolerance_frames is not None:
                if offset_abs_error > offset_tolerance_frames:
                    continue
            candidates.append((onset_abs_error, offset_abs_error, pred_idx, target_idx))

    candidates.sort(key=lambda item: (item[0], item[1], item[3], item[2]))

    matched_predictions: set[int] = set()
    matched_targets: set[int] = set()
    matches: list[MatchedEvent] = []

    for _, _, pred_idx, target_idx in candidates:
        if pred_idx in matched_predictions or target_idx in matched_targets:
            continue
        predicted = predicted_events[pred_idx]
        target = target_events[target_idx]
        matches.append(
            MatchedEvent(
                predicted_event=predicted,
                target_event=target,
                onset_error_frames=predicted.onset_frame - target.onset_frame,
                offset_error_frames=predicted.offset_frame - target.offset_frame,
            )
        )
        matched_predictions.add(pred_idx)
        matched_targets.add(target_idx)

    unmatched_predictions = [
        predicted for index, predicted in enumerate(predicted_events) if index not in matched_predictions
    ]
    unmatched_targets = [target for index, target in enumerate(target_events) if index not in matched_targets]
    matches.sort(
        key=lambda match: (
            match.target_event.onset_frame,
            match.target_event.key_id,
            match.predicted_event.onset_frame,
        )
    )
    return matches, unmatched_predictions, unmatched_targets


def _merge_short_gaps(active: np.ndarray, merge_gap_frames: int) -> np.ndarray:
    merged = np.asarray(active, dtype=bool).copy()
    if merge_gap_frames == 0 or merged.size == 0:
        return merged

    frame_idx = 0
    num_frames = merged.shape[0]
    while frame_idx < num_frames:
        if merged[frame_idx]:
            frame_idx += 1
            continue
        gap_start = frame_idx
        while frame_idx < num_frames and not merged[frame_idx]:
            frame_idx += 1
        gap_end = frame_idx
        has_left_press = gap_start > 0 and merged[gap_start - 1]
        has_right_press = gap_end < num_frames and merged[gap_end]
        if has_left_press and has_right_press and (gap_end - gap_start) <= merge_gap_frames:
            merged[gap_start:gap_end] = True
    return merged
