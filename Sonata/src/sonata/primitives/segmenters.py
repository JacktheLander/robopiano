from __future__ import annotations

from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import numpy as np
import pandas as pd
from tqdm import tqdm

from sonata.data.loading import load_episode_record
from sonata.data.score import dumps_score_context, infer_events_from_goal_roll, load_note_events, score_context_from_roll
from sonata.data.schema import ScoreEvent, SegmentRecord
from sonata.primitives.slim_cache import (
    append_episode_progress,
    build_gmr_target,
    chunk_index_from_name,
    collect_slim_chunk_names,
    compact_store_manifest_path,
    compose_segment_index,
    ensure_segment_index_columns,
    is_slim_chunk_name,
    list_incomplete_slim_chunks,
    load_completed_episodes,
    read_compact_store_manifest,
    load_slim_index_table,
    next_slim_chunk_index,
    online_segment_processing_enabled,
    resolve_slim_cache_paths,
    resolve_online_storage_format,
    save_raw_segment_chunks_enabled,
    slim_chunk_complete,
    slim_chunk_name,
    summarize_slim_cache,
    write_compact_store_manifest,
    write_slim_chunk,
)
from sonata.primitives.guards import estimate_dir_bytes, runtime_guard_projected_walltime, storage_guard
from sonata.utils.io import read_table, save_npz, write_json, write_table


@dataclass
class CandidateSegment:
    onset_step: int
    end_step: int
    segment_source: str
    score_event_id: str
    key_signature: str
    heuristic_family: str
    coarse_family: str
    control_phase: str
    phase_index: int
    phase_count: int
    event_duration_steps: int
    chord_size: int
    key_center: float
    boundary_score_peak: float = 0.0
    boundary_source: str = ""
    snapped_to_score_event: int = -1
    raw_segment_length: int = 0
    segment_filter_reason: str = ""


@dataclass
class PreparedSegment:
    row: dict[str, Any]
    feature_vector: np.ndarray
    feature_names: list[str]
    gmr_target: np.ndarray
    gmr_target_name: str
    arrays: dict[str, np.ndarray | None]
    raw_bytes_estimate: int


@dataclass
class PreparedEpisodeBatch:
    song_id: str
    episode_id: str
    score_rows: list[dict[str, Any]]
    prepared_segments: list[PreparedSegment]


class BaseSegmenter:
    name = "base"

    def segment(self, episode, score_events: list[ScoreEvent]) -> list[CandidateSegment]:
        raise NotImplementedError


class FixedWindowSegmenter(BaseSegmenter):
    name = "fixed_window"

    def __init__(self, window_steps: int, stride_steps: int):
        self.window_steps = int(window_steps)
        self.stride_steps = int(stride_steps)

    def segment(self, episode, score_events: list[ScoreEvent]) -> list[CandidateSegment]:
        del score_events
        if episode.hand_joints is None:
            return []
        segments = []
        for start in range(0, max(episode.hand_joints.shape[0] - self.window_steps + 1, 1), self.stride_steps):
            end = min(start + self.window_steps, episode.hand_joints.shape[0])
            segments.append(
                CandidateSegment(
                    onset_step=int(start),
                    end_step=int(end),
                    segment_source=self.name,
                    score_event_id="",
                    key_signature="",
                    heuristic_family="window",
                    coarse_family="transition",
                    control_phase="whole_event",
                    phase_index=0,
                    phase_count=1,
                    event_duration_steps=max(int(end - start), 1),
                    chord_size=0,
                    key_center=0.0,
                )
            )
        return segments


class ChangePointSegmenter(BaseSegmenter):
    name = "changepoint"

    def __init__(self, window_steps: int, min_gap_steps: int, velocity_quantile: float, acceleration_quantile: float):
        self.window_steps = int(window_steps)
        self.min_gap_steps = int(min_gap_steps)
        self.velocity_quantile = float(velocity_quantile)
        self.acceleration_quantile = float(acceleration_quantile)

    def segment(self, episode, score_events: list[ScoreEvent]) -> list[CandidateSegment]:
        del score_events
        if episode.hand_joints is None:
            return []
        velocity = np.gradient(np.asarray(episode.hand_joints, dtype=np.float32), axis=0)
        acceleration = np.gradient(velocity, axis=0)
        speed = np.linalg.norm(velocity, axis=1)
        accel = np.linalg.norm(acceleration, axis=1)
        threshold_speed = float(np.quantile(speed, self.velocity_quantile))
        threshold_accel = float(np.quantile(accel, self.acceleration_quantile))
        change_points = np.flatnonzero((speed >= threshold_speed) | (accel >= threshold_accel))
        if change_points.size == 0:
            change_points = np.array([0], dtype=np.int64)
        filtered: list[int] = []
        for point in change_points.tolist():
            if not filtered or point - filtered[-1] >= self.min_gap_steps:
                filtered.append(point)
        segments: list[CandidateSegment] = []
        half = self.window_steps // 2
        for point in filtered:
            start = max(point - half, 0)
            end = min(start + self.window_steps, episode.hand_joints.shape[0])
            segments.append(
                CandidateSegment(
                    onset_step=int(start),
                    end_step=int(end),
                    segment_source=self.name,
                    score_event_id="",
                    key_signature="",
                    heuristic_family="changepoint",
                    coarse_family="transition",
                    control_phase="whole_event",
                    phase_index=0,
                    phase_count=1,
                    event_duration_steps=max(int(end - start), 1),
                    chord_size=0,
                    key_center=0.0,
                )
            )
        return segments


class NoteAlignedSegmenter(BaseSegmenter):
    name = "note_aligned"

    def __init__(self, pre_steps: int, post_steps: int, note_local_horizon_steps: int | None = None):
        self.pre_steps = int(pre_steps)
        self.post_steps = int(post_steps)
        self.note_local_horizon_steps = (
            int(note_local_horizon_steps) if note_local_horizon_steps is not None and int(note_local_horizon_steps) > 0 else None
        )

    def segment(self, episode, score_events: list[ScoreEvent]) -> list[CandidateSegment]:
        segments: list[CandidateSegment] = []
        for event in score_events:
            start = max(event.onset_step - self.pre_steps, 0)
            end = event.end_step + self.post_steps
            if self.note_local_horizon_steps is not None:
                end = min(end, event.onset_step + self.note_local_horizon_steps)
            end = max(start + 1, min(end, episode.hand_joints.shape[0]))
            family = classify_interval_family(event)
            segments.append(
                CandidateSegment(
                    onset_step=start,
                    end_step=end,
                    segment_source=self.name,
                    score_event_id=event.event_id,
                    key_signature="-".join(str(item) for item in event.key_numbers),
                    heuristic_family=family,
                    coarse_family=coarse_family_for_event(event),
                    control_phase="whole_event",
                    phase_index=0,
                    phase_count=1,
                    event_duration_steps=max(int(event.end_step - event.onset_step), 1),
                    chord_size=event.chord_size,
                    key_center=event.key_center,
                )
            )
        return segments


class EventPhaseSegmenter(BaseSegmenter):
    name = "event_phase_aligned"

    def __init__(
        self,
        pre_steps: int,
        post_steps: int,
        note_local_horizon_steps: int | None = None,
        *,
        onset_window_steps_single: int = 4,
        onset_window_steps_chord: int = 6,
        approach_window_steps: int = 6,
        release_window_steps: int = 6,
        hold_min_duration_steps: int = 8,
        hold_tail_steps: int = 2,
        transition_max_gap_steps: int = 8,
        transition_window_steps: int = 8,
        staccato_duration_steps: int = 6,
        min_phase_duration_steps: int = 2,
    ):
        self.pre_steps = int(pre_steps)
        self.post_steps = int(post_steps)
        self.note_local_horizon_steps = (
            int(note_local_horizon_steps) if note_local_horizon_steps is not None and int(note_local_horizon_steps) > 0 else None
        )
        self.onset_window_steps_single = int(onset_window_steps_single)
        self.onset_window_steps_chord = int(onset_window_steps_chord)
        self.approach_window_steps = int(approach_window_steps)
        self.release_window_steps = int(release_window_steps)
        self.hold_min_duration_steps = int(hold_min_duration_steps)
        self.hold_tail_steps = int(hold_tail_steps)
        self.transition_max_gap_steps = int(transition_max_gap_steps)
        self.transition_window_steps = int(transition_window_steps)
        self.staccato_duration_steps = int(staccato_duration_steps)
        self.min_phase_duration_steps = int(min_phase_duration_steps)

    def segment(self, episode, score_events: list[ScoreEvent]) -> list[CandidateSegment]:
        if episode.hand_joints is None:
            return []
        horizon = int(episode.hand_joints.shape[0])
        segments: list[CandidateSegment] = []
        for index, event in enumerate(score_events):
            next_event = score_events[index + 1] if index + 1 < len(score_events) else None
            event_duration = max(int(event.end_step - event.onset_step), 1)
            heuristic_family = classify_interval_family(event)
            key_signature = "-".join(str(item) for item in event.key_numbers)
            onset_window = (
                self.onset_window_steps_chord if int(event.chord_size) >= 2 else self.onset_window_steps_single
            )
            if event_duration <= self.staccato_duration_steps and int(event.chord_size) <= 1:
                onset_window = max(2, min(onset_window, self.staccato_duration_steps))
            approach_steps = min(self.approach_window_steps, max(self.pre_steps, 0))
            release_steps = min(self.release_window_steps, max(self.post_steps, 0))
            phase_segments: list[CandidateSegment] = []
            self._append_phase(
                phase_segments,
                onset_step=max(int(event.onset_step) - approach_steps, 0),
                end_step=int(event.onset_step),
                score_event=event,
                heuristic_family=heuristic_family,
                key_signature=key_signature,
                control_phase="approach",
                coarse_family="move",
                event_duration_steps=event_duration,
                horizon=horizon,
            )
            onset_end = min(int(event.onset_step) + onset_window, horizon)
            self._append_phase(
                phase_segments,
                onset_step=max(int(event.onset_step) - min(2, approach_steps // 2), 0),
                end_step=onset_end,
                score_event=event,
                heuristic_family=heuristic_family,
                key_signature=key_signature,
                control_phase="press_onset",
                coarse_family="chord_press" if int(event.chord_size) >= 2 else "single_press",
                event_duration_steps=event_duration,
                horizon=horizon,
            )
            if event_duration >= self.hold_min_duration_steps:
                hold_start = min(int(event.onset_step) + max(onset_window // 2, 1), max(horizon - 1, 0))
                hold_end = min(int(event.end_step) + self.hold_tail_steps, horizon)
                self._append_phase(
                    phase_segments,
                    onset_step=hold_start,
                    end_step=hold_end,
                    score_event=event,
                    heuristic_family=heuristic_family,
                    key_signature=key_signature,
                    control_phase="hold",
                    coarse_family="hold",
                    event_duration_steps=event_duration,
                    horizon=horizon,
                )
            release_start = max(min(int(event.end_step) - 1, horizon - 1), int(event.onset_step))
            self._append_phase(
                phase_segments,
                onset_step=release_start,
                end_step=min(int(event.end_step) + release_steps, horizon),
                score_event=event,
                heuristic_family=heuristic_family,
                key_signature=key_signature,
                control_phase="release",
                coarse_family="release",
                event_duration_steps=event_duration,
                horizon=horizon,
            )
            if next_event is not None:
                gap = int(next_event.onset_step) - int(event.end_step)
                if gap >= 0 and gap <= self.transition_max_gap_steps:
                    transition_end = min(int(next_event.onset_step) + min(self.transition_window_steps, onset_window), horizon)
                    self._append_phase(
                        phase_segments,
                        onset_step=max(int(event.end_step) - 1, int(event.onset_step)),
                        end_step=transition_end,
                        score_event=event,
                        heuristic_family=heuristic_family,
                        key_signature=key_signature,
                        control_phase="local_transition",
                        coarse_family="short_sequence" if gap <= max(self.transition_max_gap_steps // 2, 1) else "transition",
                        event_duration_steps=event_duration,
                        horizon=horizon,
                    )
            if not phase_segments:
                fallback_end = int(event.end_step) + self.post_steps
                if self.note_local_horizon_steps is not None:
                    fallback_end = min(fallback_end, int(event.onset_step) + self.note_local_horizon_steps)
                self._append_phase(
                    phase_segments,
                    onset_step=max(int(event.onset_step) - self.pre_steps, 0),
                    end_step=min(max(fallback_end, int(event.onset_step) + 1), horizon),
                    score_event=event,
                    heuristic_family=heuristic_family,
                    key_signature=key_signature,
                    control_phase="whole_event",
                    coarse_family=coarse_family_for_event(event),
                    event_duration_steps=event_duration,
                    horizon=horizon,
                )
            total = len(phase_segments)
            for phase_index, segment in enumerate(phase_segments):
                segment.phase_index = int(phase_index)
                segment.phase_count = int(total)
            segments.extend(phase_segments)
        return segments

    def _append_phase(
        self,
        output: list[CandidateSegment],
        *,
        onset_step: int,
        end_step: int,
        score_event: ScoreEvent,
        heuristic_family: str,
        key_signature: str,
        control_phase: str,
        coarse_family: str,
        event_duration_steps: int,
        horizon: int,
    ) -> None:
        start = int(np.clip(onset_step, 0, max(horizon - 1, 0)))
        end = int(np.clip(end_step, start + 1, horizon))
        if end - start < self.min_phase_duration_steps:
            return
        output.append(
            CandidateSegment(
                onset_step=start,
                end_step=end,
                segment_source=self.name,
                score_event_id=score_event.event_id,
                key_signature=key_signature,
                heuristic_family=heuristic_family,
                coarse_family=coarse_family,
                control_phase=control_phase,
                phase_index=0,
                phase_count=1,
                event_duration_steps=max(int(event_duration_steps), 1),
                chord_size=int(score_event.chord_size),
                key_center=float(score_event.key_center),
            )
        )


class DTWAssistedSegmenter(NoteAlignedSegmenter):
    name = "dtw_assisted"

    def __init__(
        self,
        pre_steps: int,
        post_steps: int,
        alignment_radius: int,
        template_window: int,
        note_local_horizon_steps: int | None = None,
    ):
        super().__init__(
            pre_steps=pre_steps,
            post_steps=post_steps,
            note_local_horizon_steps=note_local_horizon_steps,
        )
        self.alignment_radius = int(alignment_radius)
        self.template_window = int(template_window)

    def segment(self, episode, score_events: list[ScoreEvent]) -> list[CandidateSegment]:
        base_segments = super().segment(episode, score_events)
        if episode.hand_joints is None or not base_segments:
            return base_segments
        velocity = np.gradient(np.asarray(episode.hand_joints, dtype=np.float32), axis=0)
        magnitude = np.linalg.norm(velocity, axis=1)
        templates: dict[str, np.ndarray] = {}
        aligned: list[CandidateSegment] = []
        for segment in base_segments:
            signature = segment.key_signature or f"{segment.chord_size}"
            start = max(segment.onset_step, 0)
            end = min(start + self.template_window, magnitude.shape[0])
            excerpt = magnitude[start:end]
            if signature not in templates or excerpt.size == 0:
                templates[signature] = excerpt
                aligned.append(segment)
                continue
            best_offset = 0
            best_distance = float("inf")
            template = templates[signature]
            for offset in range(-self.alignment_radius, self.alignment_radius + 1):
                candidate_start = np.clip(start + offset, 0, max(magnitude.shape[0] - 1, 0))
                candidate_end = min(candidate_start + self.template_window, magnitude.shape[0])
                candidate_excerpt = magnitude[candidate_start:candidate_end]
                if candidate_excerpt.size == 0:
                    continue
                distance = dtw_distance(template, candidate_excerpt)
                if distance < best_distance:
                    best_distance = distance
                    best_offset = offset
            shifted_start = max(segment.onset_step + best_offset, 0)
            shifted_end = max(shifted_start + 1, min(segment.end_step + best_offset, magnitude.shape[0]))
            aligned.append(
                CandidateSegment(
                    onset_step=int(shifted_start),
                    end_step=int(shifted_end),
                    segment_source=self.name,
                    score_event_id=segment.score_event_id,
                    key_signature=segment.key_signature,
                    heuristic_family=segment.heuristic_family,
                    coarse_family=segment.coarse_family,
                    control_phase=segment.control_phase,
                    phase_index=segment.phase_index,
                    phase_count=segment.phase_count,
                    event_duration_steps=segment.event_duration_steps,
                    chord_size=segment.chord_size,
                    key_center=segment.key_center,
                )
            )
        return aligned


def dtw_distance(sequence_a: np.ndarray, sequence_b: np.ndarray) -> float:
    a = np.asarray(sequence_a, dtype=np.float32).reshape(-1)
    b = np.asarray(sequence_b, dtype=np.float32).reshape(-1)
    dp = np.full((a.size + 1, b.size + 1), np.inf, dtype=np.float32)
    dp[0, 0] = 0.0
    for i in range(1, a.size + 1):
        for j in range(1, b.size + 1):
            cost = abs(a[i - 1] - b[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[a.size, b.size] / max(a.size + b.size, 1))


class LearnedBoundarySafeSegmenter(BaseSegmenter):
    """CPU-safe segmentation from multi-modal boundary evidence (bounded per episode)."""

    name = "learned_boundary_safe"

    def __init__(self, config: dict[str, Any]):
        self._cfg = config
        self._min_len = int(config["min_segment_length"])
        self._max_len = int(config["max_segment_length"])
        self._max_peaks = int(config["max_candidate_boundaries_per_episode"])
        self._snap_tol = int(config.get("boundary_snap_tolerance_steps", 3))
        self._max_per_song = int(config.get("max_segments_per_song", 10**9))

    def segment(self, episode, score_events: list[ScoreEvent]) -> list[CandidateSegment]:
        del score_events
        hj = episode.hand_joints
        if hj is None:
            return []
        t_total = int(hj.shape[0])
        if t_total <= self._min_len:
            return [
                self._candidate_span(
                    0,
                    t_total,
                    peak=0.0,
                    source="trivial_episode",
                    snapped=-1,
                    raw_len=t_total,
                    filter_reason="",
                )
            ]

        actions = episode.actions
        goals = episode.goals
        piano = episode.piano_states
        vel = episode.joint_velocities
        if vel is None:
            vel = np.gradient(hj, axis=0).astype(np.float32)
        accel = np.gradient(vel, axis=0).astype(np.float32)

        w_act = float(self._cfg.get("boundary_score_action", 1.0))
        w_vc = float(self._cfg.get("boundary_score_vel_change", 0.85))
        w_ac = float(self._cfg.get("boundary_score_accel_change", 0.65))
        w_go = float(self._cfg.get("boundary_score_goal_onset", 1.15))
        w_gr = float(self._cfg.get("boundary_score_goal_release", 0.95))
        w_pc = float(self._cfg.get("boundary_score_piano_change", 0.75))

        interior = np.zeros(t_total, dtype=np.float32)
        for t in range(1, t_total):
            parts: list[float] = []
            if actions is not None and actions.shape[0] >= t_total:
                parts.append(w_act * float(np.linalg.norm(actions[t] - actions[t - 1])))
            parts.append(w_vc * float(np.linalg.norm(vel[t] - vel[t - 1])))
            parts.append(w_ac * float(np.linalg.norm(accel[t] - accel[t - 1])))
            roll = goals if goals is not None else piano
            if roll is not None and roll.shape[0] >= t_total:
                active = (roll > 0.5).astype(np.float32)
                d = active[t] - active[t - 1]
                parts.append(w_go * float(np.maximum(d, 0.0).sum()))
                parts.append(w_gr * float(np.maximum(-d, 0.0).sum()))
            if piano is not None and piano.shape[0] >= t_total:
                parts.append(w_pc * float(np.linalg.norm(piano[t] - piano[t - 1])))
            interior[t] = float(sum(parts))

        scale = float(np.percentile(interior[1:], 95) + 1e-6)
        scores = np.clip(interior / scale, 0.0, 12.0)

        onset_idx, release_idx = _score_event_roll_indices(goals, piano, t_total)
        peaks = _local_maxima_indices(scores)
        ranked = sorted(peaks, key=lambda idx: float(scores[idx]), reverse=True)
        chosen: list[int] = []
        for idx in ranked:
            if len(chosen) >= self._max_peaks:
                break
            if any(abs(idx - c) < self._min_len for c in chosen):
                continue
            snapped = _snap_boundary(idx, onset_idx, release_idx, self._snap_tol)
            chosen.append(int(snapped))
        chosen = sorted(set(chosen))
        boundaries = [0] + [b for b in chosen if 0 < b < t_total] + [t_total]
        boundaries = sorted(set(boundaries))

        boundaries = _enforce_max_segment_length(boundaries, t_total, self._max_len)
        boundaries = _merge_short_segments(boundaries, t_total, self._min_len)
        spans = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]

        segments: list[CandidateSegment] = []
        for start, end in spans:
            if end - start < 1:
                continue
            peak = float(scores[start : end].max()) if end > start else 0.0
            src = "boundary_peak" if peak > 1e-3 else "forced_span"
            snap_ev = _nearest_event_distance(start, onset_idx, release_idx)
            segments.append(
                self._candidate_span(
                    start,
                    end,
                    peak=peak,
                    source=src,
                    snapped=snap_ev,
                    raw_len=int(end - start),
                    filter_reason="",
                )
            )

        segments = _split_candidate_segments_max_len(segments, self._max_len)
        segments = _reduce_segment_count_by_merging(
            segments, self._max_per_song, scores, max_segment_length=self._max_len
        )
        segments = _split_candidate_segments_max_len(segments, self._max_len)
        if len(segments) > self._max_per_song:
            segments = _uniform_stride_fallback_segments(
                t_total=t_total,
                max_segments=self._max_per_song,
                max_len=self._max_len,
                min_len=self._min_len,
                scores=scores,
                source_name=self.name,
            )
        return segments

    def _candidate_span(
        self,
        onset: int,
        end: int,
        *,
        peak: float,
        source: str,
        snapped: int,
        raw_len: int,
        filter_reason: str,
    ) -> CandidateSegment:
        dur = max(int(end - onset), 1)
        return CandidateSegment(
            onset_step=int(onset),
            end_step=int(end),
            segment_source=self.name,
            score_event_id="",
            key_signature="",
            heuristic_family="boundary_safe",
            coarse_family="motion",
            control_phase="whole_event",
            phase_index=0,
            phase_count=1,
            event_duration_steps=dur,
            chord_size=0,
            key_center=0.0,
            boundary_score_peak=float(peak),
            boundary_source=str(source),
            snapped_to_score_event=int(snapped),
            raw_segment_length=int(raw_len),
            segment_filter_reason=str(filter_reason),
        )


def _score_event_roll_indices(
    goals: np.ndarray | None, piano: np.ndarray | None, t_total: int
) -> tuple[np.ndarray, np.ndarray]:
    roll = goals if goals is not None else piano
    if roll is None or roll.shape[0] < 2:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)
    active = (roll[:t_total] > 0.5).astype(np.float32)
    if active.shape[0] < 2:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)
    d = active[1:] - active[:-1]
    onsets = np.flatnonzero(d.sum(axis=1) > 0) + 1
    releases = np.flatnonzero((-d).sum(axis=1) > 0) + 1
    return onsets.astype(np.int64), releases.astype(np.int64)


def _local_maxima_indices(scores: np.ndarray) -> list[int]:
    t = int(scores.shape[0])
    out: list[int] = []
    for i in range(1, max(t - 1, 1)):
        if scores[i] >= scores[i - 1] and scores[i] >= scores[i + 1]:
            out.append(i)
    return out


def _snap_boundary(b: int, onsets: np.ndarray, releases: np.ndarray, tol: int) -> int:
    best = b
    best_d = tol + 1
    for arr in (onsets, releases):
        for e in arr.tolist():
            d = abs(int(e) - b)
            if d <= tol and d < best_d:
                best_d = d
                best = int(e)
    return best


def _nearest_event_distance(b: int, onsets: np.ndarray, releases: np.ndarray) -> int:
    best = -1
    best_d = 10**9
    for arr in (onsets, releases):
        for e in arr.tolist():
            d = abs(int(e) - b)
            if d < best_d:
                best_d = d
                best = int(e)
    return best if best_d <= 12 else -1


def _enforce_max_segment_length(boundaries: list[int], t_total: int, max_len: int) -> list[int]:
    b = sorted(set(boundaries))
    if not b or b[0] != 0:
        b = [0] + [x for x in b if x > 0]
    if b[-1] != t_total:
        b = b + [t_total]
    out: list[int] = [b[0]]
    for i in range(len(b) - 1):
        start, end = int(b[i]), int(b[i + 1])
        cur = start
        while end - cur > max_len:
            cur = cur + max_len
            out.append(cur)
        out.append(end)
    return sorted(set(out))


def _merge_short_segments(boundaries: list[int], t_total: int, min_len: int) -> list[int]:
    b = sorted(set(boundaries))
    if not b:
        return [0, t_total]
    if b[0] != 0:
        b.insert(0, 0)
    if b[-1] != t_total:
        b.append(t_total)
    changed = True
    while changed:
        changed = False
        nb = [b[0]]
        for k in range(1, len(b) - 1):
            if b[k] - nb[-1] < min_len:
                changed = True
                continue
            nb.append(b[k])
        nb.append(b[-1])
        if nb != b:
            changed = True
        b = nb
    return b


def _uniform_stride_fallback_segments(
    *,
    t_total: int,
    max_segments: int,
    max_len: int,
    min_len: int,
    scores: np.ndarray,
    source_name: str,
) -> list[CandidateSegment]:
    """Last resort: partition [0, t_total) into at most max_segments spans, each in [min_len, max_len]."""
    n = min(int(max_segments), max(1, t_total // max(min_len, 1)))
    boundaries = [0]
    approx = max(min_len, min(max_len, int(np.ceil(t_total / float(n)))))
    cur = 0
    while cur < t_total and len(boundaries) < n + 1:
        nxt = min(cur + approx, t_total)
        if nxt - cur < min_len and nxt < t_total:
            nxt = min(cur + min_len, t_total)
        if nxt <= cur:
            nxt = min(cur + min_len, t_total)
        boundaries.append(nxt)
        cur = nxt
    if boundaries[-1] != t_total:
        boundaries[-1] = t_total
    boundaries = _enforce_max_segment_length(boundaries, t_total, max_len)
    boundaries = _merge_short_segments(boundaries, t_total, min_len)
    out: list[CandidateSegment] = []
    for i in range(len(boundaries) - 1):
        s, e = int(boundaries[i]), int(boundaries[i + 1])
        if e <= s:
            continue
        peak = float(scores[s:e].max()) if e > s else 0.0
        out.append(
            CandidateSegment(
                onset_step=s,
                end_step=e,
                segment_source=source_name,
                score_event_id="",
                key_signature="",
                heuristic_family="boundary_safe",
                coarse_family="uniform_fallback",
                control_phase="whole_event",
                phase_index=0,
                phase_count=1,
                event_duration_steps=max(e - s, 1),
                chord_size=0,
                key_center=0.0,
                boundary_score_peak=peak,
                boundary_source="uniform_stride_fallback",
                snapped_to_score_event=-1,
                raw_segment_length=int(e - s),
                segment_filter_reason="max_segments_per_song_fallback",
            )
        )
    return out[:max_segments]


def _split_candidate_segments_max_len(segments: list[CandidateSegment], max_len: int) -> list[CandidateSegment]:
    out: list[CandidateSegment] = []
    for seg in segments:
        span = int(seg.end_step - seg.onset_step)
        if span <= max_len:
            out.append(seg)
            continue
        s = int(seg.onset_step)
        end = int(seg.end_step)
        while s < end:
            e = min(s + max_len, end)
            out.append(
                CandidateSegment(
                    onset_step=s,
                    end_step=e,
                    segment_source=seg.segment_source,
                    score_event_id="",
                    key_signature="",
                    heuristic_family=seg.heuristic_family,
                    coarse_family="motion_split",
                    control_phase="whole_event",
                    phase_index=0,
                    phase_count=1,
                    event_duration_steps=max(e - s, 1),
                    chord_size=0,
                    key_center=0.0,
                    boundary_score_peak=float(seg.boundary_score_peak),
                    boundary_source="max_length_split",
                    snapped_to_score_event=-1,
                    raw_segment_length=int(e - s),
                    segment_filter_reason="split_exceeded_max_segment_length",
                )
            )
            s = e
    return out


def _reduce_segment_count_by_merging(
    segments: list[CandidateSegment], max_count: int, scores: np.ndarray, *, max_segment_length: int
) -> list[CandidateSegment]:
    if len(segments) <= max_count:
        return segments
    segs = list(segments)
    while len(segs) > max_count:
        best_i = 0
        best_cost = float("inf")
        for i in range(len(segs) - 1):
            a, b = segs[i], segs[i + 1]
            merged_len = int(b.end_step - a.onset_step)
            if merged_len > max_segment_length:
                continue
            mid = a.end_step
            cost = float(scores[mid]) if 0 <= mid < len(scores) else 0.0
            if cost < best_cost:
                best_cost = cost
                best_i = i
        if best_cost is float("inf"):
            break
        left, right = segs[best_i], segs[best_i + 1]
        merged = CandidateSegment(
            onset_step=int(left.onset_step),
            end_step=int(right.end_step),
            segment_source=left.segment_source,
            score_event_id="",
            key_signature="",
            heuristic_family="boundary_safe",
            coarse_family="motion_merge",
            control_phase="whole_event",
            phase_index=0,
            phase_count=1,
            event_duration_steps=int(right.end_step - left.onset_step),
            chord_size=0,
            key_center=0.0,
            boundary_score_peak=float(max(left.boundary_score_peak, right.boundary_score_peak)),
            boundary_source="merged_for_song_cap",
            snapped_to_score_event=-1,
            raw_segment_length=int(right.end_step - left.onset_step),
            segment_filter_reason="downsampled_max_segments_per_song",
        )
        segs = segs[:best_i] + [merged] + segs[best_i + 2 :]
    return segs


def classify_interval_family(event: ScoreEvent) -> str:
    if event.chord_size >= 3:
        return "chord"
    if event.inter_onset_steps == 0:
        return "stacked"
    return "single"


def coarse_family_for_event(event: ScoreEvent) -> str:
    duration = max(int(event.end_step - event.onset_step), 1)
    if int(event.chord_size) >= 3:
        return "chord_press"
    if duration >= 24:
        return "hold"
    if int(event.inter_onset_steps) <= 2 and duration <= 10:
        return "short_sequence"
    if int(event.chord_size) >= 2:
        return "chord_press"
    return "single_press"


def build_segmenter(config: dict[str, Any]) -> BaseSegmenter:
    strategy = str(config["segmentation_strategy"])
    if strategy == "fixed_window":
        return FixedWindowSegmenter(window_steps=config["window_steps"], stride_steps=config["stride_steps"])
    if strategy == "changepoint":
        return ChangePointSegmenter(
            window_steps=config["window_steps"],
            min_gap_steps=config["min_gap_steps"],
            velocity_quantile=config["velocity_quantile"],
            acceleration_quantile=config["acceleration_quantile"],
        )
    if strategy == "note_aligned":
        return NoteAlignedSegmenter(
            pre_steps=config["pre_steps"],
            post_steps=config["post_steps"],
            note_local_horizon_steps=config.get("note_local_horizon_steps"),
        )
    if strategy == "dtw_assisted":
        return DTWAssistedSegmenter(
            pre_steps=config["pre_steps"],
            post_steps=config["post_steps"],
            alignment_radius=config["alignment_radius"],
            template_window=config["dtw_template_window"],
            note_local_horizon_steps=config.get("note_local_horizon_steps"),
        )
    if strategy == "event_phase_aligned":
        return EventPhaseSegmenter(
            pre_steps=config["pre_steps"],
            post_steps=config["post_steps"],
            note_local_horizon_steps=config.get("note_local_horizon_steps"),
            onset_window_steps_single=int(config.get("onset_window_steps_single", 4)),
            onset_window_steps_chord=int(config.get("onset_window_steps_chord", 6)),
            approach_window_steps=int(config.get("approach_window_steps", config.get("pre_steps", 4))),
            release_window_steps=int(config.get("release_window_steps", config.get("post_steps", 4))),
            hold_min_duration_steps=int(config.get("hold_min_duration_steps", 8)),
            hold_tail_steps=int(config.get("hold_tail_steps", 2)),
            transition_max_gap_steps=int(config.get("transition_max_gap_steps", 8)),
            transition_window_steps=int(config.get("transition_window_steps", 8)),
            staccato_duration_steps=int(config.get("staccato_duration_steps", 6)),
            min_phase_duration_steps=int(config.get("min_phase_duration_steps", 2)),
        )
    if strategy == "learned_boundary_safe":
        return LearnedBoundarySafeSegmenter(config)
    raise ValueError(f"Unknown segmentation strategy: {strategy}")

def _atomic_save_npz(path: Path, **payload: Any) -> None:
    tmp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
    saved_path = save_npz(tmp_path, **payload)
    os.replace(saved_path, path)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def _append_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    write_header = not path.exists() or path.stat().st_size == 0
    df.to_csv(path, mode="a", header=write_header, index=False)


def _read_partial_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_resume_manifest(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.exists():
        return {
            "status": "new",
            "next_chunk_index": 0,
            "processed_episode_ids": [],
            "chunk_files": [],
            "num_segments_written": 0,
            "num_score_events_written": 0,
            "segment_strategy": None,
        }
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload.setdefault("status", "partial")
    payload.setdefault("next_chunk_index", 0)
    payload.setdefault("processed_episode_ids", [])
    payload.setdefault("chunk_files", [])
    payload.setdefault("num_segments_written", 0)
    payload.setdefault("num_score_events_written", 0)
    payload.setdefault("segment_strategy", None)
    return payload


def _write_resume_manifest(
    manifest_path: Path,
    *,
    status: str,
    next_chunk_index: int,
    processed_episode_ids: list[str],
    chunk_files: list[str],
    num_segments_written: int,
    num_score_events_written: int,
    config: dict[str, Any],
) -> None:
    _atomic_write_json(
        manifest_path,
        {
            "status": status,
            "next_chunk_index": int(next_chunk_index),
            "processed_episode_ids": [str(item) for item in processed_episode_ids],
            "chunk_files": [str(item) for item in chunk_files],
            "num_segments_written": int(num_segments_written),
            "num_score_events_written": int(num_score_events_written),
            "segment_strategy": str(config["segmentation_strategy"]),
            "segment_chunk_size": int(config["segment_chunk_size"]),
        },
    )


def _collect_chunk_rows(flushed_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_chunk: dict[str, list[dict[str, Any]]] = {}
    for row in flushed_rows:
        chunk_name = str(row["chunk_path"])
        by_chunk.setdefault(chunk_name, []).append(row)
    return by_chunk


def _validate_existing_chunks(segments_dir: Path, chunk_files: list[str]) -> list[str]:
    valid: list[str] = []
    for chunk_name in chunk_files:
        chunk_path = segments_dir / chunk_name
        if not chunk_path.exists():
            continue
        try:
            with np.load(chunk_path, allow_pickle=True) as bundle:
                if "segment_ids" not in bundle:
                    continue
        except Exception:
            continue
        valid.append(chunk_name)
    return valid

def run_segmentation(manifest_df: pd.DataFrame, output_dir: Path, config: dict[str, Any], logger: logging.Logger | None = None) -> dict[str, Path]:
    if not online_segment_processing_enabled(config):
        return run_segmentation_legacy(manifest_df=manifest_df, output_dir=output_dir, config=config)
    return run_segmentation_slim(manifest_df=manifest_df, output_dir=output_dir, config=config, logger=logger)


def run_segmentation_legacy(manifest_df: pd.DataFrame, output_dir: Path, config: dict[str, Any]) -> dict[str, Path]:
    segments_dir = output_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    table_base = segments_dir / "segment_index"
    score_base = segments_dir / "score_events"
    manifest_path = segments_dir / "segment_manifest.json"

    segment_partial_csv = segments_dir / "segment_index.partial.csv"
    score_partial_csv = segments_dir / "score_events.partial.csv"

    final_segment_csv = table_base.with_suffix(".csv")
    final_score_csv = score_base.with_suffix(".csv")
    force = bool(config.get("force", False))

    if force:
        for path in [segment_partial_csv, score_partial_csv, final_segment_csv, final_score_csv, manifest_path]:
            if path.exists():
                path.unlink()

    if final_segment_csv.exists() and final_score_csv.exists() and not force:
        return {"segment_table_base": table_base, "score_table_base": score_base, "manifest_path": manifest_path}

    resume_state = _load_resume_manifest(manifest_path)
    resume_state["chunk_files"] = _validate_existing_chunks(segments_dir, resume_state.get("chunk_files", []))

    segmenter = build_segmenter(config)
    writer = SegmentChunkWriter(
        output_dir=segments_dir,
        chunk_size=int(config["segment_chunk_size"]),
        start_chunk_index=int(resume_state.get("next_chunk_index", 0)),
        existing_chunk_files=resume_state.get("chunk_files", []),
    )

    processed_episode_ids = {str(item) for item in resume_state.get("processed_episode_ids", [])}

    existing_segment_df = _read_partial_csv(segment_partial_csv)
    existing_score_df = _read_partial_csv(score_partial_csv)

    num_segments_written = int(len(existing_segment_df))
    existing_score_ids = (
        set(existing_score_df["event_id"].astype(str).tolist())
        if not existing_score_df.empty and "event_id" in existing_score_df.columns
        else set()
    )
    num_score_events_written = int(len(existing_score_ids))

    remaining_df = manifest_df[~manifest_df["episode_id"].astype(str).isin(processed_episode_ids)].reset_index(drop=True)

    _write_resume_manifest(
        manifest_path,
        status="running",
        next_chunk_index=writer.chunk_index,
        processed_episode_ids=sorted(processed_episode_ids),
        chunk_files=writer.chunk_files,
        num_segments_written=num_segments_written,
        num_score_events_written=num_score_events_written,
        config=config,
    )

    for row in tqdm(remaining_df.itertuples(index=False), total=len(remaining_df), desc="Segment episodes"):
        episode = load_episode_record(row._asdict())
        if episode.hand_joints is None:
            processed_episode_ids.add(str(episode.episode_id))
            _write_resume_manifest(
                manifest_path,
                status="running",
                next_chunk_index=writer.chunk_index,
                processed_episode_ids=sorted(processed_episode_ids),
                chunk_files=writer.chunk_files,
                num_segments_written=num_segments_written,
                num_score_events_written=num_score_events_written,
                config=config,
            )
            continue

        if episode.note_path is not None and episode.note_path.exists():
            try:
                score_events = load_note_events(
                    episode.note_path,
                    episode.control_timestep,
                    chord_tolerance_steps=int(config["chord_tolerance_steps"]),
                    song_id=episode.song_id,
                    episode_id=episode.episode_id,
                )
            except Exception:
                score_events = infer_events_from_goal_roll(
                    episode.goals if episode.goals is not None else episode.piano_states,
                    song_id=episode.song_id,
                    episode_id=episode.episode_id,
                    control_timestep=episode.control_timestep,
                    chord_tolerance_steps=int(config["chord_tolerance_steps"]),
                    source="goals" if episode.goals is not None else "piano_states",
                )
        else:
            roll = episode.goals if episode.goals is not None else episode.piano_states
            score_events = infer_events_from_goal_roll(
                roll,
                song_id=episode.song_id,
                episode_id=episode.episode_id,
                control_timestep=episode.control_timestep,
                chord_tolerance_steps=int(config["chord_tolerance_steps"]),
                source="goals" if episode.goals is not None else "piano_states",
            )

        new_score_rows = []
        for event in score_events:
            event_row = event.as_row()
            event_id = str(event_row["event_id"])
            if event_id not in existing_score_ids:
                existing_score_ids.add(event_id)
                new_score_rows.append(event_row)

        candidate_segments = segmenter.segment(episode, score_events)
        for index, candidate in enumerate(candidate_segments):
            arrays = slice_segment_arrays(episode, candidate.onset_step, candidate.end_step)
            hand_joints = arrays["hand_joints"]
            if hand_joints is None:
                continue

            velocity = arrays["joint_velocities"]
            if velocity is None:
                velocity = np.gradient(hand_joints, episode.control_timestep, axis=0).astype(np.float32)
                arrays["joint_velocities"] = velocity

            context = score_context_from_roll(
                episode.goals if episode.goals is not None else episode.piano_states,
                candidate.onset_step,
            )

            record = SegmentRecord(
                segment_id=f"{episode.episode_id}_segment_{index:06d}",
                song_id=episode.song_id,
                episode_id=episode.episode_id,
                onset_step=candidate.onset_step,
                end_step=candidate.end_step,
                duration_steps=max(candidate.end_step - candidate.onset_step, 1),
                segment_source=candidate.segment_source,
                score_event_id=candidate.score_event_id,
                key_signature=candidate.key_signature,
                chunk_path="",
                chunk_index=-1,
                heuristic_family=candidate.heuristic_family,
                coarse_family=candidate.coarse_family,
                control_phase=candidate.control_phase,
                phase_index=int(candidate.phase_index),
                phase_count=int(candidate.phase_count),
                event_duration_steps=int(candidate.event_duration_steps),
                motion_energy=float(np.linalg.norm(velocity, axis=1).mean()),
                chord_size=int(candidate.chord_size),
                key_center=float(candidate.key_center),
                start_state_norm=float(np.linalg.norm(hand_joints[0])),
                end_state_norm=float(np.linalg.norm(hand_joints[-1])),
                score_context_json=dumps_score_context(context),
                boundary_score_peak=float(getattr(candidate, "boundary_score_peak", 0.0)),
                boundary_source=str(getattr(candidate, "boundary_source", "")),
                snapped_to_score_event=int(getattr(candidate, "snapped_to_score_event", -1)),
                raw_segment_length=int(getattr(candidate, "raw_segment_length", 0)),
                segment_filter_reason=str(getattr(candidate, "segment_filter_reason", "")),
            )
            writer.add(record.as_row() | {"split": row.split}, arrays)

        flushed_rows = writer.flush()
        if flushed_rows:
            _append_rows_csv(segment_partial_csv, flushed_rows)
            num_segments_written += len(flushed_rows)

        if new_score_rows:
            _append_rows_csv(score_partial_csv, new_score_rows)
            num_score_events_written = len(existing_score_ids)

        processed_episode_ids.add(str(episode.episode_id))
        _write_resume_manifest(
            manifest_path,
            status="running",
            next_chunk_index=writer.chunk_index,
            processed_episode_ids=sorted(processed_episode_ids),
            chunk_files=writer.chunk_files,
            num_segments_written=num_segments_written,
            num_score_events_written=num_score_events_written,
            config=config,
        )

    final_flush_rows = writer.flush()
    if final_flush_rows:
        _append_rows_csv(segment_partial_csv, final_flush_rows)
        num_segments_written += len(final_flush_rows)

    segment_df = _read_partial_csv(segment_partial_csv)
    score_df = _read_partial_csv(score_partial_csv)
    if not score_df.empty and "event_id" in score_df.columns:
        score_df = score_df.drop_duplicates(subset=["event_id"]).reset_index(drop=True)

    write_table(segment_df, table_base)
    write_table(score_df, score_base)

    _write_resume_manifest(
        manifest_path,
        status="completed",
        next_chunk_index=writer.chunk_index,
        processed_episode_ids=sorted(processed_episode_ids),
        chunk_files=writer.chunk_files,
        num_segments_written=int(len(segment_df)),
        num_score_events_written=int(len(score_df)),
        config=config,
    )

    return {"segment_table_base": table_base, "score_table_base": score_base, "manifest_path": manifest_path}


def run_segmentation_slim(
    manifest_df: pd.DataFrame,
    output_dir: Path,
    config: dict[str, Any],
    logger: logging.Logger | None = None,
) -> dict[str, Path]:
    segments_dir = output_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)
    table_base = segments_dir / "segment_index"
    score_base = segments_dir / "score_events"
    manifest_path = segments_dir / "segment_manifest.json"
    slim_paths = resolve_slim_cache_paths(output_dir, config)
    compact_manifest_path = compact_store_manifest_path(slim_paths)

    if bool(config.get("force", False)):
        _reset_segmentation_outputs(segments_dir=segments_dir, slim_root=slim_paths.root)

    existing_segment_df = _read_optional_table(table_base)
    existing_score_df = _read_optional_table(score_base)
    slim_segment_df = load_slim_index_table(slim_paths)
    existing_segment_df = compose_segment_index(existing_segment_df, slim_segment_df)
    previous_store_manifest = read_compact_store_manifest(slim_paths)

    incomplete_chunks = list_incomplete_slim_chunks(slim_paths)
    if incomplete_chunks and logger is not None:
        logger.warning("Ignoring %d incomplete online Stage 1 chunk(s) during resume.", len(incomplete_chunks))

    migration_summary = {"migrated_chunks": 0, "deleted_raw_chunks": 0, "source_raw_bytes": 0}
    if bool(config.get("migrate_existing_segment_chunks", False)) and not existing_segment_df.empty:
        migration_summary = migrate_existing_segment_chunks(
            segment_df=existing_segment_df,
            segments_dir=segments_dir,
            output_dir=output_dir,
            config=config,
            logger=logger,
        )
        slim_segment_df = load_slim_index_table(slim_paths)
        existing_segment_df = compose_segment_index(existing_segment_df, slim_segment_df)

    processed_episode_ids = load_completed_episodes(slim_paths)
    if not existing_segment_df.empty and "episode_id" in existing_segment_df.columns:
        processed_episode_ids.update(existing_segment_df["episode_id"].astype(str).tolist())

    remaining_df = manifest_df[
        ~manifest_df["episode_id"].astype(str).isin(processed_episode_ids)
    ].reset_index(drop=True)
    skipped_episodes = int(len(manifest_df) - len(remaining_df))

    segmenter = build_segmenter(config)
    slim_writer = OnlineSegmentWriter(
        output_dir=output_dir,
        chunk_size=int(config["segment_chunk_size"]),
        start_chunk_index=next_slim_chunk_index(slim_paths),
        config=config,
        logger=logger,
    )
    raw_writer = None
    if save_raw_segment_chunks_enabled(config):
        raw_writer = SegmentChunkWriter(
            output_dir=segments_dir,
            chunk_size=int(config["segment_chunk_size"]),
            start_chunk_index=_next_raw_chunk_index(segments_dir),
        )

    new_score_rows: list[dict[str, Any]] = []
    song_totals: dict[str, int] = defaultdict(int)
    if not existing_segment_df.empty and "song_id" in existing_segment_df.columns:
        for sid, cnt in existing_segment_df.groupby("song_id").size().items():
            song_totals[str(sid)] = int(cnt)
    max_total_segments = int(config.get("max_total_segments", 10**12))
    global_remaining = [max(0, max_total_segments - len(existing_segment_df))]
    max_per_song = int(config.get("max_segments_per_song", 10**12))
    seg_wall0 = time.perf_counter()
    episodes_done_counter = 0
    if raw_writer is None:
        segment_num_workers = _positive_int(config.get("segment_num_workers")) or 0
        prepared_batches = _iter_prepared_episode_batches(
            manifest_df=remaining_df,
            config=config,
            max_workers=segment_num_workers,
        )
        for batch in tqdm(prepared_batches, total=len(remaining_df), desc="Segment episodes"):
            episodes_done_counter += 1
            ep_wall0 = time.perf_counter()
            new_score_rows.extend(batch.score_rows)
            if batch.prepared_segments:
                capped = apply_segment_volume_caps(
                    list(batch.prepared_segments),
                    song_id=batch.song_id,
                    song_totals=song_totals,
                    max_per_song=max_per_song,
                    global_remaining=global_remaining,
                    logger=logger,
                )
                ep_elapsed = time.perf_counter() - ep_wall0
                hard_cap = config.get("max_episode_processing_seconds_hard")
                if hard_cap is not None and ep_elapsed > float(hard_cap):
                    raise RuntimeError(
                        f"max_episode_processing_seconds_hard exceeded for episode {batch.episode_id}: {ep_elapsed:.1f}s > {hard_cap}s"
                    )
                warn_cap = config.get("max_episode_processing_seconds_warn")
                if logger is not None and warn_cap is not None and ep_elapsed > float(warn_cap):
                    logger.warning("Episode %s segmentation slow: %.1fs", batch.episode_id, ep_elapsed)
                slim_writer.begin_episode(song_id=batch.song_id, episode_id=batch.episode_id)
                for segment in capped:
                    slim_writer.append_segment(
                        row=segment.row,
                        feature_vector=segment.feature_vector,
                        feature_names=segment.feature_names,
                        gmr_target=segment.gmr_target,
                        target_name=segment.gmr_target_name,
                        raw_segment_bytes=segment.raw_bytes_estimate,
                    )
                _record_episode_progress(slim_paths, slim_writer.end_episode())
            else:
                append_episode_progress(
                    slim_paths,
                    _episode_progress_payload(
                        song_id=batch.song_id,
                        episode_id=batch.episode_id,
                        num_segments=0,
                    ),
                )
            processed_episode_ids.add(batch.episode_id)
            log_every = int(config.get("runtime_log_every_n_episodes", 50))
            if log_every > 0 and episodes_done_counter % log_every == 0:
                elapsed = time.perf_counter() - seg_wall0
                runtime_guard_projected_walltime(
                    elapsed_seconds=elapsed,
                    episodes_done=episodes_done_counter,
                    episodes_total=max(len(remaining_df), 1),
                    max_walltime_seconds=float(config.get("max_walltime_seconds", 86400)),
                    logger=logger,
                )
    else:
        for row in tqdm(remaining_df.itertuples(index=False), total=len(remaining_df), desc="Segment episodes"):
            episode_id = str(row.episode_id)
            episode = load_episode_record(row._asdict())
            score_events = _load_or_infer_score_events(episode=episode, config=config)
            new_score_rows.extend([event.as_row() for event in score_events])
            prepared_segments = list(
                iter_prepared_segments(
                    manifest_row=row,
                    episode=episode,
                    score_events=score_events,
                    segmenter=segmenter,
                    config=config,
                )
            )
            if not prepared_segments:
                append_episode_progress(
                    slim_paths,
                    _episode_progress_payload(song_id=str(row.song_id), episode_id=episode_id, num_segments=0),
                )
                processed_episode_ids.add(episode_id)
                continue
            raw_rows = raw_writer.write_episode(
                rows=[dict(segment.row) for segment in prepared_segments],
                arrays=[segment.arrays for segment in prepared_segments],
            )
            slim_writer.begin_episode(song_id=str(row.song_id), episode_id=episode_id)
            for segment, raw_row in zip(prepared_segments, raw_rows):
                row_payload = dict(segment.row)
                row_payload["raw_chunk_path"] = str(raw_row["chunk_path"])
                row_payload["raw_chunk_index"] = int(raw_row["chunk_index"])
                slim_writer.append_segment(
                    row=row_payload,
                    feature_vector=segment.feature_vector,
                    feature_names=segment.feature_names,
                    gmr_target=segment.gmr_target,
                    target_name=segment.gmr_target_name,
                    raw_segment_bytes=segment.raw_bytes_estimate,
                )
            _record_episode_progress(slim_paths, slim_writer.end_episode())
            processed_episode_ids.add(episode_id)

    _record_episode_progress(slim_paths, slim_writer.flush())
    slim_segment_df = load_slim_index_table(slim_paths)
    segment_df = compose_segment_index(existing_segment_df, slim_segment_df)
    score_df = _merge_score_tables(existing_score_df=existing_score_df, new_rows=new_score_rows)
    store_summary = summarize_slim_cache(slim_paths)
    total_raw_estimate = int(previous_store_manifest.get("estimated_raw_segment_bytes", 0))
    total_raw_estimate += int(migration_summary.get("source_raw_bytes", 0))
    total_raw_estimate += int(slim_writer.stats["estimated_raw_segment_bytes"])
    estimated_bytes_per_1k = float(store_summary["total_bytes_on_disk"] * 1000.0 / max(store_summary["num_segments"], 1))
    estimated_reduction = (
        float(total_raw_estimate / max(store_summary["total_bytes_on_disk"], 1))
        if total_raw_estimate > 0 and store_summary["total_bytes_on_disk"] > 0
        else None
    )
    compact_manifest = {
        "status": "completed",
        "online_segment_processing": True,
        "save_raw_segment_chunks": save_raw_segment_chunks_enabled(config),
        "online_storage_format": resolve_online_storage_format(config),
        "num_segments": int(store_summary["num_segments"]),
        "num_chunks": int(store_summary["num_chunks"]),
        "feature_dim": int(store_summary["feature_dim"]),
        "gmr_target_steps": int(store_summary["gmr_horizon"]),
        "gmr_target_dim": int(store_summary["gmr_dim"]),
        "episodes_completed": int(len(load_completed_episodes(slim_paths))),
        "episodes_processed_this_run": int(slim_writer.stats["episodes_processed"]),
        "segments_written_this_run": int(slim_writer.stats["segments_written"]),
        "feature_rows_written_this_run": int(slim_writer.stats["feature_rows_written"]),
        "bytes_written_this_run": int(slim_writer.stats["bytes_written"]),
        "total_bytes_on_disk": int(store_summary["total_bytes_on_disk"]),
        "bytes_per_1000_segments": estimated_bytes_per_1k,
        "estimated_raw_segment_bytes": int(total_raw_estimate),
        "estimated_storage_reduction_vs_legacy": estimated_reduction,
        "incomplete_chunks_ignored": store_summary["incomplete_chunks"],
        "migrated_raw_chunks": int(migration_summary["migrated_chunks"]),
        "deleted_raw_chunks": int(migration_summary["deleted_raw_chunks"]),
        "skipped_episodes": int(skipped_episodes),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_compact_store_manifest(slim_paths, compact_manifest)
    write_table(segment_df, table_base)
    write_table(score_df, score_base)
    write_json(
        {
            "status": "completed",
            "num_segments": int(len(segment_df)),
            "num_score_events": int(len(score_df)),
            "chunk_files": sorted(path.name for path in segments_dir.glob("segment_chunk_*.npz")),
            "slim_chunk_files": collect_slim_chunk_names(slim_paths, completed_only=True),
            "segment_strategy": config["segmentation_strategy"],
            "online_segment_processing": True,
            "save_raw_segment_chunks": save_raw_segment_chunks_enabled(config),
            "online_storage_format": resolve_online_storage_format(config),
            "feature_dim": int(store_summary["feature_dim"]),
            "gmr_target_steps": int(store_summary["gmr_horizon"]),
            "gmr_target_dim": int(store_summary["gmr_dim"]),
            "compact_store_total_bytes": int(store_summary["total_bytes_on_disk"]),
            "compact_store_manifest_path": str(compact_manifest_path.resolve()),
            "bytes_per_1000_segments": estimated_bytes_per_1k,
            "estimated_storage_reduction_vs_legacy": estimated_reduction,
            "write_full_segment_cache": save_raw_segment_chunks_enabled(config),
            "write_slim_cache": True,
            "migrated_raw_chunks": int(migration_summary["migrated_chunks"]),
            "deleted_raw_chunks": int(migration_summary["deleted_raw_chunks"]),
            "skipped_episodes": int(skipped_episodes),
            "incomplete_chunks_ignored": store_summary["incomplete_chunks"],
        },
        manifest_path,
    )
    if logger is not None:
        logger.info(
            "Segmented %d episodes into %d segments (%d online chunks, %d raw chunks kept, %.2f MiB compact store, %.2f MiB/1k segments).",
            len(processed_episode_ids),
            len(segment_df),
            len(collect_slim_chunk_names(slim_paths, completed_only=True)),
            len(list(segments_dir.glob("segment_chunk_*.npz"))),
            store_summary["total_bytes_on_disk"] / (1024.0 * 1024.0),
            estimated_bytes_per_1k / (1024.0 * 1024.0),
        )
    return {
        "segment_table_base": table_base,
        "score_table_base": score_base,
        "manifest_path": manifest_path,
        "compact_store_manifest_path": compact_manifest_path,
    }


def _reset_segmentation_outputs(segments_dir: Path, slim_root: Path) -> None:
    if segments_dir.exists():
        shutil.rmtree(segments_dir)
    if slim_root.exists():
        shutil.rmtree(slim_root)
    segments_dir.mkdir(parents=True, exist_ok=True)


def _read_optional_table(base_path: Path) -> pd.DataFrame:
    if base_path.with_suffix(".parquet").exists() or base_path.with_suffix(".csv").exists():
        return read_table(base_path)
    return pd.DataFrame()


def _record_episode_progress(slim_paths, payloads: list[dict[str, Any]]) -> None:
    for payload in payloads:
        append_episode_progress(slim_paths, payload)


def _episode_progress_payload(song_id: str, episode_id: str, num_segments: int) -> dict[str, Any]:
    return {
        "song_id": song_id,
        "episode_id": episode_id,
        "num_segments": int(num_segments),
        "status": "completed",
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }


def _merge_score_tables(existing_score_df: pd.DataFrame, new_rows: list[dict[str, Any]]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if not existing_score_df.empty:
        frames.append(existing_score_df)
    if new_rows:
        frames.append(pd.DataFrame(new_rows))
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    if "event_id" in merged.columns:
        merged = merged.drop_duplicates(subset=["event_id"]).reset_index(drop=True)
    return merged


def _load_or_infer_score_events(episode, config: dict[str, Any]) -> list[ScoreEvent]:
    if episode.note_path is not None and episode.note_path.exists():
        try:
            return load_note_events(
                episode.note_path,
                episode.control_timestep,
                chord_tolerance_steps=int(config["chord_tolerance_steps"]),
                song_id=episode.song_id,
                episode_id=episode.episode_id,
            )
        except Exception:
            pass
    roll = episode.goals if episode.goals is not None else episode.piano_states
    if roll is None:
        return []
    return infer_events_from_goal_roll(
        roll,
        song_id=episode.song_id,
        episode_id=episode.episode_id,
        control_timestep=episode.control_timestep,
        chord_tolerance_steps=int(config["chord_tolerance_steps"]),
        source="goals" if episode.goals is not None else "piano_states",
    )


def _iter_prepared_episode_batches(
    manifest_df: pd.DataFrame,
    config: dict[str, Any],
    max_workers: int,
) -> Iterator[PreparedEpisodeBatch]:
    payloads = [row._asdict() for row in manifest_df.itertuples(index=False)]
    if max_workers <= 1 or len(payloads) <= 1:
        for payload in payloads:
            yield _prepare_episode_batch(payload, config)
        return

    in_flight_limit = max(max_workers * 2, 1)
    iterator = iter(payloads)
    futures: deque = deque()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for _ in range(min(in_flight_limit, len(payloads))):
            payload = next(iterator, None)
            if payload is None:
                break
            futures.append(executor.submit(_prepare_episode_batch, payload, config))
        while futures:
            future = futures.popleft()
            yield future.result()
            payload = next(iterator, None)
            if payload is not None:
                futures.append(executor.submit(_prepare_episode_batch, payload, config))


def _prepare_episode_batch(manifest_row_payload: dict[str, Any], config: dict[str, Any]) -> PreparedEpisodeBatch:
    manifest_row = SimpleNamespace(**manifest_row_payload)
    episode = load_episode_record(manifest_row_payload)
    score_events = _load_or_infer_score_events(episode=episode, config=config)
    prepared_segments = list(
        iter_prepared_segments(
            manifest_row=manifest_row,
            episode=episode,
            score_events=score_events,
            segmenter=build_segmenter(config),
            config=config,
        )
    )
    return PreparedEpisodeBatch(
        song_id=str(manifest_row.song_id),
        episode_id=str(manifest_row.episode_id),
        score_rows=[event.as_row() for event in score_events],
        prepared_segments=prepared_segments,
    )


def iter_prepared_segments(
    manifest_row,
    episode,
    score_events: list[ScoreEvent],
    segmenter: BaseSegmenter,
    config: dict[str, Any],
) -> Iterator[PreparedSegment]:
    from sonata.primitives.features import (
        build_feature_vector_from_arrays,
        build_gmr_target_from_arrays,
        build_safe_discovery_embedding_from_arrays,
        use_safe_discovery_embedding,
    )

    if episode.hand_joints is None:
        return
    feature_names: list[str] = []
    candidate_segments = segmenter.segment(episode, score_events)
    for index, candidate in enumerate(candidate_segments):
        arrays = slice_segment_arrays(episode, candidate.onset_step, candidate.end_step)
        hand_joints = arrays["hand_joints"]
        if hand_joints is None:
            continue
        velocity = arrays["joint_velocities"]
        if velocity is None:
            velocity = np.gradient(hand_joints, episode.control_timestep, axis=0).astype(np.float32)
            arrays["joint_velocities"] = velocity
        context = score_context_from_roll(episode.goals if episode.goals is not None else episode.piano_states, candidate.onset_step)
        record = SegmentRecord(
            segment_id=f"{episode.episode_id}_segment_{index:06d}",
            song_id=episode.song_id,
            episode_id=episode.episode_id,
            onset_step=candidate.onset_step,
            end_step=candidate.end_step,
            duration_steps=max(candidate.end_step - candidate.onset_step, 1),
            segment_source=candidate.segment_source,
            score_event_id=candidate.score_event_id,
            key_signature=candidate.key_signature,
            chunk_path="",
            chunk_index=-1,
            heuristic_family=candidate.heuristic_family,
            coarse_family=candidate.coarse_family,
            control_phase=candidate.control_phase,
            phase_index=int(candidate.phase_index),
            phase_count=int(candidate.phase_count),
            event_duration_steps=int(candidate.event_duration_steps),
            motion_energy=float(np.linalg.norm(velocity, axis=1).mean()),
            chord_size=int(candidate.chord_size),
            key_center=float(candidate.key_center),
            start_state_norm=float(np.linalg.norm(hand_joints[0])),
            end_state_norm=float(np.linalg.norm(hand_joints[-1])),
            score_context_json=dumps_score_context(context),
            boundary_score_peak=float(candidate.boundary_score_peak),
            boundary_source=str(candidate.boundary_source),
            snapped_to_score_event=int(candidate.snapped_to_score_event),
            raw_segment_length=int(candidate.raw_segment_length),
            segment_filter_reason=str(candidate.segment_filter_reason),
        )
        row_payload = record.as_row() | {"split": manifest_row.split}
        if use_safe_discovery_embedding(config):
            feature_vector, names = build_safe_discovery_embedding_from_arrays(row=row_payload, arrays=arrays, config=config)
        else:
            feature_vector, names = build_feature_vector_from_arrays(row=row_payload, arrays=arrays, config=config)
        gmr_target, target_name = build_gmr_target_from_arrays(arrays=arrays, config=config)
        row_payload["gmr_target_name"] = target_name
        if not feature_names:
            feature_names = names
        elif feature_names != names:
            raise ValueError(f"Incompatible feature names within episode {episode.episode_id}")
        yield PreparedSegment(
            row=row_payload,
            feature_vector=feature_vector.astype(np.float32),
            feature_names=list(names),
            gmr_target=gmr_target.astype(np.float32),
            gmr_target_name=target_name,
            arrays=arrays,
            raw_bytes_estimate=estimate_segment_storage_bytes(arrays),
        )


def apply_segment_volume_caps(
    segments: list[PreparedSegment],
    *,
    song_id: str,
    song_totals: dict[str, int],
    max_per_song: int,
    global_remaining: list[int],
    logger: logging.Logger | None,
) -> list[PreparedSegment]:
    if not segments:
        return segments
    out: list[PreparedSegment] = []
    sid = str(song_id)
    for seg in segments:
        if global_remaining[0] <= 0:
            if logger is not None:
                logger.warning("max_total_segments reached; truncating episode %s", seg.row.get("episode_id"))
            break
        if int(song_totals.get(sid, 0)) >= int(max_per_song):
            if logger is not None:
                logger.warning("max_segments_per_song reached for song %s", sid)
            break
        song_totals[sid] = int(song_totals.get(sid, 0)) + 1
        global_remaining[0] -= 1
        out.append(seg)
    if logger is not None and len(out) < len(segments):
        logger.warning("Segment budget dropped %d/%d segments for song %s.", len(segments) - len(out), len(segments), sid)
    return out


def estimate_segment_storage_bytes(arrays: dict[str, np.ndarray | None]) -> int:
    return int(sum(array.nbytes for array in arrays.values() if array is not None))


def slice_segment_arrays(episode, start: int, end: int) -> dict[str, np.ndarray | None]:
    arrays: dict[str, np.ndarray | None] = {}
    for field_name in (
        "hand_joints",
        "joint_velocities",
        "piano_states",
        "goals",
        "actions",
        "hand_fingertips",
        "wrist_pose",
        "hand_pose",
    ):
        array = getattr(episode, field_name)
        arrays[field_name] = np.asarray(array[start:end], dtype=np.float32) if array is not None else None
    return arrays


def load_segment_arrays_from_bundle(bundle: Any, index: int) -> dict[str, np.ndarray | None]:
    arrays: dict[str, np.ndarray | None] = {}
    for field_name in (
        "hand_joints",
        "joint_velocities",
        "piano_states",
        "goals",
        "actions",
        "hand_fingertips",
        "wrist_pose",
        "hand_pose",
    ):
        arrays[field_name] = load_segment_array(bundle, field_name, index)
    return arrays


def migrate_existing_segment_chunks(
    segment_df: pd.DataFrame,
    segments_dir: Path,
    output_dir: Path,
    config: dict[str, Any],
    logger: logging.Logger | None = None,
) -> dict[str, int]:
    from sonata.primitives.features import build_feature_vector

    slim_paths = resolve_slim_cache_paths(output_dir, config)
    legacy_df = ensure_segment_index_columns(segment_df)
    legacy_df = legacy_df.loc[~legacy_df["chunk_path"].astype(str).map(lambda value: not value or value == "" or value == "nan" or value == "None")]
    legacy_df = legacy_df.loc[~legacy_df["chunk_path"].astype(str).map(is_slim_chunk_name)]
    migrated_chunks = 0
    deleted_raw_chunks = 0
    source_raw_bytes = 0
    if legacy_df.empty:
        return {"migrated_chunks": 0, "deleted_raw_chunks": 0, "source_raw_bytes": 0}
    grouped = legacy_df.groupby("chunk_path", sort=True)
    iterator = tqdm(grouped, total=grouped.ngroups, desc="Migrate segment chunks")
    for raw_chunk_name, rows in iterator:
        raw_chunk_path = segments_dir / str(raw_chunk_name)
        chunk_name = slim_chunk_name(chunk_index_from_name(raw_chunk_name))
        if slim_chunk_complete(slim_paths, chunk_name):
            if bool(config.get("delete_raw_chunks_after_migration", False)) and raw_chunk_path.exists():
                raw_chunk_path.unlink()
                deleted_raw_chunks += 1
            continue
        if not raw_chunk_path.exists():
            raise FileNotFoundError(f"Missing raw segment chunk for migration: {raw_chunk_path}")
        source_raw_bytes += int(raw_chunk_path.stat().st_size)
        bundle = np.load(raw_chunk_path, allow_pickle=True)
        ordered_rows = rows.sort_values("chunk_index", kind="stable")
        slim_rows: list[dict[str, Any]] = []
        feature_rows: list[np.ndarray] = []
        feature_names: list[str] = []
        gmr_targets: list[np.ndarray] = []
        target_names: list[str] = []
        for row in ordered_rows.itertuples(index=False):
            arrays = load_segment_arrays_from_bundle(bundle, int(row.chunk_index))
            feature_vector, names = build_feature_vector(row=row, bundle=bundle, config=config)
            gmr_target, target_name = build_gmr_target(arrays=arrays, config=config)
            if not feature_names:
                feature_names = names
            elif feature_names != names:
                raise ValueError(f"Incompatible feature names while migrating {raw_chunk_name}")
            updated = dict(row._asdict())
            updated["raw_chunk_path"] = str(updated.get("raw_chunk_path") or raw_chunk_name)
            updated["raw_chunk_index"] = int(updated.get("raw_chunk_index", -1))
            if updated["raw_chunk_index"] < 0:
                updated["raw_chunk_index"] = int(row.chunk_index)
            updated["chunk_path"] = ""
            updated["chunk_index"] = -1
            updated["gmr_target_name"] = target_name
            slim_rows.append(updated)
            feature_rows.append(feature_vector.astype(np.float32))
            gmr_targets.append(gmr_target.astype(np.float32))
            target_names.append(target_name)
        write_slim_chunk(
            paths=slim_paths,
            chunk_name=chunk_name,
            segment_rows=slim_rows,
            feature_matrix=np.stack(feature_rows, axis=0).astype(np.float32),
            feature_names=feature_names,
            gmr_targets=np.stack(gmr_targets, axis=0).astype(np.float32),
            target_names=target_names,
            source_raw_chunk=str(raw_chunk_name),
            migrated=True,
            feature_storage_dtype=str(config.get("slim_feature_dtype", "float32")),
        )
        migrated_chunks += 1
        if bool(config.get("delete_raw_chunks_after_migration", False)):
            raw_chunk_path.unlink()
            deleted_raw_chunks += 1
        if logger is not None:
            logger.info("Migrated %s -> %s", raw_chunk_name, chunk_name)
    return {
        "migrated_chunks": migrated_chunks,
        "deleted_raw_chunks": deleted_raw_chunks,
        "source_raw_bytes": int(source_raw_bytes),
    }


@dataclass
class OnlineSegmentWriter:
    output_dir: Path
    chunk_size: int
    start_chunk_index: int
    config: dict[str, Any]
    logger: logging.Logger | None = None

    def __post_init__(self) -> None:
        self.paths = resolve_slim_cache_paths(self.output_dir, self.config)
        self.chunk_index = int(self.start_chunk_index)
        self.buffer_rows: list[dict[str, Any]] = []
        self.buffer_features: list[np.ndarray] = []
        self.buffer_targets: list[np.ndarray] = []
        self.buffer_target_names: list[str] = []
        self.buffer_episodes: list[dict[str, Any]] = []
        self.feature_names: list[str] | None = None
        self.current_episode_song_id: str | None = None
        self.current_episode_id: str | None = None
        self.current_episode_segments = 0
        self.stats: dict[str, float] = {
            "episodes_processed": 0,
            "segments_written": 0,
            "feature_rows_written": 0,
            "bytes_written": 0,
            "estimated_raw_segment_bytes": 0,
        }

    def begin_episode(self, song_id: str, episode_id: str) -> None:
        if self.current_episode_id is not None:
            raise ValueError("Cannot begin a new episode before ending the current one.")
        self.current_episode_song_id = song_id
        self.current_episode_id = episode_id
        self.current_episode_segments = 0

    def append_segment(
        self,
        row: dict[str, Any],
        feature_vector: np.ndarray,
        feature_names: list[str],
        gmr_target: np.ndarray,
        target_name: str,
        raw_segment_bytes: int,
    ) -> None:
        if self.current_episode_id is None or self.current_episode_song_id is None:
            raise ValueError("OnlineSegmentWriter.append_segment requires begin_episode() first.")
        if self.feature_names is None:
            self.feature_names = list(feature_names)
        elif self.feature_names != list(feature_names):
            raise ValueError("Incompatible feature names encountered while writing the online Stage 1 store.")
        self.buffer_rows.append(dict(row))
        self.buffer_features.append(np.asarray(feature_vector, dtype=np.float32))
        self.buffer_targets.append(np.asarray(gmr_target, dtype=np.float32))
        self.buffer_target_names.append(str(target_name))
        self.current_episode_segments += 1
        self.stats["estimated_raw_segment_bytes"] += int(raw_segment_bytes)

    def end_episode(self) -> list[dict[str, Any]]:
        if self.current_episode_id is None or self.current_episode_song_id is None:
            raise ValueError("Cannot end an episode before begin_episode().")
        payloads: list[dict[str, Any]] = []
        self.buffer_episodes.append(
            _episode_progress_payload(
                song_id=self.current_episode_song_id,
                episode_id=self.current_episode_id,
                num_segments=self.current_episode_segments,
            )
        )
        self.stats["episodes_processed"] += 1
        self.current_episode_song_id = None
        self.current_episode_id = None
        self.current_episode_segments = 0
        if len(self.buffer_rows) >= self.chunk_size:
            payloads.extend(self.flush())
        return payloads

    def flush(self) -> list[dict[str, Any]]:
        if not self.buffer_rows:
            return []
        if self.feature_names is None:
            raise ValueError("Cannot flush the online Stage 1 store without feature names.")
        chunk_name = slim_chunk_name(self.chunk_index)
        write_slim_chunk(
            paths=self.paths,
            chunk_name=chunk_name,
            segment_rows=self.buffer_rows,
            feature_matrix=np.stack(self.buffer_features, axis=0).astype(np.float32),
            feature_names=self.feature_names,
            gmr_targets=np.stack(self.buffer_targets, axis=0).astype(np.float32),
            target_names=self.buffer_target_names,
            feature_storage_dtype=str(self.config.get("slim_feature_dtype", "float32")),
        )
        chunk_bytes = int(sum(path.stat().st_size for path in (
            self.paths.feature_dir / chunk_name,
            self.paths.gmr_target_dir / chunk_name,
            self.paths.index_dir / f"{Path(chunk_name).stem}.csv",
            self.paths.manifest_dir / f"{Path(chunk_name).stem}.json",
        ) if path.exists()))
        self.stats["segments_written"] += len(self.buffer_rows)
        self.stats["feature_rows_written"] += len(self.buffer_rows)
        self.stats["bytes_written"] += chunk_bytes
        soft_b = self.config.get("max_storage_bytes_soft")
        hard_b = self.config.get("max_storage_bytes_hard")
        if soft_b is not None or hard_b is not None:
            total_disk = estimate_dir_bytes(self.output_dir)
            storage_guard(
                bytes_written=total_disk,
                soft_limit=int(soft_b or hard_b or total_disk),
                hard_limit=int(hard_b or soft_b or total_disk),
                logger=self.logger,
            )
        if self.logger is not None:
            per_1k = float(self.stats["bytes_written"] * 1000.0 / max(int(self.stats["segments_written"]), 1))
            self.logger.info(
                "Online Stage 1 wrote %d episodes / %d segments / %d feature rows, %.2f MiB total, %.2f MiB per 1k segments.",
                int(self.stats["episodes_processed"]),
                int(self.stats["segments_written"]),
                int(self.stats["feature_rows_written"]),
                float(self.stats["bytes_written"]) / (1024.0 * 1024.0),
                per_1k / (1024.0 * 1024.0),
            )
        payloads = list(self.buffer_episodes)
        self.chunk_index += 1
        self.buffer_rows = []
        self.buffer_features = []
        self.buffer_targets = []
        self.buffer_target_names = []
        self.buffer_episodes = []
        return payloads


@dataclass
class SegmentChunkWriter:
    output_dir: Path
    chunk_size: int
    start_chunk_index: int = 0
    existing_chunk_files: list[str] | None = None

    def __post_init__(self) -> None:
        self.buffer_rows: list[dict[str, Any]] = []
        self.buffer_arrays: list[dict[str, np.ndarray | None]] = []
        self.chunk_index = int(self.start_chunk_index)
        self.chunk_files: list[str] = list(self.existing_chunk_files or [])

    def add(self, row: dict[str, Any], arrays: dict[str, np.ndarray | None]) -> list[dict[str, Any]]:
        self.buffer_rows.append(row)
        self.buffer_arrays.append(arrays)
        if len(self.buffer_rows) >= self.chunk_size:
            return self.flush()
        return []

    def flush(self) -> list[dict[str, Any]]:
        if not self.buffer_rows:
            return []

        chunk_name = f"segment_chunk_{self.chunk_index:05d}.npz"
        chunk_path = self.output_dir / chunk_name
        payload: dict[str, Any] = {
            "segment_ids": np.asarray([row["segment_id"] for row in self.buffer_rows], dtype=object)
        }

        for field_name in sorted({key for item in self.buffer_arrays for key in item}):
            stacked, lengths, available = stack_variable([item.get(field_name) for item in self.buffer_arrays])
            payload[field_name] = stacked
            payload[f"{field_name}_lengths"] = lengths
            payload[f"{field_name}_available"] = available

        _atomic_save_npz(chunk_path, **payload)

        flushed_rows: list[dict[str, Any]] = []
        for index, row in enumerate(self.buffer_rows):
            updated = dict(row)
            updated["chunk_path"] = chunk_name
            updated["chunk_index"] = index
            flushed_rows.append(updated)

        if chunk_name not in self.chunk_files:
            self.chunk_files.append(chunk_name)

        self.chunk_index += 1
        self.buffer_rows = []
        self.buffer_arrays = []
        return flushed_rows

    def write_episode(self, rows: list[dict[str, Any]], arrays: list[dict[str, np.ndarray | None]]) -> list[dict[str, Any]]:
        if self.buffer_rows or self.buffer_arrays:
            raise ValueError("SegmentChunkWriter.write_episode expects an empty buffer.")
        self.buffer_rows = [dict(row) for row in rows]
        self.buffer_arrays = list(arrays)
        return self.flush()


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _next_raw_chunk_index(segments_dir: Path) -> int:
    chunk_ids = [chunk_index_from_name(path.name) for path in segments_dir.glob("segment_chunk_*.npz")]
    return max(chunk_ids) + 1 if chunk_ids else 0


def stack_variable(arrays: list[np.ndarray | None]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    available = np.asarray([array is not None for array in arrays], dtype=bool)
    template = next((array for array in arrays if array is not None), None)
    if template is None:
        return np.zeros((len(arrays), 1, 1), dtype=np.float32), np.zeros((len(arrays),), dtype=np.int64), available
    max_length = max(array.shape[0] if array is not None else 0 for array in arrays)
    stacked = np.zeros((len(arrays), max_length) + template.shape[1:], dtype=template.dtype)
    lengths = np.zeros((len(arrays),), dtype=np.int64)
    for index, array in enumerate(arrays):
        if array is None:
            continue
        length = array.shape[0]
        stacked[index, :length] = array
        lengths[index] = length
    return stacked, lengths, available


def load_segment_array(bundle: Any, name: str, index: int) -> np.ndarray | None:
    available_key = f"{name}_available"
    lengths_key = f"{name}_lengths"
    if available_key in bundle and not bool(bundle[available_key][index]):
        return None
    if name not in bundle or lengths_key not in bundle:
        return None
    length = int(bundle[lengths_key][index])
    return np.asarray(bundle[name][index, :length], dtype=np.float32)
