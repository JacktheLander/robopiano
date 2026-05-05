from __future__ import annotations

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
from sonata.primitives.parallel import iter_parallel_map, resolve_worker_count
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
from sonata.utils.io import read_table, save_npz, write_json, write_table


@dataclass
class CandidateSegment:
    onset_step: int
    end_step: int
    segment_source: str
    score_event_id: str
    key_signature: str
    heuristic_family: str
    chord_size: int
    key_center: float
    coarse_family: str = ""
    proposal_size: int = 1
    proposal_span_steps: int = 0
    boundary_energy: float = 0.0
    boundary_alignment_score: float = 0.0
    duplicate_iou: float = 0.0
    merge_count: int = 0
    split_count: int = 0
    target_key_count: int = 0
    target_key_signature: str = ""
    target_onset_step: int = -1
    next_onset_gap_steps: int = -1
    truncated_by_next_onset: bool = False
    causal_segment: bool = False
    segment_alignment: str = ""
    inactive_start: bool = False
    activation_after_start: bool = False
    contact_near_onset: bool = False
    rejection_reason: str = ""


@dataclass
class PreparedSegment:
    row: dict[str, Any]
    feature_vector: np.ndarray
    feature_names: list[str]
    gmr_target: np.ndarray
    gmr_target_name: str
    arrays: dict[str, np.ndarray | None] | None
    raw_bytes_estimate: int


@dataclass
class PreparedEpisodeBatch:
    song_id: str
    episode_id: str
    score_rows: list[dict[str, Any]]
    prepared_segments: list[PreparedSegment]
    stats: dict[str, Any]


class BaseSegmenter:
    name = "base"
    last_stats: dict[str, Any]

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
            self.last_stats = {"proposed_segments": 0, "accepted_segments": 0}
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
                    chord_size=0,
                    key_center=0.0,
                )
            )
        self.last_stats = {"proposed_segments": len(segments), "accepted_segments": len(segments)}
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
            self.last_stats = {"proposed_segments": 0, "accepted_segments": 0}
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
                    chord_size=0,
                    key_center=0.0,
                )
            )
        self.last_stats = {"proposed_segments": len(segments), "accepted_segments": len(segments)}
        return segments


class NoteAlignedSegmenter(BaseSegmenter):
    name = "note_aligned"

    def __init__(self, pre_steps: int, post_steps: int):
        self.pre_steps = int(pre_steps)
        self.post_steps = int(post_steps)

    def segment(self, episode, score_events: list[ScoreEvent]) -> list[CandidateSegment]:
        segments: list[CandidateSegment] = []
        for event in score_events:
            start = max(event.onset_step - self.pre_steps, 0)
            end = max(start + 1, min(event.end_step + self.post_steps, episode.hand_joints.shape[0]))
            family = classify_interval_family(event)
            segments.append(
                CandidateSegment(
                    onset_step=start,
                    end_step=end,
                    segment_source=self.name,
                    score_event_id=event.event_id,
                    key_signature="-".join(str(item) for item in event.key_numbers),
                    heuristic_family=family,
                    chord_size=event.chord_size,
                    key_center=event.key_center,
                    coarse_family=_heuristic_to_coarse_family(family=family, chord_size=event.chord_size),
                    proposal_size=1,
                    proposal_span_steps=max(int(event.end_step) - int(event.onset_step), 1),
                )
            )
        self.last_stats = {"proposed_segments": len(segments), "accepted_segments": len(segments)}
        return segments


class DTWAssistedSegmenter(NoteAlignedSegmenter):
    name = "dtw_assisted"

    def __init__(self, pre_steps: int, post_steps: int, alignment_radius: int, template_window: int):
        super().__init__(pre_steps=pre_steps, post_steps=post_steps)
        self.alignment_radius = int(alignment_radius)
        self.template_window = int(template_window)

    def segment(self, episode, score_events: list[ScoreEvent]) -> list[CandidateSegment]:
        base_segments = super().segment(episode, score_events)
        if episode.hand_joints is None or not base_segments:
            self.last_stats = {"proposed_segments": len(base_segments), "accepted_segments": len(base_segments)}
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
                    chord_size=segment.chord_size,
                    key_center=segment.key_center,
                    coarse_family=segment.coarse_family,
                    proposal_size=segment.proposal_size,
                    proposal_span_steps=segment.proposal_span_steps,
                    boundary_energy=segment.boundary_energy,
                    boundary_alignment_score=segment.boundary_alignment_score,
                    duplicate_iou=segment.duplicate_iou,
                    merge_count=segment.merge_count,
                    split_count=segment.split_count,
                )
            )
        self.last_stats = {"proposed_segments": len(base_segments), "accepted_segments": len(aligned)}
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


def classify_interval_family(event: ScoreEvent) -> str:
    if event.chord_size >= 3:
        return "chord"
    if event.inter_onset_steps == 0:
        return "stacked"
    return "single"


def _heuristic_to_coarse_family(*, family: str, chord_size: int) -> str:
    if family == "chord":
        return "chord_press"
    if family == "stacked":
        return "chord_press" if int(chord_size) > 1 else "repeat_press"
    if family in {"changepoint", "window"}:
        return "reposition"
    return "single_press"


def _coarse_to_heuristic_family(*, coarse_family: str, chord_size: int) -> str:
    if coarse_family in {"dyad_press", "triad_press", "chord_press"}:
        return "chord" if int(chord_size) >= 3 else "stacked"
    if coarse_family in {"single_press", "repeat_press"}:
        return "single"
    if coarse_family == "reposition":
        return "changepoint"
    return "window"


def _keyset_coarse_family(chord_size: int) -> str:
    if int(chord_size) <= 1:
        return "single_press"
    if int(chord_size) == 2:
        return "dyad_press"
    if int(chord_size) == 3:
        return "triad_press"
    return "chord_press"


class KeysetOnsetSegmenter(BaseSegmenter):
    name = "keyset_onset"

    def __init__(
        self,
        *,
        pre_steps: int,
        post_steps: int,
        min_len: int,
        max_len: int,
        chord_tolerance_steps: int,
        truncate_at_next_onset: bool,
    ):
        self.pre_steps = max(int(pre_steps), 0)
        self.post_steps = max(int(post_steps), 1)
        self.min_len = max(int(min_len), 1)
        self.max_len = max(int(max_len), self.min_len)
        self.chord_tolerance_steps = max(int(chord_tolerance_steps), 0)
        self.truncate_at_next_onset = bool(truncate_at_next_onset)
        self.last_stats = {"proposed_segments": 0, "accepted_segments": 0}

    def segment(self, episode, score_events: list[ScoreEvent]) -> list[CandidateSegment]:
        if episode.hand_joints is None or not score_events:
            self.last_stats = {"proposed_segments": 0, "accepted_segments": 0}
            return []
        boundary_signal, _, _ = _build_boundary_signal(episode)
        groups = self._build_onset_groups(score_events)
        segments = [
            self._candidate_from_group(
                episode=episode,
                group=group,
                next_group=groups[index + 1] if index + 1 < len(groups) else None,
                boundary_signal=boundary_signal,
            )
            for index, group in enumerate(groups)
        ]
        self.last_stats = {
            "proposed_segments": int(len(groups)),
            "accepted_segments": int(len(segments)),
            "segments_per_score_onset": 1.0 if groups else 0.0,
        }
        return segments

    def _build_onset_groups(self, score_events: list[ScoreEvent]) -> list[list[ScoreEvent]]:
        groups: list[list[ScoreEvent]] = []
        current: list[ScoreEvent] = []
        current_onset = 0
        for event in sorted(score_events, key=lambda item: (item.onset_step, item.end_step, item.event_id)):
            if not current:
                current = [event]
                current_onset = int(event.onset_step)
                continue
            if abs(int(event.onset_step) - current_onset) <= self.chord_tolerance_steps:
                current.append(event)
            else:
                groups.append(current)
                current = [event]
                current_onset = int(event.onset_step)
        if current:
            groups.append(current)
        return groups

    def _candidate_from_group(
        self,
        *,
        episode,
        group: list[ScoreEvent],
        next_group: list[ScoreEvent] | None,
        boundary_signal: np.ndarray,
    ) -> CandidateSegment:
        onset_step = int(min(event.onset_step for event in group))
        next_onset = int(min(event.onset_step for event in next_group)) if next_group else None
        start = max(onset_step - self.pre_steps, 0)
        end_cap = min(onset_step + self.post_steps, episode.hand_joints.shape[0])
        if self.truncate_at_next_onset and next_onset is not None:
            end_cap = min(end_cap, max(next_onset, onset_step + 1))
        end = min(max(start + self.min_len, end_cap), start + self.max_len, episode.hand_joints.shape[0])
        if self.truncate_at_next_onset and next_onset is not None and end > next_onset:
            end = max(min(next_onset, episode.hand_joints.shape[0]), onset_step + 1)
            start = max(min(start, end - self.min_len), 0)
        if end <= start:
            end = min(start + 1, episode.hand_joints.shape[0])

        unique_keys = sorted({int(key) for event in group for key in event.key_numbers})
        key_signature = "-".join(str(key) for key in unique_keys) if unique_keys else "none"
        chord_size = len(unique_keys)
        boundary_index = int(np.clip(onset_step, 0, max(boundary_signal.shape[0] - 1, 0))) if boundary_signal.size else 0
        boundary_energy = float(boundary_signal[boundary_index]) if boundary_signal.size else 0.0
        next_gap = int(next_onset - onset_step) if next_onset is not None else -1
        score_event_id = "|".join(str(event.event_id) for event in group)
        coarse_family = _keyset_coarse_family(chord_size)
        return CandidateSegment(
            onset_step=int(start),
            end_step=int(end),
            segment_source=self.name,
            score_event_id=score_event_id,
            key_signature=key_signature,
            heuristic_family=_coarse_to_heuristic_family(coarse_family=coarse_family, chord_size=chord_size),
            chord_size=int(chord_size),
            key_center=float(np.mean(unique_keys) / 87.0) if unique_keys else 0.0,
            coarse_family=coarse_family,
            proposal_size=len(group),
            proposal_span_steps=max(int(max(event.end_step for event in group)) - onset_step, 1),
            boundary_energy=boundary_energy,
            boundary_alignment_score=boundary_energy,
            target_key_count=int(chord_size),
            target_key_signature=key_signature,
            target_onset_step=int(onset_step),
            next_onset_gap_steps=next_gap,
            truncated_by_next_onset=bool(next_onset is not None and end_cap >= next_onset),
        )


class NoteGroupRefinedSegmenter(BaseSegmenter):
    name = "note_group_refined"

    def __init__(
        self,
        *,
        pre_steps: int,
        post_steps: int,
        grouping_window: int,
        boundary_refine_radius: int,
        min_len: int,
        max_len: int,
        duplicate_iou_threshold: float,
        merge_enabled: bool,
        split_enabled: bool,
        chord_tolerance_steps: int,
        group_max_events: int,
        group_max_span_steps: int,
        key_center_tolerance: float,
        repeat_window_steps: int,
    ):
        self.pre_steps = int(pre_steps)
        self.post_steps = int(post_steps)
        self.grouping_window = int(grouping_window)
        self.boundary_refine_radius = int(boundary_refine_radius)
        self.min_len = max(int(min_len), 2)
        self.max_len = max(int(max_len), self.min_len + 1)
        self.duplicate_iou_threshold = float(duplicate_iou_threshold)
        self.merge_enabled = bool(merge_enabled)
        self.split_enabled = bool(split_enabled)
        self.chord_tolerance_steps = int(chord_tolerance_steps)
        self.group_max_events = max(int(group_max_events), 1)
        self.group_max_span_steps = max(int(group_max_span_steps), self.max_len)
        self.key_center_tolerance = float(key_center_tolerance)
        self.repeat_window_steps = max(int(repeat_window_steps), 1)
        self.last_stats = {"proposed_segments": 0, "accepted_segments": 0}

    def segment(self, episode, score_events: list[ScoreEvent]) -> list[CandidateSegment]:
        if episode.hand_joints is None or not score_events:
            self.last_stats = {"proposed_segments": 0, "accepted_segments": 0}
            return []

        boundary_signal, contact_roll, pedal_trace = _build_boundary_signal(episode)
        groups = self._build_groups(score_events)
        initial = [
            self._candidate_from_group(
                episode=episode,
                group=group,
                boundary_signal=boundary_signal,
                contact_roll=contact_roll,
                pedal_trace=pedal_trace,
            )
            for group in groups
        ]
        cleaned, cleanup_stats = self._cleanup_candidates(initial=initial, boundary_signal=boundary_signal)
        self.last_stats = {
            "proposed_segments": int(len(initial)),
            "accepted_segments": int(len(cleaned)),
            **cleanup_stats,
        }
        return cleaned

    def _build_groups(self, score_events: list[ScoreEvent]) -> list[list[ScoreEvent]]:
        groups: list[list[ScoreEvent]] = []
        current: list[ScoreEvent] = []
        current_keys: set[int] = set()
        current_start = 0
        current_end = 0
        for event in sorted(score_events, key=lambda item: (item.onset_step, item.end_step, item.event_id)):
            if not current:
                current = [event]
                current_keys = set(int(key) for key in event.key_numbers)
                current_start = int(event.onset_step)
                current_end = int(event.end_step)
                continue
            onset_gap = int(event.onset_step) - int(current[-1].onset_step)
            span = int(event.end_step) - current_start
            repeated_key = bool(current_keys.intersection(int(key) for key in event.key_numbers))
            nearby_key = abs(float(event.key_center) - float(np.mean([item.key_center for item in current]))) <= self.key_center_tolerance
            should_group = (
                onset_gap <= self.grouping_window
                and span <= self.group_max_span_steps
                and len(current) < self.group_max_events
                and (
                    onset_gap <= self.chord_tolerance_steps
                    or repeated_key
                    or nearby_key
                    or int(event.chord_size) > 1
                    or any(int(item.chord_size) > 1 for item in current)
                )
            )
            if should_group:
                current.append(event)
                current_keys.update(int(key) for key in event.key_numbers)
                current_end = max(current_end, int(event.end_step))
            else:
                groups.append(current)
                current = [event]
                current_keys = set(int(key) for key in event.key_numbers)
                current_start = int(event.onset_step)
                current_end = int(event.end_step)
        if current:
            groups.append(current)
        return groups

    def _candidate_from_group(
        self,
        *,
        episode,
        group: list[ScoreEvent],
        boundary_signal: np.ndarray,
        contact_roll: np.ndarray,
        pedal_trace: np.ndarray,
    ) -> CandidateSegment:
        first = group[0]
        last = group[-1]
        start = max(int(first.onset_step) - self.pre_steps, 0)
        end = min(int(last.end_step) + self.post_steps, episode.hand_joints.shape[0])
        start, end, boundary_energy, alignment_score = _refine_boundaries(
            boundary_signal=boundary_signal,
            start=start,
            end=end,
            radius=self.boundary_refine_radius,
            min_len=self.min_len,
        )
        unique_keys = sorted({int(key) for event in group for key in event.key_numbers})
        chord_size = len(unique_keys)
        coarse_family = _infer_group_family(
            group=group,
            start=start,
            end=end,
            contact_roll=contact_roll,
            pedal_trace=pedal_trace,
            repeat_window_steps=self.repeat_window_steps,
        )
        score_event_id = first.event_id if len(group) == 1 else f"{first.event_id}|{last.event_id}"
        key_signature = "-".join(str(key) for key in unique_keys) if unique_keys else "none"
        return CandidateSegment(
            onset_step=int(start),
            end_step=int(end),
            segment_source=self.name,
            score_event_id=score_event_id,
            key_signature=key_signature,
            heuristic_family=_coarse_to_heuristic_family(coarse_family=coarse_family, chord_size=chord_size),
            coarse_family=coarse_family,
            chord_size=int(chord_size),
            key_center=float(np.mean(unique_keys) / 87.0) if unique_keys else float(np.mean([event.key_center for event in group])),
            proposal_size=len(group),
            proposal_span_steps=max(int(last.end_step) - int(first.onset_step), 1),
            boundary_energy=float(boundary_energy),
            boundary_alignment_score=float(alignment_score),
        )

    def _cleanup_candidates(
        self,
        *,
        initial: list[CandidateSegment],
        boundary_signal: np.ndarray,
    ) -> tuple[list[CandidateSegment], dict[str, Any]]:
        ordered = sorted(initial, key=lambda item: (item.onset_step, item.end_step, item.score_event_id))
        merged_segments = 0
        split_segments = 0
        duplicate_segments_dropped = 0
        if self.merge_enabled:
            merged: list[CandidateSegment] = []
            for candidate in ordered:
                if not merged:
                    merged.append(candidate)
                    continue
                previous = merged[-1]
                if _should_merge_candidates(previous=previous, current=candidate, boundary_signal=boundary_signal, max_len=self.max_len):
                    merged[-1] = _merge_candidates(previous=previous, current=candidate)
                    merged_segments += 1
                else:
                    merged.append(candidate)
            ordered = merged
        if self.split_enabled:
            split_output: list[CandidateSegment] = []
            for candidate in ordered:
                parts = _split_candidate(candidate=candidate, boundary_signal=boundary_signal, min_len=self.min_len, max_len=self.max_len)
                split_segments += max(len(parts) - 1, 0)
                split_output.extend(parts)
            ordered = split_output
        deduped: list[CandidateSegment] = []
        for candidate in sorted(ordered, key=lambda item: (item.onset_step, item.end_step, item.score_event_id)):
            if not deduped:
                deduped.append(candidate)
                continue
            previous = deduped[-1]
            iou = _segment_iou(previous, candidate)
            same_context = previous.coarse_family == candidate.coarse_family or previous.key_signature == candidate.key_signature
            if same_context and iou >= self.duplicate_iou_threshold:
                keep_current = float(candidate.boundary_alignment_score) >= float(previous.boundary_alignment_score)
                survivor = candidate if keep_current else previous
                survivor = CandidateSegment(
                    onset_step=survivor.onset_step,
                    end_step=survivor.end_step,
                    segment_source=survivor.segment_source,
                    score_event_id=survivor.score_event_id,
                    key_signature=survivor.key_signature,
                    heuristic_family=survivor.heuristic_family,
                    chord_size=survivor.chord_size,
                    key_center=survivor.key_center,
                    coarse_family=survivor.coarse_family,
                    proposal_size=survivor.proposal_size,
                    proposal_span_steps=survivor.proposal_span_steps,
                    boundary_energy=survivor.boundary_energy,
                    boundary_alignment_score=survivor.boundary_alignment_score,
                    duplicate_iou=max(float(previous.duplicate_iou), float(candidate.duplicate_iou), float(iou)),
                    merge_count=survivor.merge_count,
                    split_count=survivor.split_count,
                    target_key_count=survivor.target_key_count,
                    target_key_signature=survivor.target_key_signature,
                    target_onset_step=survivor.target_onset_step,
                    next_onset_gap_steps=survivor.next_onset_gap_steps,
                    truncated_by_next_onset=survivor.truncated_by_next_onset,
                )
                deduped[-1] = survivor
                duplicate_segments_dropped += 1
            else:
                deduped.append(candidate)
        return deduped, {
            "merged_segments": int(merged_segments),
            "split_segments": int(split_segments),
            "duplicate_segments_dropped": int(duplicate_segments_dropped),
        }


def _build_boundary_signal(episode) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hand_joints = np.asarray(episode.hand_joints, dtype=np.float32)
    contact_roll_source = episode.goals if episode.goals is not None else episode.piano_states
    contact_roll = np.asarray(contact_roll_source[:, :88], dtype=np.float32) if contact_roll_source is not None else np.zeros((hand_joints.shape[0], 88), dtype=np.float32)
    pedal_trace = np.asarray(contact_roll_source[:, 88:], dtype=np.float32) if contact_roll_source is not None and contact_roll_source.shape[1] > 88 else np.zeros((hand_joints.shape[0], 1), dtype=np.float32)

    signal = _normalized_trace(np.linalg.norm(np.gradient(hand_joints, axis=0), axis=1))
    signal += 0.35 * _normalized_trace(np.linalg.norm(np.gradient(np.gradient(hand_joints, axis=0), axis=0), axis=1))
    if episode.actions is not None:
        signal += 0.30 * _normalized_trace(_frame_delta_norm(np.asarray(episode.actions, dtype=np.float32)))
    if episode.hand_fingertips is not None:
        signal += 0.25 * _normalized_trace(np.linalg.norm(np.gradient(np.asarray(episode.hand_fingertips, dtype=np.float32), axis=0), axis=1))
    if episode.wrist_pose is not None:
        signal += 0.20 * _normalized_trace(_frame_delta_norm(np.asarray(episode.wrist_pose, dtype=np.float32)))
    signal += 0.20 * _normalized_trace(_contact_transition_trace(contact_roll))
    signal += 0.15 * _normalized_trace(_contact_transition_trace(pedal_trace))
    return signal.astype(np.float32), contact_roll, pedal_trace


def _frame_delta_norm(array: np.ndarray) -> np.ndarray:
    if array.shape[0] <= 1:
        return np.zeros((array.shape[0],), dtype=np.float32)
    delta = np.diff(array, axis=0, prepend=array[:1])
    return np.linalg.norm(delta, axis=1).astype(np.float32)


def _normalized_trace(trace: np.ndarray) -> np.ndarray:
    values = np.asarray(trace, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return values
    median = float(np.median(values))
    scale = float(np.quantile(np.abs(values - median), 0.75)) + 1e-6
    normalized = (values - median) / scale
    normalized = np.clip(normalized, -4.0, 8.0)
    normalized -= float(normalized.min())
    maximum = float(normalized.max())
    return normalized / maximum if maximum > 0 else np.zeros_like(normalized)


def _contact_transition_trace(roll: np.ndarray) -> np.ndarray:
    if roll.size == 0:
        return np.zeros((0,), dtype=np.float32)
    binary = np.asarray(roll > 0.5, dtype=np.float32)
    delta = np.abs(np.diff(binary, axis=0, prepend=binary[:1]))
    return delta.sum(axis=1).astype(np.float32) / max(binary.shape[1], 1)


def _refine_boundaries(
    *,
    boundary_signal: np.ndarray,
    start: int,
    end: int,
    radius: int,
    min_len: int,
) -> tuple[int, int, float, float]:
    if boundary_signal.size == 0:
        return int(start), int(end), 0.0, 0.0
    best_start, start_score = _search_boundary(boundary_signal=boundary_signal, anchor=start, radius=radius, direction="start")
    best_end, end_score = _search_boundary(boundary_signal=boundary_signal, anchor=end, radius=radius, direction="end")
    if best_end - best_start < int(min_len):
        best_end = min(max(best_start + int(min_len), end), boundary_signal.shape[0])
    boundary_energy = float(np.mean(boundary_signal[[min(best_start, boundary_signal.shape[0] - 1), min(max(best_end - 1, 0), boundary_signal.shape[0] - 1)]]))
    return int(best_start), int(best_end), boundary_energy, float((start_score + end_score) * 0.5)


def _search_boundary(*, boundary_signal: np.ndarray, anchor: int, radius: int, direction: str) -> tuple[int, float]:
    left = max(int(anchor) - int(radius), 0)
    right = min(int(anchor) + int(radius), boundary_signal.shape[0] - 1)
    best_index = int(np.clip(anchor, left, right))
    best_score = float("-inf")
    look = max(int(radius // 2), 1)
    for candidate in range(left, right + 1):
        current = float(boundary_signal[candidate])
        if direction == "start":
            future = boundary_signal[candidate : min(candidate + look + 1, boundary_signal.shape[0])]
            past = boundary_signal[max(candidate - look, 0) : candidate + 1]
        else:
            future = boundary_signal[max(candidate - look, 0) : candidate + 1]
            past = boundary_signal[candidate : min(candidate + look + 1, boundary_signal.shape[0])]
        future_peak = float(np.max(future)) if future.size else current
        past_mean = float(np.mean(past)) if past.size else current
        score = future_peak - current - 0.25 * past_mean
        if score > best_score:
            best_score = score
            best_index = candidate
    return int(best_index), float(best_score if np.isfinite(best_score) else 0.0)


def _infer_group_family(
    *,
    group: list[ScoreEvent],
    start: int,
    end: int,
    contact_roll: np.ndarray,
    pedal_trace: np.ndarray,
    repeat_window_steps: int,
) -> str:
    unique_keys = sorted({int(key) for event in group for key in event.key_numbers})
    chord_size = len(unique_keys)
    onset_step = int(group[0].onset_step)
    repeated_notes = len(unique_keys) < len(group)
    pedal_delta = 0.0
    if pedal_trace.size:
        left = max(start - 1, 0)
        right = min(max(end - 1, 0), pedal_trace.shape[0] - 1)
        pedal_delta = float(np.abs(pedal_trace[right] - pedal_trace[left]).sum())
    if pedal_delta > 0.2:
        return "pedal_transition"
    if chord_size >= 2:
        return "chord_press"
    if repeated_notes and (int(group[-1].onset_step) - onset_step) <= int(repeat_window_steps):
        return "repeat_press"
    if contact_roll.size:
        before = contact_roll[max(onset_step - 1, 0)]
        after = contact_roll[min(onset_step, contact_roll.shape[0] - 1)]
        active_delta = float((after > 0.5).sum() - (before > 0.5).sum())
        if active_delta < 0:
            return "release"
        if abs(active_delta) <= 0.0 and chord_size == 0:
            return "reposition"
    return "single_press" if chord_size <= 1 else "mixed_unknown"


def _should_merge_candidates(
    *,
    previous: CandidateSegment,
    current: CandidateSegment,
    boundary_signal: np.ndarray,
    max_len: int,
) -> bool:
    gap = int(current.onset_step) - int(previous.end_step)
    merged_len = int(current.end_step) - int(previous.onset_step)
    if merged_len > int(max_len):
        return False
    if gap > max(int(0.2 * max_len), 2):
        return False
    family_compatible = previous.coarse_family == current.coarse_family or previous.heuristic_family == current.heuristic_family
    key_compatible = abs(float(previous.key_center) - float(current.key_center)) <= 0.08
    if not (family_compatible and key_compatible):
        return False
    if boundary_signal.size == 0:
        return True
    boundary_index = min(max(previous.end_step, 0), boundary_signal.shape[0] - 1)
    return float(boundary_signal[boundary_index]) <= float(np.quantile(boundary_signal, 0.55))


def _merge_candidates(*, previous: CandidateSegment, current: CandidateSegment) -> CandidateSegment:
    chord_size = max(int(previous.chord_size), int(current.chord_size))
    key_signature = previous.key_signature if previous.key_signature == current.key_signature else f"{previous.key_signature}|{current.key_signature}"
    score_event_id = previous.score_event_id if previous.score_event_id == current.score_event_id else f"{previous.score_event_id}|{current.score_event_id}"
    return CandidateSegment(
        onset_step=min(int(previous.onset_step), int(current.onset_step)),
        end_step=max(int(previous.end_step), int(current.end_step)),
        segment_source=previous.segment_source,
        score_event_id=score_event_id,
        key_signature=key_signature,
        heuristic_family=previous.heuristic_family if previous.heuristic_family == current.heuristic_family else "window",
        chord_size=chord_size,
        key_center=float((float(previous.key_center) + float(current.key_center)) * 0.5),
        coarse_family=previous.coarse_family if previous.coarse_family == current.coarse_family else "mixed_unknown",
        proposal_size=int(previous.proposal_size) + int(current.proposal_size),
        proposal_span_steps=max(int(current.end_step) - int(previous.onset_step), 1),
        boundary_energy=float((float(previous.boundary_energy) + float(current.boundary_energy)) * 0.5),
        boundary_alignment_score=float(max(float(previous.boundary_alignment_score), float(current.boundary_alignment_score))),
        duplicate_iou=max(float(previous.duplicate_iou), float(current.duplicate_iou)),
        merge_count=int(previous.merge_count) + int(current.merge_count) + 1,
        split_count=int(previous.split_count) + int(current.split_count),
        target_key_count=max(int(previous.target_key_count), int(current.target_key_count)),
        target_key_signature=key_signature,
        target_onset_step=min(int(previous.target_onset_step), int(current.target_onset_step)),
        next_onset_gap_steps=int(current.next_onset_gap_steps),
        truncated_by_next_onset=bool(previous.truncated_by_next_onset or current.truncated_by_next_onset),
    )


def _split_candidate(
    *,
    candidate: CandidateSegment,
    boundary_signal: np.ndarray,
    min_len: int,
    max_len: int,
) -> list[CandidateSegment]:
    duration = int(candidate.end_step) - int(candidate.onset_step)
    if duration <= max_len or duration < max(int(min_len) * 2, 4):
        return [candidate]
    left = int(candidate.onset_step) + int(min_len)
    right = int(candidate.end_step) - int(min_len)
    if right <= left:
        return [candidate]
    midpoint = (left + right) // 2
    search_left = max(midpoint - int(min_len), left)
    search_right = min(midpoint + int(min_len), right)
    if boundary_signal.size:
        valley = int(search_left + np.argmin(boundary_signal[search_left:search_right])) if search_right > search_left else midpoint
    else:
        valley = midpoint
    if valley <= left or valley >= right:
        return [candidate]
    first = CandidateSegment(
        onset_step=int(candidate.onset_step),
        end_step=int(valley),
        segment_source=candidate.segment_source,
        score_event_id=candidate.score_event_id,
        key_signature=candidate.key_signature,
        heuristic_family=candidate.heuristic_family,
        chord_size=candidate.chord_size,
        key_center=candidate.key_center,
        coarse_family=candidate.coarse_family,
        proposal_size=max(int(candidate.proposal_size // 2), 1),
        proposal_span_steps=max(int(valley) - int(candidate.onset_step), 1),
        boundary_energy=candidate.boundary_energy,
        boundary_alignment_score=candidate.boundary_alignment_score,
        duplicate_iou=candidate.duplicate_iou,
        merge_count=candidate.merge_count,
        split_count=candidate.split_count + 1,
        target_key_count=candidate.target_key_count,
        target_key_signature=candidate.target_key_signature,
        target_onset_step=candidate.target_onset_step,
        next_onset_gap_steps=candidate.next_onset_gap_steps,
        truncated_by_next_onset=candidate.truncated_by_next_onset,
    )
    second = CandidateSegment(
        onset_step=int(valley),
        end_step=int(candidate.end_step),
        segment_source=candidate.segment_source,
        score_event_id=candidate.score_event_id,
        key_signature=candidate.key_signature,
        heuristic_family=candidate.heuristic_family,
        chord_size=candidate.chord_size,
        key_center=candidate.key_center,
        coarse_family=candidate.coarse_family,
        proposal_size=max(int(candidate.proposal_size - first.proposal_size), 1),
        proposal_span_steps=max(int(candidate.end_step) - int(valley), 1),
        boundary_energy=candidate.boundary_energy,
        boundary_alignment_score=candidate.boundary_alignment_score,
        duplicate_iou=candidate.duplicate_iou,
        merge_count=candidate.merge_count,
        split_count=candidate.split_count + 1,
        target_key_count=candidate.target_key_count,
        target_key_signature=candidate.target_key_signature,
        target_onset_step=candidate.target_onset_step,
        next_onset_gap_steps=candidate.next_onset_gap_steps,
        truncated_by_next_onset=candidate.truncated_by_next_onset,
    )
    return [first, second]


def _segment_iou(left: CandidateSegment, right: CandidateSegment) -> float:
    intersection = max(0, min(int(left.end_step), int(right.end_step)) - max(int(left.onset_step), int(right.onset_step)))
    union = max(int(left.end_step), int(right.end_step)) - min(int(left.onset_step), int(right.onset_step))
    return float(intersection / max(union, 1))


class PrepressCausalSegmenter(BaseSegmenter):
    name = "prepress_causal"

    def __init__(
        self,
        *,
        prepress_steps: int,
        post_onset_steps: int,
        min_inactive_pre_steps: int,
        min_hold_steps: int,
        activation_threshold: float,
        max_events_per_onset: int | None,
        require_contact_near_onset: bool,
        contact_tolerance_frames: int,
        fingertip_key_distance_threshold_m: float,
        segment_min_len: int,
        segment_max_len: int,
    ):
        self.prepress_steps = max(int(prepress_steps), 1)
        self.post_onset_steps = max(int(post_onset_steps), 1)
        self.min_inactive_pre_steps = max(int(min_inactive_pre_steps), 1)
        self.min_hold_steps = max(int(min_hold_steps), 1)
        self.activation_threshold = float(activation_threshold)
        self.max_events_per_onset = None if max_events_per_onset is None else max(int(max_events_per_onset), 1)
        self.require_contact_near_onset = bool(require_contact_near_onset)
        self.contact_tolerance_frames = max(int(contact_tolerance_frames), 0)
        self.fingertip_key_distance_threshold_m = float(fingertip_key_distance_threshold_m)
        self.segment_min_len = max(int(segment_min_len), 1)
        self.segment_max_len = max(int(segment_max_len), self.segment_min_len)
        self.last_stats = {"proposed_segments": 0, "accepted_segments": 0, "rejection_counts": {}}

    def segment(self, episode, score_events: list[ScoreEvent]) -> list[CandidateSegment]:
        del score_events
        roll_source = episode.piano_states if episode.piano_states is not None else episode.goals
        rejection_counts: dict[str, int] = {}
        if episode.hand_joints is None:
            self.last_stats = {"proposed_segments": 0, "accepted_segments": 0, "rejection_counts": {"missing_hand_joints": 1}}
            return []
        if episode.actions is None:
            self.last_stats = {"proposed_segments": 0, "accepted_segments": 0, "rejection_counts": {"missing_actions": 1}}
            return []
        if roll_source is None:
            self.last_stats = {"proposed_segments": 0, "accepted_segments": 0, "rejection_counts": {"missing_activation_roll": 1}}
            return []
        roll = np.asarray(roll_source, dtype=np.float32)
        if roll.ndim != 2 or roll.shape[0] == 0:
            self.last_stats = {"proposed_segments": 0, "accepted_segments": 0, "rejection_counts": {"invalid_activation_roll": 1}}
            return []
        key_roll = np.zeros((roll.shape[0], 88), dtype=bool)
        key_roll[:, : min(88, roll.shape[1])] = roll[:, : min(88, roll.shape[1])] >= self.activation_threshold
        onset_groups = self._activation_onset_groups(key_roll)
        proposed = len(onset_groups)
        accepted: list[CandidateSegment] = []
        for onset_step, keys in onset_groups:
            if self.max_events_per_onset is not None:
                keys = keys[: self.max_events_per_onset]
            candidate = self._build_candidate(
                episode=episode,
                key_roll=key_roll,
                onset_step=int(onset_step),
                keys=[int(key) for key in keys],
                rejection_counts=rejection_counts,
            )
            if candidate is not None:
                accepted.append(candidate)
        self.last_stats = {
            "proposed_segments": int(proposed),
            "accepted_segments": int(len(accepted)),
            "rejection_counts": dict(sorted(rejection_counts.items())),
        }
        return accepted

    def _activation_onset_groups(self, key_roll: np.ndarray) -> list[tuple[int, list[int]]]:
        groups: list[tuple[int, list[int]]] = []
        previous = key_roll[0]
        for frame_index in range(1, key_roll.shape[0]):
            current = key_roll[frame_index]
            onset_keys = np.flatnonzero(current & ~previous).astype(int).tolist()
            if onset_keys:
                groups.append((int(frame_index), onset_keys))
            previous = current
        return groups

    def _build_candidate(
        self,
        *,
        episode,
        key_roll: np.ndarray,
        onset_step: int,
        keys: list[int],
        rejection_counts: dict[str, int],
    ) -> CandidateSegment | None:
        start = int(onset_step) - self.prepress_steps
        end = int(onset_step) + self.post_onset_steps
        if start < 0:
            self._reject(rejection_counts, "insufficient_prepress_history")
            return None
        if end > key_roll.shape[0]:
            self._reject(rejection_counts, "insufficient_post_onset_window")
            return None
        duration = int(end) - int(start)
        if duration < self.segment_min_len or duration > self.segment_max_len:
            self._reject(rejection_counts, "duration_out_of_bounds")
            return None
        if episode.actions is None or np.asarray(episode.actions[start:end]).size == 0:
            self._reject(rejection_counts, "empty_action_segment")
            return None
        if np.any(key_roll[start, keys]):
            self._reject(rejection_counts, "target_active_at_segment_start")
            return None
        inactive_left = max(int(onset_step) - self.min_inactive_pre_steps, start)
        if np.any(key_roll[inactive_left:onset_step, :][:, keys]):
            self._reject(rejection_counts, "target_not_inactive_before_onset")
            return None
        if np.any(key_roll[start:onset_step, :][:, keys]):
            self._reject(rejection_counts, "target_active_in_prepress_window")
            return None
        hold_right = min(int(onset_step) + self.min_hold_steps, key_roll.shape[0])
        if hold_right - int(onset_step) < self.min_hold_steps or not np.all(key_roll[onset_step:hold_right, :][:, keys]):
            self._reject(rejection_counts, "min_hold_failed")
            return None
        contact_near_onset = self._contact_near_onset(episode=episode, keys=keys, onset_step=int(onset_step))
        if self.require_contact_near_onset and not contact_near_onset:
            self._reject(rejection_counts, "contact_near_onset_failed")
            return None
        key_signature = "-".join(str(key) for key in sorted(keys)) if keys else "none"
        chord_size = len(keys)
        return CandidateSegment(
            onset_step=int(start),
            end_step=int(end),
            segment_source=self.name,
            score_event_id=f"{episode.episode_id}_prepress_{int(onset_step):06d}_{key_signature}",
            key_signature=key_signature,
            heuristic_family="single" if chord_size <= 1 else "chord",
            chord_size=int(chord_size),
            key_center=float(np.mean(keys) / 87.0) if keys else 0.0,
            coarse_family=_keyset_coarse_family(chord_size),
            proposal_size=int(chord_size),
            proposal_span_steps=max(int(self.post_onset_steps), 1),
            target_key_count=int(chord_size),
            target_key_signature=key_signature,
            target_onset_step=int(onset_step),
            causal_segment=True,
            segment_alignment="prepress_to_onset",
            inactive_start=True,
            activation_after_start=True,
            contact_near_onset=bool(contact_near_onset),
        )

    def _contact_near_onset(self, *, episode, keys: list[int], onset_step: int) -> bool:
        if episode.hand_fingertips is None:
            return False
        # RP1M caches fingertip poses but not a stable piano-key position table. Keep this hook
        # explicit for future datasets instead of fabricating contact from piano state.
        key_positions = getattr(episode, "key_positions", None)
        if key_positions is None:
            return False
        tips = np.asarray(episode.hand_fingertips, dtype=np.float32).reshape(episode.hand_fingertips.shape[0], -1, 3)
        keys_xyz = np.asarray(key_positions, dtype=np.float32).reshape(-1, 3)
        left = max(int(onset_step) - self.contact_tolerance_frames, 0)
        right = min(int(onset_step) + self.contact_tolerance_frames + 1, tips.shape[0])
        if right <= left or not keys:
            return False
        distances = np.linalg.norm(tips[left:right, :, None, :] - keys_xyz[None, None, keys, :], axis=-1)
        return bool(np.any(distances < self.fingertip_key_distance_threshold_m))

    @staticmethod
    def _reject(rejection_counts: dict[str, int], reason: str) -> None:
        rejection_counts[str(reason)] = int(rejection_counts.get(str(reason), 0)) + 1


def build_segmenter(config: dict[str, Any]) -> BaseSegmenter:
    strategy = str(config.get("segmenter_name", config.get("segmentation_strategy", "note_aligned")))
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
        return NoteAlignedSegmenter(pre_steps=config["pre_steps"], post_steps=config["post_steps"])
    if strategy == "dtw_assisted":
        return DTWAssistedSegmenter(
            pre_steps=config["pre_steps"],
            post_steps=config["post_steps"],
            alignment_radius=config["alignment_radius"],
            template_window=config["dtw_template_window"],
        )
    if strategy == "keyset_onset":
        return KeysetOnsetSegmenter(
            pre_steps=int(config.get("segment_press_pre_steps", config.get("pre_steps", 2))),
            post_steps=int(config.get("segment_press_post_steps", config.get("post_steps", 6))),
            min_len=int(config.get("segment_min_len", 4)),
            max_len=int(config.get("segment_max_len", 12)),
            chord_tolerance_steps=int(config.get("chord_tolerance_steps", 1)),
            truncate_at_next_onset=bool(config.get("segment_truncate_at_next_onset", True)),
        )
    if strategy == "note_group_refined":
        return NoteGroupRefinedSegmenter(
            pre_steps=int(config.get("pre_steps", 6)),
            post_steps=int(config.get("post_steps", 12)),
            grouping_window=int(config.get("segment_grouping_window", config.get("chord_tolerance_steps", 2))),
            boundary_refine_radius=int(config.get("segment_boundary_refine_radius", 6)),
            min_len=int(config.get("segment_min_len", 4)),
            max_len=int(config.get("segment_max_len", 48)),
            duplicate_iou_threshold=float(config.get("segment_duplicate_iou_threshold", 0.85)),
            merge_enabled=bool(config.get("segment_merge_enabled", True)),
            split_enabled=bool(config.get("segment_split_enabled", True)),
            chord_tolerance_steps=int(config.get("chord_tolerance_steps", 1)),
            group_max_events=int(config.get("segment_group_max_events", 8)),
            group_max_span_steps=int(config.get("segment_group_max_span_steps", config.get("segment_max_len", 48))),
            key_center_tolerance=float(config.get("segment_group_key_center_tolerance", 0.12)),
            repeat_window_steps=int(config.get("segment_repeat_window_steps", 12)),
        )
    if strategy == "prepress_causal":
        return PrepressCausalSegmenter(
            prepress_steps=int(config.get("prepress_steps", 12)),
            post_onset_steps=int(config.get("post_onset_steps", 3)),
            min_inactive_pre_steps=int(config.get("min_inactive_pre_steps", 4)),
            min_hold_steps=int(config.get("min_hold_steps", 2)),
            activation_threshold=float(config.get("activation_threshold", 0.5)),
            max_events_per_onset=config.get("max_events_per_onset"),
            require_contact_near_onset=bool(config.get("require_contact_near_onset", False)),
            contact_tolerance_frames=int(config.get("contact_tolerance_frames", 2)),
            fingertip_key_distance_threshold_m=float(config.get("fingertip_key_distance_threshold_m", 0.025)),
            segment_min_len=int(config.get("segment_min_len", 8)),
            segment_max_len=int(config.get("segment_max_len", 20)),
        )
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
            "segment_strategy": str(config.get("segmenter_name", config.get("segmentation_strategy", ""))),
            "segment_config_signature": _segment_config_signature(config),
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


_SEGMENT_CONFIG_SIGNATURE_KEYS = (
    "segmenter_name",
    "segmentation_strategy",
    "pre_steps",
    "post_steps",
    "segment_press_pre_steps",
    "segment_press_post_steps",
    "segment_truncate_at_next_onset",
    "segment_min_len",
    "segment_max_len",
    "chord_tolerance_steps",
    "window_steps",
    "stride_steps",
    "min_gap_steps",
    "velocity_quantile",
    "acceleration_quantile",
    "alignment_radius",
    "dtw_template_window",
    "segment_grouping_window",
    "segment_boundary_refine_radius",
    "segment_duplicate_iou_threshold",
    "segment_merge_enabled",
    "segment_split_enabled",
    "segment_group_max_events",
    "segment_group_max_span_steps",
    "segment_group_key_center_tolerance",
    "segment_repeat_window_steps",
    "prepress_steps",
    "post_onset_steps",
    "min_inactive_pre_steps",
    "min_hold_steps",
    "activation_threshold",
    "max_events_per_onset",
    "require_contact_near_onset",
    "contact_tolerance_frames",
    "fingertip_key_distance_threshold_m",
    "segment_end_mode",
    "segment_start_mode",
)


def _segment_config_payload(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: config.get(key)
        for key in _SEGMENT_CONFIG_SIGNATURE_KEYS
        if key in config
    }


def _segment_config_signature(config: dict[str, Any]) -> str:
    return json.dumps(_segment_config_payload(config), sort_keys=True, separators=(",", ":"))


def _load_segment_manifest(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.exists():
        return {}
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _segment_cache_matches_config(manifest_path: Path, config: dict[str, Any]) -> bool:
    payload = _load_segment_manifest(manifest_path)
    if not payload:
        return True
    expected_strategy = str(config.get("segmenter_name", config.get("segmentation_strategy", "")))
    cached_strategy = str(payload.get("segment_strategy", ""))
    if cached_strategy and cached_strategy != expected_strategy:
        return False
    cached_signature = str(payload.get("segment_config_signature", ""))
    if cached_signature and cached_signature != _segment_config_signature(config):
        return False
    return True


def run_segmentation(
    manifest_df: pd.DataFrame,
    output_dir: Path,
    config: dict[str, Any],
    logger: logging.Logger | None = None,
    evaluator=None,
) -> dict[str, Path]:
    if not online_segment_processing_enabled(config):
        return run_segmentation_legacy(manifest_df=manifest_df, output_dir=output_dir, config=config)
    return run_segmentation_slim(manifest_df=manifest_df, output_dir=output_dir, config=config, logger=logger, evaluator=evaluator)


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
    reset_cache = force or not _segment_cache_matches_config(manifest_path, config)

    if reset_cache:
        for path in [segment_partial_csv, score_partial_csv, final_segment_csv, final_score_csv, manifest_path]:
            if path.exists():
                path.unlink()
        for path in segments_dir.glob("segment_chunk_*.npz"):
            path.unlink()

    if final_segment_csv.exists() and final_score_csv.exists() and not reset_cache:
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
                coarse_family=candidate.coarse_family or _heuristic_to_coarse_family(family=candidate.heuristic_family, chord_size=candidate.chord_size),
                motion_energy=float(np.linalg.norm(velocity, axis=1).mean()),
                chord_size=int(candidate.chord_size),
                key_center=float(candidate.key_center),
                start_state_norm=float(np.linalg.norm(hand_joints[0])),
                end_state_norm=float(np.linalg.norm(hand_joints[-1])),
                score_context_json=dumps_score_context(context),
                proposal_size=int(candidate.proposal_size),
                proposal_span_steps=int(candidate.proposal_span_steps),
                boundary_energy=float(candidate.boundary_energy),
                boundary_alignment_score=float(candidate.boundary_alignment_score),
                duplicate_iou=float(candidate.duplicate_iou),
                merge_count=int(candidate.merge_count),
                split_count=int(candidate.split_count),
                target_key_count=int(candidate.target_key_count or candidate.chord_size),
                target_key_signature=str(candidate.target_key_signature or candidate.key_signature),
                target_onset_step=int(candidate.target_onset_step if candidate.target_onset_step >= 0 else candidate.onset_step),
                next_onset_gap_steps=int(candidate.next_onset_gap_steps),
                truncated_by_next_onset=bool(candidate.truncated_by_next_onset),
                causal_segment=bool(candidate.causal_segment),
                segment_alignment=str(candidate.segment_alignment),
                inactive_start=bool(candidate.inactive_start),
                activation_after_start=bool(candidate.activation_after_start),
                contact_near_onset=bool(candidate.contact_near_onset),
                rejection_reason=str(candidate.rejection_reason),
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
    evaluator=None,
) -> dict[str, Path]:
    segments_dir = output_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)
    table_base = segments_dir / "segment_index"
    score_base = segments_dir / "score_events"
    manifest_path = segments_dir / "segment_manifest.json"
    slim_paths = resolve_slim_cache_paths(output_dir, config)
    compact_manifest_path = compact_store_manifest_path(slim_paths)

    reset_cache = bool(config.get("force", False)) or not _segment_cache_matches_config(manifest_path, config)
    if reset_cache:
        if logger is not None and not bool(config.get("force", False)):
            logger.warning(
                "Existing Stage 1 segmentation cache does not match the current segmenter config; rebuilding %s.",
                output_dir,
            )
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
    early_stop_decision = None
    loop_start_time = time.monotonic()
    log_interval = max(int(config.get("segment_log_interval_episodes", 32)), 1)
    processed_this_loop = 0
    segments_this_loop = 0
    aggregate_rejection_counts: dict[str, int] = {}
    if raw_writer is None:
        segment_num_workers = resolve_worker_count(config.get("segment_num_workers"), default=0)
        prepared_batches = _iter_prepared_episode_batches(
            manifest_df=remaining_df,
            config=config,
            max_workers=segment_num_workers,
        )
        for batch in tqdm(prepared_batches, total=len(remaining_df), desc="Segment episodes"):
            new_score_rows.extend(batch.score_rows)
            if batch.prepared_segments:
                slim_writer.begin_episode(song_id=batch.song_id, episode_id=batch.episode_id)
                for segment in batch.prepared_segments:
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
            processed_this_loop += 1
            segments_this_loop += len(batch.prepared_segments)
            _merge_rejection_counts(aggregate_rejection_counts, batch.stats.get("rejection_counts", {}))
            if evaluator is not None:
                early_stop_decision = evaluator.observe_segmentation(
                    [segment.row for segment in batch.prepared_segments],
                    batch.stats,
                )
                if early_stop_decision is not None and early_stop_decision.stop:
                    break
            if logger is not None and processed_this_loop % log_interval == 0:
                elapsed = max(time.monotonic() - loop_start_time, 1e-6)
                logger.info(
                    "Stage 1 segmentation progress: %d/%d episodes, %d segments, %.2f episodes/s, %.2f segments/s.",
                    processed_this_loop,
                    len(remaining_df),
                    segments_this_loop,
                    processed_this_loop / elapsed,
                    segments_this_loop / elapsed,
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
                    include_arrays=True,
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
            processed_this_loop += 1
            segments_this_loop += len(prepared_segments)
            _merge_rejection_counts(aggregate_rejection_counts, getattr(segmenter, "last_stats", {}).get("rejection_counts", {}))
            if evaluator is not None:
                early_stop_decision = evaluator.observe_segmentation(
                    [segment.row for segment in prepared_segments],
                    dict(getattr(segmenter, "last_stats", {})) | {"accepted_segments": len(prepared_segments)},
                )
                if early_stop_decision is not None and early_stop_decision.stop:
                    break
            if logger is not None and processed_this_loop % log_interval == 0:
                elapsed = max(time.monotonic() - loop_start_time, 1e-6)
                logger.info(
                    "Stage 1 segmentation progress: %d/%d episodes, %d segments, %.2f episodes/s, %.2f segments/s.",
                    processed_this_loop,
                    len(remaining_df),
                    segments_this_loop,
                    processed_this_loop / elapsed,
                    segments_this_loop / elapsed,
                )

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
    final_status = "early_stopped" if early_stop_decision is not None and early_stop_decision.stop else "completed"
    compact_manifest = {
        "status": final_status,
        "online_segment_processing": True,
        "save_raw_segment_chunks": save_raw_segment_chunks_enabled(config),
        "online_storage_format": resolve_online_storage_format(config),
        "segment_strategy": config.get("segmenter_name", config.get("segmentation_strategy", "note_aligned")),
        "segment_config_signature": _segment_config_signature(config),
        "rejection_counts": dict(sorted(aggregate_rejection_counts.items())),
        "num_rejected_segments": int(sum(aggregate_rejection_counts.values())),
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
            "status": final_status,
            "num_segments": int(len(segment_df)),
            "num_score_events": int(len(score_df)),
            "chunk_files": sorted(path.name for path in segments_dir.glob("segment_chunk_*.npz")),
            "slim_chunk_files": collect_slim_chunk_names(slim_paths, completed_only=True),
            "segment_strategy": config.get("segmenter_name", config.get("segmentation_strategy", "note_aligned")),
            "segment_config_signature": _segment_config_signature(config),
            "rejection_counts": dict(sorted(aggregate_rejection_counts.items())),
            "num_rejected_segments": int(sum(aggregate_rejection_counts.values())),
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


def _merge_rejection_counts(target: dict[str, int], source: Any) -> None:
    if not isinstance(source, dict):
        return
    for key, value in source.items():
        try:
            count = int(value)
        except (TypeError, ValueError):
            continue
        target[str(key)] = int(target.get(str(key), 0)) + count


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
            yield _prepare_episode_batch(payload, config, include_arrays=False)
        return

    jobs = [
        {
            "manifest_row_payload": payload,
            "config": config,
            "include_arrays": False,
        }
        for payload in payloads
    ]
    use_process_pool = bool(config.get("use_process_pool", True))
    start_method = str(config.get("process_start_method", "spawn"))
    yield from iter_parallel_map(
        _prepare_episode_batch_job,
        jobs,
        max_workers=max_workers,
        use_process_pool=use_process_pool,
        in_flight_multiplier=2,
        start_method=start_method,
    )


def _prepare_episode_batch_job(job_payload: dict[str, Any]) -> PreparedEpisodeBatch:
    return _prepare_episode_batch(
        job_payload["manifest_row_payload"],
        job_payload["config"],
        include_arrays=bool(job_payload.get("include_arrays", False)),
    )


def _prepare_episode_batch(
    manifest_row_payload: dict[str, Any],
    config: dict[str, Any],
    *,
    include_arrays: bool,
) -> PreparedEpisodeBatch:
    manifest_row = SimpleNamespace(**manifest_row_payload)
    episode = load_episode_record(manifest_row_payload)
    score_events = _load_or_infer_score_events(episode=episode, config=config)
    segmenter = build_segmenter(config)
    prepared_segments = list(
        iter_prepared_segments(
            manifest_row=manifest_row,
            episode=episode,
            score_events=score_events,
            segmenter=segmenter,
            config=config,
            include_arrays=include_arrays,
        )
    )
    stats = dict(getattr(segmenter, "last_stats", {}))
    stats["accepted_segments"] = int(len(prepared_segments))
    return PreparedEpisodeBatch(
        song_id=str(manifest_row.song_id),
        episode_id=str(manifest_row.episode_id),
        score_rows=[event.as_row() for event in score_events],
        prepared_segments=prepared_segments,
        stats=stats,
    )


def iter_prepared_segments(
    manifest_row,
    episode,
    score_events: list[ScoreEvent],
    segmenter: BaseSegmenter,
    config: dict[str, Any],
    *,
    include_arrays: bool,
) -> Iterator[PreparedSegment]:
    from sonata.primitives.features import build_feature_vector_from_arrays, build_gmr_target_from_arrays

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
            coarse_family=candidate.coarse_family or _heuristic_to_coarse_family(family=candidate.heuristic_family, chord_size=candidate.chord_size),
            motion_energy=float(np.linalg.norm(velocity, axis=1).mean()),
            chord_size=int(candidate.chord_size),
            key_center=float(candidate.key_center),
            start_state_norm=float(np.linalg.norm(hand_joints[0])),
            end_state_norm=float(np.linalg.norm(hand_joints[-1])),
            score_context_json=dumps_score_context(context),
            proposal_size=int(candidate.proposal_size),
            proposal_span_steps=int(candidate.proposal_span_steps),
            boundary_energy=float(candidate.boundary_energy),
            boundary_alignment_score=float(candidate.boundary_alignment_score),
            duplicate_iou=float(candidate.duplicate_iou),
            merge_count=int(candidate.merge_count),
            split_count=int(candidate.split_count),
            target_key_count=int(candidate.target_key_count or candidate.chord_size),
            target_key_signature=str(candidate.target_key_signature or candidate.key_signature),
            target_onset_step=int(candidate.target_onset_step if candidate.target_onset_step >= 0 else candidate.onset_step),
            next_onset_gap_steps=int(candidate.next_onset_gap_steps),
            truncated_by_next_onset=bool(candidate.truncated_by_next_onset),
            causal_segment=bool(candidate.causal_segment),
            segment_alignment=str(candidate.segment_alignment),
            inactive_start=bool(candidate.inactive_start),
            activation_after_start=bool(candidate.activation_after_start),
            contact_near_onset=bool(candidate.contact_near_onset),
            rejection_reason=str(candidate.rejection_reason),
        )
        row_payload = record.as_row() | {"split": manifest_row.split}
        feature_vector, names = build_feature_vector_from_arrays(row=row_payload, arrays=arrays, config=config)
        gmr_target, target_name = build_gmr_target_from_arrays(arrays=arrays, config=config)
        row_payload["gmr_target_name"] = target_name
        raw_bytes_estimate = estimate_segment_storage_bytes(arrays)
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
            arrays=arrays if include_arrays else None,
            raw_bytes_estimate=raw_bytes_estimate,
        )


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
