from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from sonata.data.loading import load_episode_record
from sonata.data.score import dumps_score_context, infer_events_from_goal_roll, load_note_events, score_context_from_roll
from sonata.data.schema import ScoreEvent, SegmentRecord
from sonata.utils.io import save_npz, write_json, write_table


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
                    chord_size=0,
                    key_center=0.0,
                )
            )
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
                )
            )
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


def classify_interval_family(event: ScoreEvent) -> str:
    if event.chord_size >= 3:
        return "chord"
    if event.inter_onset_steps == 0:
        return "stacked"
    return "single"


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
        return NoteAlignedSegmenter(pre_steps=config["pre_steps"], post_steps=config["post_steps"])
    if strategy == "dtw_assisted":
        return DTWAssistedSegmenter(
            pre_steps=config["pre_steps"],
            post_steps=config["post_steps"],
            alignment_radius=config["alignment_radius"],
            template_window=config["dtw_template_window"],
        )
    raise ValueError(f"Unknown segmentation strategy: {strategy}")


def run_segmentation(manifest_df: pd.DataFrame, output_dir: Path, config: dict[str, Any]) -> dict[str, Path]:
    segments_dir = output_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)
    table_base = segments_dir / "segment_index"
    score_base = segments_dir / "score_events"
    manifest_path = segments_dir / "segment_manifest.json"
    if table_base.with_suffix(".csv").exists() and not bool(config.get("force", False)):
        return {"segment_table_base": table_base, "score_table_base": score_base, "manifest_path": manifest_path}

    segmenter = build_segmenter(config)
    writer = SegmentChunkWriter(output_dir=segments_dir, chunk_size=int(config["segment_chunk_size"]))
    segment_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []

    for row in tqdm(manifest_df.itertuples(index=False), total=len(manifest_df), desc="Segment episodes"):
        episode = load_episode_record(row._asdict())
        if episode.hand_joints is None:
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
        score_rows.extend([event.as_row() for event in score_events])
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
                motion_energy=float(np.linalg.norm(velocity, axis=1).mean()),
                chord_size=int(candidate.chord_size),
                key_center=float(candidate.key_center),
                start_state_norm=float(np.linalg.norm(hand_joints[0])),
                end_state_norm=float(np.linalg.norm(hand_joints[-1])),
                score_context_json=dumps_score_context(context),
            )
            row_updates = writer.add(record.as_row() | {"split": row.split}, arrays)
            segment_rows.extend(row_updates)

    segment_rows.extend(writer.flush())
    segment_df = pd.DataFrame(segment_rows)
    score_df = pd.DataFrame(score_rows).drop_duplicates(subset=["event_id"]).reset_index(drop=True)
    write_table(segment_df, table_base)
    write_table(score_df, score_base)
    write_json(
        {
            "num_segments": int(len(segment_df)),
            "num_score_events": int(len(score_df)),
            "chunk_files": writer.chunk_files,
            "segment_strategy": config["segmentation_strategy"],
        },
        manifest_path,
    )
    return {"segment_table_base": table_base, "score_table_base": score_base, "manifest_path": manifest_path}


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


@dataclass
class SegmentChunkWriter:
    output_dir: Path
    chunk_size: int

    def __post_init__(self) -> None:
        self.buffer_rows: list[dict[str, Any]] = []
        self.buffer_arrays: list[dict[str, np.ndarray | None]] = []
        self.chunk_index = 0
        self.chunk_files: list[str] = []

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
        payload: dict[str, Any] = {"segment_ids": np.asarray([row["segment_id"] for row in self.buffer_rows], dtype=object)}
        for field_name in sorted({key for item in self.buffer_arrays for key in item}):
            stacked, lengths, available = stack_variable([item.get(field_name) for item in self.buffer_arrays])
            payload[field_name] = stacked
            payload[f"{field_name}_lengths"] = lengths
            payload[f"{field_name}_available"] = available
        save_npz(chunk_path, **payload)
        flushed_rows: list[dict[str, Any]] = []
        for index, row in enumerate(self.buffer_rows):
            updated = dict(row)
            updated["chunk_path"] = chunk_name
            updated["chunk_index"] = index
            flushed_rows.append(updated)
        self.chunk_files.append(chunk_name)
        self.chunk_index += 1
        self.buffer_rows = []
        self.buffer_arrays = []
        return flushed_rows


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
