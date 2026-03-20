from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
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


@dataclass
class PreparedSegment:
    row: dict[str, Any]
    feature_vector: np.ndarray
    feature_names: list[str]
    gmr_target: np.ndarray
    gmr_target_name: str
    arrays: dict[str, np.ndarray | None]
    raw_bytes_estimate: int


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

def _atomic_save_npz(path: Path, **payload: Any) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    save_npz(tmp_path, **payload)
    os.replace(tmp_path, path)


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
                motion_energy=float(np.linalg.norm(velocity, axis=1).mean()),
                chord_size=int(candidate.chord_size),
                key_center=float(candidate.key_center),
                start_state_norm=float(np.linalg.norm(hand_joints[0])),
                end_state_norm=float(np.linalg.norm(hand_joints[-1])),
                score_context_json=dumps_score_context(context),
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
    skipped_episodes = 0
    for row in tqdm(manifest_df.itertuples(index=False), total=len(manifest_df), desc="Segment episodes"):
        episode_id = str(row.episode_id)
        if episode_id in processed_episode_ids:
            skipped_episodes += 1
            continue
        episode = load_episode_record(row._asdict())
        score_events = _load_or_infer_score_events(episode=episode, config=config)
        new_score_rows.extend([event.as_row() for event in score_events])
        if raw_writer is None:
            produced_segments = 0
            for segment in iter_prepared_segments(
                manifest_row=row,
                episode=episode,
                score_events=score_events,
                segmenter=segmenter,
                config=config,
            ):
                if produced_segments == 0:
                    slim_writer.begin_episode(song_id=str(row.song_id), episode_id=episode_id)
                slim_writer.append_segment(
                    row=segment.row,
                    feature_vector=segment.feature_vector,
                    feature_names=segment.feature_names,
                    gmr_target=segment.gmr_target,
                    target_name=segment.gmr_target_name,
                    raw_segment_bytes=segment.raw_bytes_estimate,
                )
                produced_segments += 1
            if produced_segments:
                _record_episode_progress(slim_paths, slim_writer.end_episode())
            else:
                append_episode_progress(
                    slim_paths,
                    _episode_progress_payload(song_id=str(row.song_id), episode_id=episode_id, num_segments=0),
                )
        else:
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


def iter_prepared_segments(
    manifest_row,
    episode,
    score_events: list[ScoreEvent],
    segmenter: BaseSegmenter,
    config: dict[str, Any],
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
            motion_energy=float(np.linalg.norm(velocity, axis=1).mean()),
            chord_size=int(candidate.chord_size),
            key_center=float(candidate.key_center),
            start_state_norm=float(np.linalg.norm(hand_joints[0])),
            end_state_norm=float(np.linalg.norm(hand_joints[-1])),
            score_context_json=dumps_score_context(context),
        )
        row_payload = record.as_row() | {"split": manifest_row.split}
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
