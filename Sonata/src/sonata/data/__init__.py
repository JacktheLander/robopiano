"""Dataset indexing, loading, and score parsing for Sonata-3."""

from sonata.data.indexer import scan_dataset
from sonata.data.loading import load_episode_record, load_manifest
from sonata.data.score import dumps_score_context, infer_events_from_goal_roll, load_note_events, score_context_from_roll
from sonata.data.schema import EpisodeRecord, ManifestRecord, ScoreEvent, SegmentRecord

__all__ = [
    "EpisodeRecord",
    "ManifestRecord",
    "ScoreEvent",
    "SegmentRecord",
    "dumps_score_context",
    "infer_events_from_goal_roll",
    "load_episode_record",
    "load_manifest",
    "load_note_events",
    "scan_dataset",
    "score_context_from_roll",
]
