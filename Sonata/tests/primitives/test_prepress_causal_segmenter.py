from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sonata.data.schema import EpisodeRecord  # noqa: E402
from sonata.primitives.segmenters import budget_segment_candidates, build_segmenter  # noqa: E402


def _episode(piano_states: np.ndarray) -> EpisodeRecord:
    steps = int(piano_states.shape[0])
    return EpisodeRecord(
        song_id="song",
        episode_id="episode",
        split="train",
        note_path=None,
        control_timestep=0.05,
        actions=np.ones((steps, 39), dtype=np.float32),
        goals=np.zeros_like(piano_states),
        piano_states=piano_states.astype(np.float32),
        hand_joints=np.ones((steps, 10), dtype=np.float32),
        joint_velocities=None,
        hand_fingertips=np.ones((steps, 30), dtype=np.float32),
        wrist_pose=None,
        hand_pose=None,
    )


def test_prepress_causal_segmenter_captures_pre_onset_motion() -> None:
    piano_states = np.zeros((24, 89), dtype=np.float32)
    piano_states[12:16, 10] = 1.0
    segmenter = build_segmenter(
        {
            "segmenter_name": "prepress_causal",
            "prepress_steps": 12,
            "post_onset_steps": 3,
            "min_inactive_pre_steps": 4,
            "min_hold_steps": 2,
            "activation_threshold": 0.5,
            "segment_min_len": 8,
            "segment_max_len": 20,
        }
    )

    segments = segmenter.segment(_episode(piano_states), [])

    assert len(segments) == 1
    segment = segments[0]
    assert segment.onset_step == 0
    assert segment.end_step == 15
    assert segment.target_onset_step == 12
    assert segment.key_signature == "10"
    assert segment.causal_segment is True
    assert segment.segment_alignment == "prepress_to_onset"
    assert segment.inactive_start is True
    assert segment.activation_after_start is True
    accepted, _ = budget_segment_candidates(
        segments,
        {
            "prepress_steps": 12,
            "post_onset_steps": 3,
            "segment_min_len": 8,
            "segment_max_len": 20,
            "segment_budget": {"enabled": True, "max_segments_per_score_onset": 1, "max_segments_per_target_signature": 1},
        },
    )
    assert accepted[0].causal_press_score > 0


def test_prepress_causal_segmenter_rejects_already_active_start() -> None:
    piano_states = np.zeros((24, 89), dtype=np.float32)
    piano_states[0:4, 10] = 1.0
    piano_states[12:16, 10] = 1.0
    segmenter = build_segmenter(
        {
            "segmenter_name": "prepress_causal",
            "prepress_steps": 12,
            "post_onset_steps": 3,
            "min_inactive_pre_steps": 4,
            "min_hold_steps": 2,
            "activation_threshold": 0.5,
            "segment_min_len": 8,
            "segment_max_len": 20,
        }
    )

    segments = segmenter.segment(_episode(piano_states), [])

    assert segments == []
    assert segmenter.last_stats["rejection_counts"]["target_active_at_segment_start"] == 1
