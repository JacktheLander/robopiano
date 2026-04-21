from __future__ import annotations

import numpy as np

from sonata.data.schema import EpisodeRecord
from sonata.primitives.segmenters import LearnedBoundarySafeSegmenter, apply_segment_volume_caps, PreparedSegment


def _episode(t: int = 200) -> EpisodeRecord:
    hj = np.random.randn(t, 16).astype(np.float32) * 0.1
    act = np.random.randn(t, 8).astype(np.float32) * 0.05
    goals = np.zeros((t, 88), dtype=np.float32)
    goals[t // 4 : t // 4 + 10, 0] = 1.0
    goals[t // 2 : t // 2 + 8, 1] = 1.0
    return EpisodeRecord(
        song_id="s1",
        episode_id="e1",
        split="train",
        note_path=None,
        control_timestep=0.05,
        actions=act,
        goals=goals,
        piano_states=None,
        hand_joints=hj,
        joint_velocities=None,
        hand_fingertips=None,
        wrist_pose=None,
        hand_pose=None,
    )


def test_learned_boundary_respects_max_segments_per_song() -> None:
    cfg = {
        "min_segment_length": 6,
        "max_segment_length": 40,
        "max_candidate_boundaries_per_episode": 48,
        "max_segments_per_song": 8,
        "boundary_snap_tolerance_steps": 3,
        "boundary_score_action": 1.0,
        "boundary_score_vel_change": 0.85,
        "boundary_score_accel_change": 0.65,
        "boundary_score_goal_onset": 1.15,
        "boundary_score_goal_release": 0.95,
        "boundary_score_piano_change": 0.75,
    }
    seg = LearnedBoundarySafeSegmenter(cfg)
    ep = _episode(300)
    out = seg.segment(ep, [])
    assert len(out) <= cfg["max_segments_per_song"]
    for c in out:
        assert c.end_step - c.onset_step <= cfg["max_segment_length"]
        assert c.end_step - c.onset_step >= 1


def test_apply_segment_volume_caps() -> None:
    rows = [{"segment_id": f"s{i}", "song_id": "s", "episode_id": "e", "split": "train"} for i in range(5)]
    feats = [np.zeros((4,), dtype=np.float32) for _ in rows]
    segs = [
        PreparedSegment(
            row=r,
            feature_vector=f,
            feature_names=["a"],
            gmr_target=np.zeros((2, 3), dtype=np.float32),
            gmr_target_name="actions",
            arrays={},
            raw_bytes_estimate=0,
        )
        for r, f in zip(rows, feats, strict=False)
    ]
    song_totals: dict[str, int] = {}
    rem = [3]
    capped = apply_segment_volume_caps(segs, song_id="s", song_totals=song_totals, max_per_song=10, global_remaining=rem, logger=None)
    assert len(capped) == 3
    assert rem[0] == 0


def test_storage_guard_raises() -> None:
    from sonata.primitives.guards import storage_guard

    try:
        storage_guard(bytes_written=100, soft_limit=50, hard_limit=80, logger=None)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected hard stop")
