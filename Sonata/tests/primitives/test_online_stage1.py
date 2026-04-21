from __future__ import annotations

import sys
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sonata.primitives.features import (
    build_feature_vector_from_arrays,
    build_gmr_target_from_arrays,
    extract_segment_features,
    load_feature_matrix_from_store,
)
from sonata.primitives.segmenters import EventPhaseSegmenter, NoteAlignedSegmenter
from sonata.primitives.slim_cache import (
    collect_slim_chunk_names,
    index_chunk_path,
    list_incomplete_slim_chunks,
    load_slim_index_table,
    resolve_slim_cache_paths,
    write_slim_chunk,
)
from sonata.data.schema import ScoreEvent


def _segment_row(segment_id: str, song_id: str, episode_id: str) -> dict[str, object]:
    return {
        "segment_id": segment_id,
        "song_id": song_id,
        "episode_id": episode_id,
        "split": "train",
        "onset_step": 0,
        "end_step": 8,
        "duration_steps": 8,
        "motion_energy": 1.0,
        "chord_size": 1,
        "key_center": 0.0,
        "start_state_norm": 0.0,
        "end_state_norm": 0.0,
        "score_context_json": "{}",
    }


def test_load_feature_matrix_from_store_preserves_segment_index_order(tmp_path: Path) -> None:
    config = {"slim_cache_dir": "slim"}
    paths = resolve_slim_cache_paths(tmp_path, config)
    feature_names = ["feature_0", "feature_1"]
    gmr_targets = np.zeros((2, 4, 3), dtype=np.float32)
    write_slim_chunk(
        paths=paths,
        chunk_name="slim_chunk_00000.npz",
        segment_rows=[
            _segment_row("seg_a", "song_a", "ep_a"),
            _segment_row("seg_b", "song_a", "ep_a"),
        ],
        feature_matrix=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        feature_names=feature_names,
        gmr_targets=gmr_targets,
        target_names=["actions", "actions"],
    )
    write_slim_chunk(
        paths=paths,
        chunk_name="slim_chunk_00001.npz",
        segment_rows=[_segment_row("seg_c", "song_b", "ep_b")],
        feature_matrix=np.asarray([[5.0, 6.0]], dtype=np.float32),
        feature_names=feature_names,
        gmr_targets=np.zeros((1, 4, 3), dtype=np.float32),
        target_names=["actions"],
    )

    segment_df = pd.DataFrame(
        [
            {"segment_id": "seg_c", "chunk_path": "slim_chunk_00001.npz", "chunk_index": 0},
            {"segment_id": "seg_a", "chunk_path": "slim_chunk_00000.npz", "chunk_index": 0},
            {"segment_id": "seg_b", "chunk_path": "slim_chunk_00000.npz", "chunk_index": 1},
        ]
    )
    feature_matrix, loaded_names = load_feature_matrix_from_store(segment_df=segment_df, output_dir=tmp_path, config=config)

    assert loaded_names == feature_names
    np.testing.assert_allclose(
        feature_matrix,
        np.asarray([[5.0, 6.0], [1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    )


def test_load_slim_index_table_ignores_incomplete_chunks(tmp_path: Path) -> None:
    config = {"slim_cache_dir": "slim"}
    paths = resolve_slim_cache_paths(tmp_path, config)
    write_slim_chunk(
        paths=paths,
        chunk_name="slim_chunk_00000.npz",
        segment_rows=[_segment_row("seg_a", "song_a", "ep_a")],
        feature_matrix=np.asarray([[1.0, 2.0]], dtype=np.float32),
        feature_names=["feature_0", "feature_1"],
        gmr_targets=np.zeros((1, 4, 3), dtype=np.float32),
        target_names=["actions"],
    )
    incomplete_index = pd.DataFrame([{"segment_id": "seg_partial", "chunk_path": "slim_chunk_00001.npz", "chunk_index": 0}])
    incomplete_index.to_csv(index_chunk_path(paths, "slim_chunk_00001.npz"), index=False)

    all_chunks = collect_slim_chunk_names(paths, completed_only=False)
    complete_chunks = collect_slim_chunk_names(paths, completed_only=True)
    loaded_index = load_slim_index_table(paths)

    assert "slim_chunk_00001.npz" in all_chunks
    assert complete_chunks == ["slim_chunk_00000.npz"]
    assert list_incomplete_slim_chunks(paths) == ["slim_chunk_00001.npz"]
    assert loaded_index["segment_id"].tolist() == ["seg_a"]


def test_build_gmr_target_from_arrays_uses_fixed_resample_steps() -> None:
    actions = np.asarray([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32)
    target, target_name = build_gmr_target_from_arrays(
        arrays={"actions": actions, "hand_joints": None},
        config={"gmr_target_actions": True, "gmr_resample_steps": 5},
    )

    assert target_name == "actions"
    assert target.shape == (5, 3)
    np.testing.assert_allclose(target[0], actions[0])
    np.testing.assert_allclose(target[-1], actions[-1])


def test_extract_segment_features_reuses_slim_feature_store(tmp_path: Path) -> None:
    config = {"slim_cache_dir": "slim", "force": True}
    paths = resolve_slim_cache_paths(tmp_path, config)
    feature_names = ["feature_0", "feature_1"]
    write_slim_chunk(
        paths=paths,
        chunk_name="slim_chunk_00000.npz",
        segment_rows=[
            _segment_row("seg_a", "song_a", "ep_a"),
            _segment_row("seg_b", "song_a", "ep_a"),
        ],
        feature_matrix=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        feature_names=feature_names,
        gmr_targets=np.zeros((2, 4, 3), dtype=np.float32),
        target_names=["actions", "actions"],
    )
    segment_df = pd.DataFrame(
        [
            _segment_row("seg_b", "song_a", "ep_a") | {"chunk_path": "slim_chunk_00000.npz", "chunk_index": 1},
            _segment_row("seg_a", "song_a", "ep_a") | {"chunk_path": "slim_chunk_00000.npz", "chunk_index": 0},
        ]
    )

    outputs = extract_segment_features(
        segment_df=segment_df,
        segments_dir=tmp_path / "segments",
        output_dir=tmp_path,
        config=config,
    )

    bundle = np.load(outputs["feature_bundle_path"], allow_pickle=True)
    manifest = json.loads(Path(outputs["manifest_path"]).read_text())
    np.testing.assert_allclose(
        np.asarray(bundle["feature_matrix"], dtype=np.float32),
        np.asarray([[3.0, 4.0], [1.0, 2.0]], dtype=np.float32),
    )
    assert [str(item) for item in bundle["feature_names"].tolist()] == feature_names
    assert manifest["source"] == "slim_store"


def test_note_aligned_segmenter_caps_long_note_with_local_horizon() -> None:
    segmenter = NoteAlignedSegmenter(pre_steps=4, post_steps=4, note_local_horizon_steps=8)
    episode = SimpleNamespace(hand_joints=np.zeros((32, 4), dtype=np.float32))
    score_events = [
        ScoreEvent(
            event_id="event_0",
            song_id="song_a",
            episode_id="ep_a",
            onset_step=10,
            end_step=24,
            start_time_sec=0.5,
            end_time_sec=1.2,
            key_numbers=(40,),
            chord_size=1,
            key_center=0.5,
            inter_onset_steps=0,
            source="goals",
        )
    ]

    segments = segmenter.segment(episode, score_events)

    assert len(segments) == 1
    assert segments[0].onset_step == 6
    assert segments[0].end_step == 18


def test_build_feature_vector_from_arrays_includes_local_purity_signals() -> None:
    hand_joints = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [2.0, 0.0],
        ],
        dtype=np.float32,
    )
    goals = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    feature_vector, names = build_feature_vector_from_arrays(
        row={
            "segment_id": "seg_attack",
            "duration_steps": 4,
            "motion_energy": 1.0,
            "chord_size": 1,
            "key_center": 0.0,
            "start_state_norm": 0.0,
            "end_state_norm": 2.0,
            "score_context_json": "{}",
        },
        arrays={
            "hand_joints": hand_joints,
            "joint_velocities": np.asarray(
                [
                    [4.0, 0.0],
                    [4.0, 0.0],
                    [0.5, 0.0],
                    [0.0, 0.0],
                ],
                dtype=np.float32,
            ),
            "actions": None,
            "goals": goals,
            "piano_states": None,
        },
        config={"trajectory_resample_steps": 4, "fallback_action_dim": 2},
    )

    feature_map = {name: float(value) for name, value in zip(names, feature_vector.tolist())}

    assert "motion_frontload_ratio" in feature_map
    assert "contact_onset_frontload_ratio" in feature_map
    assert feature_map["motion_frontload_ratio"] > feature_map["motion_tail_ratio"]
    assert feature_map["contact_onset_frontload_ratio"] == 1.0
    assert feature_map["contact_release_density"] > 0.0
    assert feature_map["control_phase_press_onset"] == 0.0
    assert feature_map["control_phase_whole_event"] == 1.0


def test_event_phase_segmenter_emits_phase_specific_segments() -> None:
    segmenter = EventPhaseSegmenter(
        pre_steps=4,
        post_steps=6,
        note_local_horizon_steps=16,
        onset_window_steps_single=4,
        onset_window_steps_chord=6,
        approach_window_steps=4,
        release_window_steps=4,
        hold_min_duration_steps=6,
        transition_max_gap_steps=6,
        transition_window_steps=6,
        staccato_duration_steps=4,
        min_phase_duration_steps=2,
    )
    episode = SimpleNamespace(hand_joints=np.zeros((32, 4), dtype=np.float32))
    score_events = [
        ScoreEvent(
            event_id="event_0",
            song_id="song_a",
            episode_id="ep_a",
            onset_step=10,
            end_step=22,
            start_time_sec=0.5,
            end_time_sec=1.1,
            key_numbers=(40, 44, 47),
            chord_size=3,
            key_center=0.5,
            inter_onset_steps=4,
            source="goals",
        ),
        ScoreEvent(
            event_id="event_1",
            song_id="song_a",
            episode_id="ep_a",
            onset_step=25,
            end_step=29,
            start_time_sec=1.25,
            end_time_sec=1.45,
            key_numbers=(50,),
            chord_size=1,
            key_center=0.6,
            inter_onset_steps=3,
            source="goals",
        ),
    ]

    segments = segmenter.segment(episode, score_events)
    phases = {(segment.score_event_id, segment.control_phase) for segment in segments}

    assert ("event_0", "approach") in phases
    assert ("event_0", "press_onset") in phases
    assert ("event_0", "hold") in phases
    assert ("event_0", "release") in phases
    assert any(segment.coarse_family == "chord_press" for segment in segments if segment.score_event_id == "event_0")
    assert any(segment.control_phase == "local_transition" for segment in segments if segment.score_event_id == "event_0")
