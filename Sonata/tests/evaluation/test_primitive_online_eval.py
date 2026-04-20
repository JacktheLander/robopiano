from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sonata.evaluation.primitive_online_eval import (
    build_primitive_instances,
    extract_key_events_from_goals,
    extract_key_events_from_piano_states,
    load_primitive_online_artifacts,
    match_key_events,
    sample_primitive_assignment_rows,
)
from sonata.primitives.slim_cache import resolve_slim_cache_paths, write_slim_chunk
from sonata.utils.io import save_npz, write_table
from sonata.utils.robopianist import resolve_robopianist_import_root


def test_extract_key_events_from_goals_tracks_onsets_and_releases() -> None:
    goals = np.zeros((5, 89), dtype=np.float32)
    goals[0:2, 10] = 1.0
    goals[2:5, 12] = 1.0

    bundle = extract_key_events_from_goals(goals)

    assert [event.as_tuple() for event in bundle.events] == [(10, 0, 2), (12, 2, 5)]
    assert bundle.unique_keys == (10, 12)


def test_extract_key_events_from_piano_states_uses_threshold() -> None:
    piano_states = np.zeros((4, 89), dtype=np.float32)
    piano_states[0, 8] = 0.45
    piano_states[1:3, 8] = 0.8
    piano_states[2:, 20] = 0.9

    bundle = extract_key_events_from_piano_states(piano_states, key_threshold=0.5)

    assert [event.as_tuple() for event in bundle.events] == [(8, 1, 3), (20, 2, 4)]


def test_match_key_events_counts_false_positives_and_missed_notes() -> None:
    predicted = [(10, 0, 2), (12, 3, 4)]
    ground_truth = [(10, 1, 2), (15, 3, 5)]

    metrics = match_key_events(
        predicted_events=[
            type("Event", (), {"key_id": key_id, "onset_frame": onset, "release_frame": release})()
            for key_id, onset, release in predicted
        ],
        ground_truth_events=[
            type("Event", (), {"key_id": key_id, "onset_frame": onset, "release_frame": release})()
            for key_id, onset, release in ground_truth
        ],
        onset_tolerance_frames=1,
    )

    assert metrics["true_positives"] == 1
    assert metrics["false_positive_events"] == 1
    assert metrics["missed_events"] == 1
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
    assert metrics["mean_abs_timing_error_frames"] == 1.0


def test_build_primitive_instances_recovers_segment_context(tmp_path: Path) -> None:
    primitive_root = tmp_path / "primitives"
    data_output_root = tmp_path / "data"
    song_dir = tmp_path / "dataset" / "song_a"
    primitive_root.mkdir(parents=True, exist_ok=True)
    data_output_root.mkdir(parents=True, exist_ok=True)
    song_dir.mkdir(parents=True, exist_ok=True)

    actions = np.asarray(
        [
            [
                [0.0, 0.1, 0.2],
                [0.3, 0.4, 0.5],
                [0.6, 0.7, 0.8],
                [0.9, 1.0, 1.1],
                [1.2, 1.3, 1.4],
                [1.5, 1.6, 1.7],
            ]
        ],
        dtype=np.float32,
    )
    hand_joints = np.asarray(
        [
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 1.1, 1.2, 1.3],
                [2.0, 2.1, 2.2, 2.3],
                [3.0, 3.1, 3.2, 3.3],
                [4.0, 4.1, 4.2, 4.3],
                [5.0, 5.1, 5.2, 5.3],
            ]
        ],
        dtype=np.float32,
    )
    hand_fingertips = np.asarray(
        [
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5, 0.6, 0.6, 0.6],
                [0.7, 0.7, 0.7, 0.8, 0.8, 0.8],
                [0.9, 0.9, 0.9, 1.0, 1.0, 1.0],
                [1.1, 1.1, 1.1, 1.2, 1.2, 1.2],
                [1.3, 1.3, 1.3, 1.4, 1.4, 1.4],
            ]
        ],
        dtype=np.float32,
    )
    goals = np.zeros((1, 6, 89), dtype=np.float32)
    piano_states = np.zeros((1, 6, 89), dtype=np.float32)
    goals[0, 1:3, 10] = 1.0
    goals[0, 3:5, 11] = 1.0
    piano_states[0, 1:3, 10] = 0.9
    piano_states[0, 3:5, 11] = 0.8

    np.save(song_dir / "actions.npy", actions)
    np.save(song_dir / "hand_joints.npy", hand_joints)
    np.save(song_dir / "hand_fingertips.npy", hand_fingertips)
    np.save(song_dir / "goals.npy", goals)
    np.save(song_dir / "piano_states.npy", piano_states)

    manifest_df = pd.DataFrame(
        [
            {
                "song_id": "song_a",
                "episode_id": "song_a__ep00000",
                "split": "train",
                "backend": "npy_dir",
                "dataset_root": str(song_dir.parent),
                "song_key": "",
                "song_path": str(song_dir),
                "episode_index": 0,
                "note_path": "",
                "control_timestep": 0.05,
                "num_steps": 6,
                "action_dim": 3,
                "goal_dim": 89,
                "piano_state_dim": 89,
                "hand_joint_dim": 4,
                "hand_fingertip_dim": 6,
                "joint_velocity_dim": 0,
                "wrist_pose_dim": 0,
                "hand_pose_dim": 0,
                "has_actions": True,
                "has_goals": True,
                "has_piano_states": True,
                "has_hand_joints": True,
                "has_hand_fingertips": True,
                "has_joint_velocities": False,
                "has_wrist_pose": False,
                "has_hand_pose": False,
            }
        ]
    )
    write_table(manifest_df, data_output_root / "dataset_manifest")

    run_config = {
        "data_output_root": str(data_output_root),
        "data_manifest_name": "dataset_manifest",
        "slim_cache_dir": "slim",
        "gmr_target_actions": True,
    }
    (primitive_root / "run_config.json").write_text(json.dumps(run_config), encoding="utf-8")

    assignments_df = pd.DataFrame(
        [
            {
                "segment_id": "song_a__ep00000_segment_000000",
                "primitive_id": "primitive_000",
                "song_id": "song_a",
                "episode_id": "song_a__ep00000",
                "split": "train",
                "onset_step": 1,
                "end_step": 5,
                "duration_steps": 4,
                "segment_source": "note_aligned",
                "chunk_path": "slim_chunk_00000.npz",
                "chunk_index": 0,
                "gmr_target_name": "actions",
                "score_context_json": "{}",
                "chord_size": 1,
                "key_center": 10.0 / 87.0,
            }
        ]
    )
    write_table(assignments_df, primitive_root / "clustering" / "segment_assignments")

    library_prior_path = primitive_root / "library" / "primitive_000_prior.npz"
    save_npz(library_prior_path, prior_mean=np.ones((4, 3), dtype=np.float32))
    library_df = pd.DataFrame([{"primitive_id": "primitive_000", "prior_path": str(library_prior_path)}])
    write_table(library_df, primitive_root / "library" / "primitive_library")

    slim_paths = resolve_slim_cache_paths(primitive_root, run_config)
    write_slim_chunk(
        paths=slim_paths,
        chunk_name="slim_chunk_00000.npz",
        segment_rows=[assignments_df.iloc[0].to_dict()],
        feature_matrix=np.asarray([[1.0, 2.0]], dtype=np.float32),
        feature_names=["feature_a", "feature_b"],
        gmr_targets=np.asarray([actions[0, 1:5]], dtype=np.float32),
        target_names=["actions"],
    )

    artifacts = load_primitive_online_artifacts(primitive_root)
    sampled = sample_primitive_assignment_rows(
        assignments_df=artifacts.assignments_df,
        sampling_config={"instances_per_primitive": 1},
        seed=7,
    )
    instances, failures = build_primitive_instances(
        artifacts=artifacts,
        assignments_df=sampled,
        events_config={"use_goals": True, "use_piano_states": True},
    )

    assert failures == []
    assert len(instances) == 1
    instance = instances[0]
    np.testing.assert_allclose(instance.start_joint_state, hand_joints[0, 1])
    np.testing.assert_allclose(instance.start_piano_state, piano_states[0, 1])
    assert instance.duration_steps == 4
    assert instance.intended_keys == (10, 11)
    assert instance.realized_keys_gt == (10, 11)
    assert tuple(event.as_tuple() for event in instance.realized_events_gt) == ((10, 0, 2), (11, 2, 4))
    assert instance.conditioning_feature_norm > 0.0


def test_resolve_robopianist_import_root_accepts_clone_root_or_package_dir(tmp_path: Path) -> None:
    clone_root = tmp_path / "clone"
    package_root = clone_root / "robopianist"
    suite_root = package_root / "suite"
    suite_root.mkdir(parents=True)
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (suite_root / "__init__.py").write_text("", encoding="utf-8")

    assert resolve_robopianist_import_root(clone_root) == clone_root
    assert resolve_robopianist_import_root(package_root) == clone_root


def test_resolve_instance_rollout_source_uses_external_manifest(tmp_path: Path) -> None:
    midi_a = tmp_path / "maestro_a.mid"
    midi_b = tmp_path / "maestro_b.mid"
    midi_a.write_bytes(b"MThd")
    midi_b.write_bytes(b"MThd")
    manifest_df = pd.DataFrame(
        [
            {
                "song_id": "maestro_a",
                "episode_id": "maestro_a__ep00000",
                "split": "benchmark",
                "benchmark_name": "maestro",
                "note_path": str(midi_a),
            },
            {
                "song_id": "maestro_b",
                "episode_id": "maestro_b__ep00000",
                "split": "benchmark",
                "benchmark_name": "maestro",
                "note_path": str(midi_b),
            },
        ]
    )
    write_table(manifest_df, tmp_path / "external_midi_manifest")

    source = resolve_instance_rollout_source(
        instance=_make_test_instance(primitive_id="primitive_001"),
        rollout_config={
            "source_mode": "external_midi_manifest",
            "external_midi_manifest": str((tmp_path / "external_midi_manifest.csv").resolve()),
        },
    )

    assert source["source_mode"] == "external_midi_manifest"
    assert source["source_label"] in {"maestro:maestro_a", "maestro:maestro_b"}
    assert source["midi_file"] in {midi_a.resolve(), midi_b.resolve()}


def test_resolve_instance_rollout_source_uses_external_dataset_root(tmp_path: Path) -> None:
    dataset_root = tmp_path / "maestro"
    nested = dataset_root / "2018"
    nested.mkdir(parents=True)
    midi_path = nested / "piece_01.midi"
    midi_path.write_bytes(b"MThd")

    source = resolve_instance_rollout_source(
        instance=_make_test_instance(primitive_id="primitive_000"),
        rollout_config={
            "source_mode": "external_midi_dataset",
            "external_midi_dataset_root": str(dataset_root.resolve()),
            "external_midi_recursive": True,
        },
    )

    assert source["source_mode"] == "external_midi_dataset"
    assert source["source_label"] == "2018/piece_01.midi"
    assert source["midi_file"] == midi_path.resolve()


def _make_test_instance(*, primitive_id: str) -> PrimitiveInstance:
    return PrimitiveInstance(
        segment_id=f"{primitive_id}_segment",
        primitive_id=primitive_id,
        song_id="song_a",
        demo_id="song_a__ep00000",
        episode_id="song_a__ep00000",
        split="train",
        start_frame=0,
        end_frame=4,
        duration_steps=4,
        control_timestep=0.05,
        hand=None,
        start_joint_state=None,
        start_joint_velocity=None,
        start_fingertip_state=None,
        start_piano_state=None,
        intended_keys=(),
        realized_keys_gt=(),
        onset_frames_gt=(),
        release_frames_gt=(),
        conditioning_features=None,
        chunk_path="",
        chunk_index=0,
        raw_chunk_path=None,
        raw_chunk_index=None,
        gmr_target_name="actions",
        primitive_prior_path=None,
        actions_gt=None,
        goals=None,
        piano_states_gt=None,
        hand_joints_gt=None,
        hand_fingertips_gt=None,
        joint_velocities_gt=None,
        source_midi_path=None,
    )
