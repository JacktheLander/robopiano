from __future__ import annotations

import sys
import subprocess
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sonata.primitives.discovery import select_clustering_features  # noqa: E402
from sonata.primitives.features import (  # noqa: E402
    build_condition_feature_vector,
    build_feature_vector_from_arrays,
    enrich_segment_row_with_primitive_context,
)


def _row() -> dict[str, object]:
    return {
        "segment_id": "seg0",
        "target_key_signature": "40-47",
        "key_signature": "40-47",
        "key_center": np.mean([40, 47]) / 87.0,
        "chord_size": 2,
        "duration_steps": 15,
        "motion_energy": 0.2,
        "coarse_family": "dyad_press",
        "segment_alignment": "prepress_to_onset",
        "inactive_start": True,
        "target_onset_step": 12,
        "score_context_json": "{}",
    }


def _arrays() -> dict[str, np.ndarray | None]:
    steps = 15
    hand_joints = np.linspace(0.0, 0.1, steps, dtype=np.float32)[:, None] * np.ones((steps, 39), dtype=np.float32)
    fingertips = np.zeros((steps, 30), dtype=np.float32)
    fingertips[:, 3:6] = np.asarray([-0.08, 0.0, 0.02], dtype=np.float32)
    fingertips[:, 12:15] = np.asarray([0.08, 0.0, 0.02], dtype=np.float32)
    wrist_pose = np.zeros((steps, 7), dtype=np.float32)
    wrist_pose[:, :3] = np.asarray([0.0, -0.08, 0.05], dtype=np.float32)
    wrist_pose[:, 3:7] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    actions = np.gradient(hand_joints, axis=0).astype(np.float32)
    piano_states = np.zeros((steps, 89), dtype=np.float32)
    piano_states[12:, [40, 47]] = 1.0
    return {
        "hand_joints": hand_joints,
        "joint_velocities": np.gradient(hand_joints, axis=0).astype(np.float32),
        "actions": actions,
        "goals": None,
        "piano_states": piano_states,
        "hand_fingertips": fingertips,
        "wrist_pose": wrist_pose,
        "hand_pose": None,
    }


def _config() -> dict[str, object]:
    return {
        "trajectory_resample_steps": 4,
        "gmr_resample_steps": 8,
        "fallback_action_dim": 39,
        "include_action_trajectory": True,
        "num_duration_buckets": 6,
        "num_dynamics_buckets": 5,
        "primitive_frame": {
            "mode": "wrist_key_relative",
            "normalize_wrist_translation": True,
            "normalize_chord_center": True,
        },
        "clustering": {
            "mode": "reusable_motor_motif",
            "exclude_absolute_key_position_from_clustering": True,
            "include_relative_geometry": True,
            "include_finger_set": True,
            "include_chord_interval_pattern": True,
            "include_motion_family": True,
            "include_wrist_relative_features": True,
        },
        "gmr_condition_features": [
            "relative_wrist_anchor",
            "chord_center_key_id_normalized",
            "interval_pattern_embedding",
            "finger_set_id",
            "fingertip_to_target_key_offsets",
            "start_joint_state",
            "duration_bucket",
            "dynamics_bucket",
        ],
    }


def test_wrist_relative_context_features_are_extracted_and_stored() -> None:
    row = enrich_segment_row_with_primitive_context(_row(), _arrays(), _config())

    assert row["primitive_frame_mode"] == "wrist_key_relative"
    assert row["target_key_ids"] == "[40, 47]"
    assert row["interval_pattern_bucket"] == "-4,4"
    assert row["finger_set"] != "unknown"
    assert row["segment_alignment"] == "prepress_to_onset"


def test_condition_vector_and_reusable_clustering_exclude_absolute_keyset() -> None:
    row = enrich_segment_row_with_primitive_context(_row(), _arrays(), _config())
    vector, names = build_feature_vector_from_arrays(row=row, arrays=_arrays(), config=_config())
    condition, condition_names, _ = build_condition_feature_vector(vector, names, _config())

    assert condition.size > 0
    assert any(name.startswith("context_relative_wrist_anchor") for name in condition_names)
    selected, selected_names, _ = select_clustering_features(
        feature_matrix=vector[None, :],
        feature_names=names,
        config=_config(),
    )

    assert selected.shape[1] < len(names)
    assert not any(name.startswith("target_keyset_") for name in selected_names)
    assert any(name.startswith("context_interval_pattern") for name in selected_names)


def test_validation_contract_accepts_wrist_relative_action_priors(tmp_path: Path) -> None:
    primitive_root = tmp_path / "primitive_root"
    validation_dir = primitive_root / "validation"
    validation_dir.mkdir(parents=True)
    contract = {
        "gmr_target_name": "actions",
        "any_prior_not_action_dim": False,
        "any_prior_uses_non_action_target": False,
        "no_piano_state_or_goal_target": True,
        "wrist_key_relative_frame": True,
        "primitive_frame_mode": "wrist_key_relative",
        "condition_features_exist": True,
        "missing_library_metadata_columns": [],
        "absolute_key_position_main_clustering_driver": False,
        "causal_segment": True,
        "percent_segments_with_inactive_start": 100.0,
        "percent_segments_with_activation_after_start": 100.0,
        "segment_alignment": "prepress_to_onset",
        "num_rejected_segments": 0,
        "rejection_counts": {},
        "pass_training_contract": True,
    }
    (validation_dir / "primitive_training_contract.json").write_text(__import__("json").dumps(contract), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "validate_primitives_contract.py"), "--primitive-root", str(primitive_root)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
