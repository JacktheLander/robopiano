from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sonata.data.schema import ScoreEvent
from sonata.primitives.discovery import fit_primitive_gmm
from sonata.primitives.evaluation import Stage1OnlineEvaluator
from sonata.primitives.features import build_feature_vector_from_arrays
from sonata.primitives.gmr import fit_phase_gmr_with_selection
from sonata.primitives.gpu_utils import GpuBackend
from sonata.primitives.segmenters import _segment_cache_matches_config, _segment_config_signature, build_segmenter


def test_keyset_onset_segments_simultaneous_keys_and_splits_later_onsets() -> None:
    config = {
        "segmenter_name": "keyset_onset",
        "segmentation_strategy": "keyset_onset",
        "segment_press_pre_steps": 2,
        "segment_press_post_steps": 6,
        "segment_truncate_at_next_onset": True,
        "segment_min_len": 4,
        "segment_max_len": 12,
        "chord_tolerance_steps": 1,
    }
    episode = SimpleNamespace(
        hand_joints=np.cumsum(np.ones((32, 6), dtype=np.float32) * 0.01, axis=0),
        actions=np.cumsum(np.ones((32, 4), dtype=np.float32) * 0.02, axis=0),
        hand_fingertips=None,
        wrist_pose=np.cumsum(np.ones((32, 3), dtype=np.float32) * 0.01, axis=0),
        goals=np.zeros((32, 89), dtype=np.float32),
        piano_states=None,
    )
    events = [
        ScoreEvent("ev0", "song", "ep", 10, 13, 0.5, 0.65, (40,), 1, 40 / 87.0, 10, "score"),
        ScoreEvent("ev1", "song", "ep", 10, 13, 0.5, 0.65, (44,), 1, 44 / 87.0, 0, "score"),
        ScoreEvent("ev2", "song", "ep", 14, 16, 0.7, 0.8, (47,), 1, 47 / 87.0, 4, "score"),
    ]

    segmenter = build_segmenter(config)
    segments = segmenter.segment(episode, events)

    assert len(segments) == 2
    assert segments[0].segment_source == "keyset_onset"
    assert segments[0].target_key_signature == "40-44"
    assert segments[0].target_key_count == 2
    assert segments[0].coarse_family == "dyad_press"
    assert segments[0].target_onset_step == 10
    assert segments[0].end_step <= 14
    assert segments[0].truncated_by_next_onset is True
    assert segments[1].target_key_signature == "47"
    assert segments[1].coarse_family == "single_press"


def test_keyset_features_are_stable_for_chord_targets() -> None:
    arrays = {
        "hand_joints": np.linspace(0.0, 1.0, 24, dtype=np.float32).reshape(6, 4),
        "joint_velocities": None,
        "actions": np.linspace(0.0, 1.0, 18, dtype=np.float32).reshape(6, 3),
        "goals": np.zeros((6, 89), dtype=np.float32),
        "piano_states": None,
        "hand_fingertips": None,
        "wrist_pose": None,
        "hand_pose": None,
    }
    row = {
        "segment_id": "seg",
        "score_context_json": "{}",
        "duration_steps": 6,
        "motion_energy": 1.0,
        "chord_size": 3,
        "key_center": 44 / 87.0,
        "start_state_norm": 0.0,
        "end_state_norm": 1.0,
        "boundary_energy": 0.0,
        "boundary_alignment_score": 0.0,
        "proposal_size": 3,
        "proposal_span_steps": 1,
        "target_key_count": 3,
        "target_key_signature": "40-44-47",
        "next_onset_gap_steps": 4,
        "truncated_by_next_onset": True,
    }

    vector, names = build_feature_vector_from_arrays(
        row=row,
        arrays=arrays,
        config={
            "trajectory_resample_steps": 4,
            "include_action_trajectory": True,
            "fallback_action_dim": 3,
            "relative_wrist_frame": True,
            "relative_key_center_frame": True,
            "hand_specific_normalization": True,
        },
    )

    lookup = {name: index for index, name in enumerate(names)}
    assert vector[lookup["target_keyset_0040"]] == 1.0
    assert vector[lookup["target_keyset_0044"]] == 1.0
    assert vector[lookup["target_keyset_0047"]] == 1.0
    assert vector[lookup["target_keyset_summary_0000"]] == 3.0
    assert "target_interval_hist_0004" in lookup


def test_segmentation_cache_signature_rejects_stale_strategy(tmp_path: Path) -> None:
    config = {
        "segmenter_name": "keyset_onset",
        "segmentation_strategy": "keyset_onset",
        "segment_press_pre_steps": 2,
        "segment_press_post_steps": 6,
        "segment_min_len": 4,
        "segment_max_len": 12,
        "chord_tolerance_steps": 1,
    }
    manifest_path = tmp_path / "segment_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "segment_strategy": "note_group_refined",
                "segment_config_signature": _segment_config_signature(config),
            }
        ),
        encoding="utf-8",
    )

    assert _segment_cache_matches_config(manifest_path, config) is False

    manifest_path.write_text(
        json.dumps(
            {
                "segment_strategy": "keyset_onset",
                "segment_config_signature": _segment_config_signature(config),
            }
        ),
        encoding="utf-8",
    )
    assert _segment_cache_matches_config(manifest_path, config) is True


def test_note_group_refined_groups_repeated_events() -> None:
    config = {
        "segmenter_name": "note_group_refined",
        "segmentation_strategy": "note_group_refined",
        "pre_steps": 2,
        "post_steps": 3,
        "segment_grouping_window": 3,
        "segment_boundary_refine_radius": 2,
        "segment_merge_enabled": True,
        "segment_split_enabled": False,
        "segment_duplicate_iou_threshold": 0.9,
        "segment_min_len": 2,
        "segment_max_len": 16,
        "segment_group_max_events": 8,
        "segment_group_max_span_steps": 16,
        "segment_group_key_center_tolerance": 0.2,
        "segment_repeat_window_steps": 6,
        "chord_tolerance_steps": 1,
    }
    episode = SimpleNamespace(
        hand_joints=np.cumsum(np.ones((32, 6), dtype=np.float32) * 0.01, axis=0),
        actions=np.cumsum(np.ones((32, 4), dtype=np.float32) * 0.02, axis=0),
        hand_fingertips=None,
        wrist_pose=np.cumsum(np.ones((32, 3), dtype=np.float32) * 0.01, axis=0),
        goals=np.zeros((32, 88), dtype=np.float32),
        piano_states=None,
    )
    events = [
        ScoreEvent("ev0", "song", "ep", 10, 12, 0.5, 0.6, (40,), 1, 0.45, 2, "score"),
        ScoreEvent("ev1", "song", "ep", 12, 14, 0.6, 0.7, (40,), 1, 0.45, 2, "score"),
        ScoreEvent("ev2", "song", "ep", 14, 16, 0.7, 0.8, (40,), 1, 0.45, 2, "score"),
    ]

    segmenter = build_segmenter(config)
    segments = segmenter.segment(episode, events)

    assert len(segments) == 1
    assert segments[0].proposal_size == 3
    assert segments[0].coarse_family == "repeat_press"
    assert segmenter.last_stats["accepted_segments"] == 1


def test_stage1_evaluator_writes_failure_summary_on_stop(tmp_path: Path) -> None:
    evaluator = Stage1OnlineEvaluator(
        config={
            "enable_stage1_online_eval": True,
            "stage1_warn_only": False,
            "stage1_early_stop_enabled": True,
            "stage1_eval_interval_segments": 4,
            "stage1_eval_subsample_size": 16,
            "stage1_min_segments_before_stop_check": 4,
            "stage1_max_short_segment_frac": 0.2,
            "stage1_patience_windows": 10,
            "segment_min_len": 4,
            "segment_max_len": 32,
        },
        output_dir=tmp_path,
        logger=None,
    )
    rows = [
        {
            "episode_id": "ep",
            "onset_step": index * 2,
            "end_step": index * 2 + 1,
            "duration_steps": 1,
            "coarse_family": "single_press",
            "boundary_energy": 0.1,
            "boundary_alignment_score": 0.2,
            "duplicate_iou": 0.0,
        }
        for index in range(4)
    ]

    decision = evaluator.observe_segmentation(rows, {"accepted_segments": 4, "proposed_segments": 4})

    assert decision is not None
    assert decision.stop is True
    assert evaluator.failure_json_path.exists()
    assert evaluator.failure_txt_path.exists()


def test_fit_phase_gmr_with_selection_prefers_multiple_components_for_bimodal_motion() -> None:
    horizon = 16
    trajectories = []
    for split in (4, 11):
        for _ in range(8):
            signal = np.zeros((horizon, 2), dtype=np.float32)
            signal[split:, 0] = 1.0
            signal[split:, 1] = 0.5
            trajectories.append(signal)
    stacked = np.stack(trajectories, axis=0)

    _, diagnostics = fit_phase_gmr_with_selection(
        trajectories=stacked,
        component_candidates=[1, 2, 4],
        reg_covar=1e-4,
        random_state=3,
        min_samples_per_component=2,
        strike_weight=2.0,
    )

    assert diagnostics["selected"]["component_count"] >= 2
    assert diagnostics["selected"]["weighted_strike_error"] >= diagnostics["selected"]["reconstruction_mse"]


def test_family_aware_clustering_keeps_bucket_assignments() -> None:
    rng = np.random.default_rng(5)
    single_features = rng.normal(loc=0.0, scale=0.2, size=(6, 6)).astype(np.float32)
    chord_features = rng.normal(loc=4.0, scale=0.2, size=(6, 6)).astype(np.float32)
    feature_matrix = np.vstack([single_features, chord_features]).astype(np.float32)
    segment_df = pd.DataFrame(
        {
            "segment_id": [f"seg_{index:02d}" for index in range(12)],
            "split": ["train"] * 12,
            "coarse_family": ["single_press"] * 6 + ["chord_press"] * 6,
            "heuristic_family": ["single"] * 6 + ["chord"] * 6,
            "duration_steps": np.ones((12,), dtype=np.float32) * 8,
            "motion_energy": np.ones((12,), dtype=np.float32),
            "chord_size": [1] * 6 + [3] * 6,
            "key_center": np.linspace(0.1, 0.9, 12),
            "start_state_norm": np.zeros((12,), dtype=np.float32),
            "end_state_norm": np.zeros((12,), dtype=np.float32),
        }
    )

    assignments_df, sweep_df, bundle = fit_primitive_gmm(
        segment_df=segment_df,
        feature_matrix=feature_matrix,
        feature_names=[f"feat_{index}" for index in range(feature_matrix.shape[1])],
        config={
            "pca_components": 4,
            "gmm_k_candidates": [1, 2],
            "model_selection_metric": "bic",
            "gmm_reg_covar": 1e-4,
            "gmm_seed": 7,
            "gmm_n_init": 2,
            "gmm_use_staged_k_search": True,
            "gmm_k_screen_subset_size": 8,
            "gmm_top_k_full_fits": 1,
            "gmm_candidate_covariance_types": ["full"],
            "gmm_screen_max_iter": 8,
            "silhouette_max_examples": 32,
            "family_aware_clustering": True,
            "gpu_subsample_limit": 64,
        },
        gpu_backend=GpuBackend(name="cpu", active=False),
        evaluator=None,
    )

    assert set(assignments_df["primitive_family_bucket"].astype(str)) == {"single_press", "chord_press"}
    assert bundle["family_aware"] is True
    assert set(sweep_df["bucket_name"].astype(str)) == {"single_press", "chord_press"}
