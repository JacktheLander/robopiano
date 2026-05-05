from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from glob import glob
import json
import logging
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from sonata.data.loading import build_manifest_lookup, load_episode_record, load_stage1_source_manifest
from sonata.evaluation.causal_rollout_contract import (
    CausalRolloutConfig,
    collect_fingertip_key_contacts,
    contact_gated_keypress_metrics,
    default_causal_eval_config,
    forbid_dataset_piano_state_restore,
    mark_uncausal_result,
    reset_or_validate_neutral_piano,
    run_zero_action_ablation,
)
from sonata.evaluation.task_config import build_rollout_task_kwargs, validate_rollout_action_dim
from sonata.primitives.features import build_feature_vector_from_arrays, load_feature_matrix_from_store
from sonata.utils.io import ensure_dir, read_json, save_npz, write_json, write_table
from sonata.utils.robopianist import ensure_local_robopianist_on_path, format_robopianist_import_error

LOGGER = logging.getLogger(__name__)
_NUM_PIANO_KEYS = 88
_ENV_SUFFIX_RE = re.compile(r"(-v\d+)_\d+$")
_DEFAULT_EXAMPLE_MIDI_SOURCES = (
    {
        "label": "TwinkleTwinkleRousseau",
        "environment_name": "RoboPianist-debug-TwinkleTwinkleRousseau-v0",
        "relative_path": Path("music/data/rousseau/twinkle-twinkle-trimmed.mid"),
    },
    {
        "label": "NocturneRousseau",
        "environment_name": "RoboPianist-debug-NocturneRousseau-v0",
        "relative_path": Path("music/data/rousseau/nocturne-trimmed.mid"),
    },
)


@dataclass(slots=True, frozen=True)
class KeyEvent:
    key_id: int
    onset_frame: int
    release_frame: int

    @property
    def duration_frames(self) -> int:
        return max(int(self.release_frame) - int(self.onset_frame), 0)

    def as_tuple(self) -> tuple[int, int, int]:
        return int(self.key_id), int(self.onset_frame), int(self.release_frame)


@dataclass(slots=True)
class KeyEventBundle:
    source: str
    key_roll: np.ndarray
    sustain_roll: np.ndarray
    events: list[KeyEvent]
    unique_keys: tuple[int, ...]


@dataclass(slots=True)
class PrimitiveInstance:
    segment_id: str
    primitive_id: str
    song_id: str
    demo_id: str | None
    episode_id: str
    split: str
    start_frame: int
    end_frame: int
    duration_steps: int
    control_timestep: float
    hand: str | None
    start_joint_state: np.ndarray | None
    start_joint_velocity: np.ndarray | None
    start_fingertip_state: np.ndarray | None
    start_piano_state: np.ndarray | None
    intended_keys: tuple[int, ...]
    realized_keys_gt: tuple[int, ...]
    onset_frames_gt: tuple[int, ...]
    release_frames_gt: tuple[int, ...]
    conditioning_features: np.ndarray | None
    chunk_path: str
    chunk_index: int
    raw_chunk_path: str | None
    raw_chunk_index: int | None
    gmr_target_name: str
    primitive_prior_path: str | None
    segment_source: str | None = None
    heuristic_family: str | None = None
    coarse_family: str | None = None
    control_phase: str | None = None
    chord_size: int | None = None
    key_center: float | None = None
    intended_events: list[KeyEvent] = field(default_factory=list)
    realized_events_gt: list[KeyEvent] = field(default_factory=list)
    actions_gt: np.ndarray | None = None
    goals: np.ndarray | None = None
    piano_states_gt: np.ndarray | None = None
    hand_joints_gt: np.ndarray | None = None
    hand_fingertips_gt: np.ndarray | None = None
    joint_velocities_gt: np.ndarray | None = None
    source_midi_path: str | None = None
    target_source: str = "goals"
    primitive_frame_mode: str = "absolute"
    relative_wrist_anchor: str = "[]"
    finger_set: str = "unknown"
    target_key_ids: str = "[]"
    interval_pattern: str = "[]"

    @property
    def conditioning_feature_norm(self) -> float:
        if self.conditioning_features is None:
            return float("nan")
        return float(np.linalg.norm(self.conditioning_features))


@dataclass(slots=True)
class PrimitiveInstanceBuildFailure:
    segment_id: str
    primitive_id: str
    song_id: str
    episode_id: str
    error: str


@dataclass(slots=True)
class PrimitiveOnlineRolloutResult:
    segment_id: str
    primitive_id: str
    status: str
    success: bool
    error: str | None
    restore_mode: str
    alignment_mode: str
    predicted_actions: np.ndarray | None
    raw_observed_piano_states: np.ndarray | None
    observed_piano_states: np.ndarray | None
    raw_observed_hand_joints: np.ndarray | None
    observed_hand_joints: np.ndarray | None
    raw_observed_hand_fingertips: np.ndarray | None
    observed_hand_fingertips: np.ndarray | None
    rollout_source_mode: str
    rollout_source_label: str | None
    rollout_environment_name: str
    rollout_midi_path: str | None
    observed_key_events: list[KeyEvent] = field(default_factory=list)
    observed_key_roll: np.ndarray | None = None
    contact_roll: np.ndarray | None = None
    contact_method: str = "unavailable"
    causal_metrics: dict[str, Any] = field(default_factory=dict)
    zero_action_metrics: dict[str, Any] = field(default_factory=dict)
    causal_validated: bool = False
    causal_failure_reason: str | None = None
    neutral_start_passed: bool = False
    zero_action_ablation_passed: bool | None = None
    contact_gate_passed: bool = False
    target_source: str = "goals"
    observed_source: str = "simulator_piano_activation"
    action_max_abs: float = float("nan")
    hand_action_max_abs: float = float("nan")
    pedal_action_max_abs: float = float("nan")
    pedal_action_indices: tuple[int, ...] = ()
    video_path: str | None = None
    video_format: str | None = None
    video_warning: str | None = None
    notes: list[str] = field(default_factory=list)
    condition_features_used: tuple[str, ...] = ()
    condition_feature_norm: float = float("nan")
    conditioned_prior_used: bool = False
    unconditional_fallback_used: bool = True
    primitive_frame_mode: str = "absolute"
    relative_wrist_anchor: str = "[]"
    finger_set: str = "unknown"
    target_key_ids: str = "[]"
    interval_pattern: str = "[]"


@dataclass(slots=True)
class PrimitiveLibraryEntry:
    primitive_id: str
    prior_path: Path | None
    prior_mean: np.ndarray | None
    prototype_means: np.ndarray | None
    prototype_latent_centroids: np.ndarray | None
    prototype_weights: np.ndarray | None
    default_prototype_index: int
    metadata: dict[str, Any]
    condition_feature_names: tuple[str, ...] = ()
    condition_feature_indices: tuple[int, ...] = ()
    condition_mean: np.ndarray | None = None
    condition_std: np.ndarray | None = None
    condition_regression_coef: np.ndarray | None = None
    condition_regression_intercept: np.ndarray | None = None
    supports_conditioned_rollout: bool = False
    fallback_unconditional_available: bool = True


@dataclass(slots=True)
class PrimitiveOnlineArtifacts:
    primitive_root: Path
    run_config: dict[str, Any]
    assignments_df: pd.DataFrame
    library_df: pd.DataFrame
    manifest_df: pd.DataFrame
    manifest_lookup: dict[tuple[str, str], dict[str, Any]]


class RoboPianistPrimitiveRuntime:
    def __init__(self, *, robopianist_root: str | Path | None, seed: int, logger: logging.Logger | None = None):
        self.robopianist_root = robopianist_root
        self.seed = int(seed)
        self.logger = logger or LOGGER
        self._suite = None
        self._env_cache: dict[tuple[str, str | None, float, int], Any] = {}

    def preflight(self) -> tuple[bool, str | None]:
        ensure_local_robopianist_on_path(self.robopianist_root)
        try:
            from robopianist import suite
        except Exception as exc:  # pragma: no cover
            return False, format_robopianist_import_error(exc, self.robopianist_root)
        self._suite = suite
        return True, None

    def get_env(
        self,
        *,
        environment_name: str,
        midi_file: Path | None,
        control_timestep: float,
        expected_action_dim: int,
    ) -> Any:
        if self._suite is None:
            available, error = self.preflight()
            if not available:
                raise RuntimeError(error or "RoboPianist runtime was unavailable.")
        key = (
            str(environment_name),
            str(midi_file.resolve()) if midi_file is not None else None,
            float(control_timestep),
            int(expected_action_dim),
        )
        env = self._env_cache.get(key)
        if env is not None:
            return env
        env = self._suite.load(
            environment_name=environment_name,
            midi_file=midi_file,
            seed=self.seed,
            task_kwargs=build_rollout_task_kwargs(
                control_timestep=float(control_timestep),
                expected_action_dim=int(expected_action_dim),
            ),
        )
        self._env_cache[key] = env
        return env

    def close(self) -> None:
        for env in self._env_cache.values():
            _safe_close(env)
        self._env_cache = {}


def evaluate_primitives_online(config: dict[str, Any], logger: logging.Logger | None = None) -> dict[str, Any]:
    logger = logger or LOGGER
    resolved = _resolve_eval_config(config)
    primitive_root = Path(str(resolved["primitive_root"])).resolve()
    output_root = ensure_dir(resolved["output_root"])
    plots_dir = output_root / "plots"
    debug_dir = output_root / "debug"
    videos_dir = output_root / "videos"
    frames_tmp_dir = output_root / "frames_tmp"
    if bool(resolved.get("force", False)):
        _clear_previous_outputs(output_root)
    ensure_dir(output_root)
    if bool(resolved.get("save_plots", True)):
        ensure_dir(plots_dir)
    if bool(resolved.get("save_debug", False)):
        ensure_dir(debug_dir)
    if bool(resolved["rollout"].get("render_video", False)):
        ensure_dir(videos_dir)

    write_json(_json_ready(resolved), output_root / "config_snapshot.json")
    artifacts = load_primitive_online_artifacts(primitive_root)
    sampled = sample_primitive_assignment_rows(
        assignments_df=artifacts.assignments_df,
        sampling_config=resolved["sampling"],
        seed=int(resolved["seed"]),
    )
    preflight = {
        "primitive_root": str(primitive_root),
        "output_root": str(output_root),
        "num_assignment_rows": int(len(artifacts.assignments_df)),
        "num_sampled_rows": int(len(sampled)),
        "num_sampled_primitives": int(sampled["primitive_id"].nunique()) if not sampled.empty else 0,
    }
    write_json(_json_ready(preflight), output_root / "preflight.json")
    if sampled.empty:
        empty_df = pd.DataFrame()
        write_table(empty_df, output_root / "primitive_instances_enriched")
        write_table(empty_df, output_root / "primitive_instance_metrics")
        write_table(empty_df, output_root / "primitive_summary_metrics")
        aggregate = {
            "status": "empty",
            "num_assignment_rows": int(preflight["num_assignment_rows"]),
            "num_sampled_rows": 0,
            "num_result_rows": 0,
        }
        write_json(aggregate, output_root / "aggregate_metrics.json")
        write_json({}, output_root / "failure_counts.json")
        return {
            "status": "empty",
            "output_root": str(output_root),
            "num_instances": 0,
            "num_primitives": 0,
            "instance_metrics_path": str((output_root / "primitive_instance_metrics.csv").resolve()),
            "summary_metrics_path": str((output_root / "primitive_summary_metrics.csv").resolve()),
            "aggregate_metrics_path": str((output_root / "aggregate_metrics.json").resolve()),
        }

    existing_rows = _load_existing_instance_rows(output_root) if bool(resolved.get("resume", True)) else []
    completed_segment_ids = {str(row["segment_id"]) for row in existing_rows if row.get("segment_id")}

    instances, build_failures = build_primitive_instances(
        artifacts=artifacts,
        assignments_df=sampled,
        events_config=resolved["events"],
    )
    runtime = RoboPianistPrimitiveRuntime(
        robopianist_root=resolved.get("robopianist_root"),
        seed=int(resolved["seed"]),
        logger=logger,
    )
    runtime_available, runtime_error = runtime.preflight()
    write_json(
        {
            "available": bool(runtime_available),
            "robopianist_root": str(resolved.get("robopianist_root")) if resolved.get("robopianist_root") else None,
            "error": runtime_error,
        },
        output_root / "runtime_status.json",
    )
    if not runtime_available:
        payload = {
            "status": "backend_unavailable",
            "error": runtime_error,
            "output_root": str(output_root),
        }
        write_json(payload, output_root / "aggregate_metrics.json")
        return payload

    primitive_library = load_primitive_library_lookup(
        library_df=artifacts.library_df,
        primitive_ids=sorted({instance.primitive_id for instance in instances}),
        primitive_root=artifacts.primitive_root,
    )

    jsonl_path = output_root / "primitive_instance_metrics.jsonl"
    result_rows = list(existing_rows)
    failure_rows = [_build_failure_row_from_build_failure(item) for item in build_failures]
    for row in failure_rows:
        if str(row["segment_id"]) in completed_segment_ids:
            continue
        _append_jsonl_row(jsonl_path, row)
        result_rows.append(row)
        completed_segment_ids.add(str(row["segment_id"]))

    rendered_instances = 0
    max_render_instances = resolved["rollout"].get("max_render_instances")
    for instance in instances:
        if instance.segment_id in completed_segment_ids:
            continue
        library_entry = primitive_library.get(instance.primitive_id)
        rollout_config = dict(resolved["rollout"])
        should_render = bool(rollout_config.get("render_video", False)) and (
            max_render_instances is None or rendered_instances < int(max_render_instances)
        )
        rollout_config["render_video"] = should_render
        rollout_config["videos_dir"] = str(videos_dir)
        rollout_config["frames_tmp_dir"] = str(frames_tmp_dir)
        rollout = rollout_primitive_instance(
            instance=instance,
            library_entry=library_entry,
            runtime=runtime,
            rollout_config=rollout_config,
        )
        if should_render and rollout.video_path is not None:
            rendered_instances += 1
        debug_artifact_path = None
        if bool(resolved.get("save_debug", False)):
            debug_artifact_path = save_rollout_debug_artifact(
                output_dir=debug_dir,
                instance=instance,
                rollout=rollout,
            )
        row = build_instance_result_row(
            instance=instance,
            rollout=rollout,
            event_config=resolved["events"],
            debug_artifact_path=debug_artifact_path,
        )
        _append_jsonl_row(jsonl_path, row)
        result_rows.append(row)
        completed_segment_ids.add(instance.segment_id)

    runtime.close()
    result_df = (
        pd.DataFrame(result_rows).sort_values(["primitive_id", "segment_id"], kind="stable").reset_index(drop=True)
        if result_rows
        else pd.DataFrame()
    )
    write_table(result_df, output_root / "primitive_instances_enriched")
    metric_columns = [
        column
        for column in result_df.columns
        if column
        not in {
            "intended_events_json",
            "realized_events_gt_json",
            "predicted_events_json",
            "intended_keys_json",
            "realized_keys_gt_json",
            "predicted_keys_json",
            "error",
            "notes_json",
        }
    ]
    write_table(result_df[metric_columns], output_root / "primitive_instance_metrics")

    summary_df = aggregate_per_primitive_reports(result_df)
    write_table(summary_df, output_root / "primitive_summary_metrics")

    failure_counts = Counter(str(item) for item in result_df.get("status", pd.Series(dtype=object)).fillna("unknown"))
    write_json(dict(sorted(failure_counts.items())), output_root / "failure_counts.json")

    aggregate = build_aggregate_metrics(
        result_df=result_df,
        summary_df=summary_df,
        preflight=preflight,
        runtime_error=runtime_error,
    )
    write_json(_json_ready(aggregate), output_root / "aggregate_metrics.json")

    if bool(resolved.get("save_plots", True)):
        save_summary_plots(
            output_dir=plots_dir,
            result_df=result_df,
            summary_df=summary_df,
            max_plot_primitives=int(resolved["aggregation"].get("max_plot_primitives", 24)),
        )
        if bool(resolved.get("save_debug", False)):
            save_representative_timing_plots(
                output_dir=plots_dir / "representatives",
                result_df=result_df,
                top_k=int(resolved["aggregation"].get("top_k_examples", 3)),
            )

    return {
        "status": "completed",
        "output_root": str(output_root),
        "num_instances": int(len(result_df)),
        "num_primitives": int(summary_df["primitive_id"].nunique()) if not summary_df.empty else 0,
        "instance_metrics_path": str((output_root / "primitive_instance_metrics.csv").resolve()),
        "summary_metrics_path": str((output_root / "primitive_summary_metrics.csv").resolve()),
        "aggregate_metrics_path": str((output_root / "aggregate_metrics.json").resolve()),
    }


def load_primitive_online_artifacts(primitive_root: Path) -> PrimitiveOnlineArtifacts:
    run_config = read_json(primitive_root / "run_config.json")
    assignments_base = primitive_root / "clustering" / "segment_assignments"
    if assignments_base.with_suffix(".csv").exists() or assignments_base.with_suffix(".parquet").exists():
        assignments_df = _read_table_base(assignments_base)
    else:
        assignments_df = _read_table_base(primitive_root / "tokens" / "primitive_tokens")
    library_df = _read_table_base(primitive_root / "library" / "primitive_library")
    manifest_df = load_stage1_source_manifest(primitive_root)
    manifest_lookup = build_manifest_lookup(manifest_df)
    return PrimitiveOnlineArtifacts(
        primitive_root=primitive_root,
        run_config=run_config,
        assignments_df=assignments_df,
        library_df=library_df,
        manifest_df=manifest_df,
        manifest_lookup=manifest_lookup,
    )


def sample_primitive_assignment_rows(
    *,
    assignments_df: pd.DataFrame,
    sampling_config: dict[str, Any],
    seed: int,
) -> pd.DataFrame:
    if assignments_df.empty:
        return assignments_df.copy()
    frame = assignments_df.copy()
    primitive_ids = [str(item) for item in sampling_config.get("primitive_ids", []) if str(item).strip()]
    if primitive_ids:
        frame = frame.loc[frame["primitive_id"].astype(str).isin(primitive_ids)]
    split = sampling_config.get("split")
    if split not in (None, "", "all") and "split" in frame.columns:
        frame = frame.loc[frame["split"].astype(str) == str(split)]
    min_chord_size = sampling_config.get("min_chord_size")
    max_chord_size = sampling_config.get("max_chord_size")
    if min_chord_size is not None and "chord_size" in frame.columns:
        frame = frame.loc[pd.to_numeric(frame["chord_size"], errors="coerce") >= int(min_chord_size)]
    if max_chord_size is not None and "chord_size" in frame.columns:
        frame = frame.loc[pd.to_numeric(frame["chord_size"], errors="coerce") <= int(max_chord_size)]
    min_duration_steps = sampling_config.get("min_duration_steps")
    max_duration_steps = sampling_config.get("max_duration_steps")
    if min_duration_steps is not None and "duration_steps" in frame.columns:
        frame = frame.loc[pd.to_numeric(frame["duration_steps"], errors="coerce") >= int(min_duration_steps)]
    if max_duration_steps is not None and "duration_steps" in frame.columns:
        frame = frame.loc[pd.to_numeric(frame["duration_steps"], errors="coerce") <= int(max_duration_steps)]
    if frame.empty:
        return frame.reset_index(drop=True)

    rng = np.random.default_rng(int(seed))
    instances_per_primitive = sampling_config.get("instances_per_primitive")
    sampled_groups: list[pd.DataFrame] = []
    if instances_per_primitive is not None:
        for _, group in frame.groupby("primitive_id", sort=True):
            group = group.reset_index(drop=True)
            take = min(int(instances_per_primitive), len(group))
            indices = np.arange(len(group), dtype=np.int64)
            rng.shuffle(indices)
            sampled_groups.append(group.iloc[np.sort(indices[:take])])
        frame = pd.concat(sampled_groups, ignore_index=True) if sampled_groups else frame.iloc[0:0].copy()
    max_instances = sampling_config.get("max_instances")
    if max_instances is None:
        max_instances = sampling_config.get("max_instances_total")
    if max_instances is not None and len(frame) > int(max_instances):
        indices = np.arange(len(frame), dtype=np.int64)
        rng.shuffle(indices)
        frame = frame.iloc[np.sort(indices[: int(max_instances)])]
    sort_columns = [column for column in ("primitive_id", "song_id", "episode_id", "onset_step", "segment_id") if column in frame.columns]
    return frame.sort_values(sort_columns, kind="stable").reset_index(drop=True)


def build_primitive_instances(
    *,
    artifacts: PrimitiveOnlineArtifacts,
    assignments_df: pd.DataFrame,
    events_config: dict[str, Any],
) -> tuple[list[PrimitiveInstance], list[PrimitiveInstanceBuildFailure]]:
    if assignments_df.empty:
        return [], []
    feature_lookup = _load_conditioning_feature_lookup(
        primitive_root=artifacts.primitive_root,
        assignments_df=assignments_df,
        stage1_config=artifacts.run_config,
    )
    instances: list[PrimitiveInstance] = []
    failures: list[PrimitiveInstanceBuildFailure] = []
    grouped = assignments_df.groupby("episode_id", sort=False)
    for episode_id, group in grouped:
        first_row = group.iloc[0]
        key = (str(first_row.get("song_id", "")), str(episode_id))
        manifest_row = artifacts.manifest_lookup.get(key)
        if manifest_row is None:
            for row in group.itertuples(index=False):
                failures.append(
                    PrimitiveInstanceBuildFailure(
                        segment_id=str(row.segment_id),
                        primitive_id=str(row.primitive_id),
                        song_id=str(row.song_id),
                        episode_id=str(row.episode_id),
                        error=f"Missing source manifest row for episode `{row.episode_id}`.",
                    )
                )
            continue
        episode = load_episode_record(manifest_row)
        for row in group.itertuples(index=False):
            try:
                arrays = {
                    "actions": _slice_episode_field(episode.actions, int(row.onset_step), int(row.end_step)),
                    "goals": _slice_episode_field(episode.goals, int(row.onset_step), int(row.end_step)),
                    "piano_states": _slice_episode_field(episode.piano_states, int(row.onset_step), int(row.end_step)),
                    "hand_joints": _slice_episode_field(episode.hand_joints, int(row.onset_step), int(row.end_step)),
                    "joint_velocities": _slice_episode_field(episode.joint_velocities, int(row.onset_step), int(row.end_step)),
                    "hand_fingertips": _slice_episode_field(episode.hand_fingertips, int(row.onset_step), int(row.end_step)),
                    "wrist_pose": _slice_episode_field(episode.wrist_pose, int(row.onset_step), int(row.end_step)),
                    "hand_pose": _slice_episode_field(episode.hand_pose, int(row.onset_step), int(row.end_step)),
                }
                intended_bundle, realized_bundle = infer_instance_key_events(
                    goals=arrays["goals"],
                    piano_states=arrays["piano_states"],
                    events_config=events_config,
                )
                merged = merge_intended_and_realized_events(intended_bundle, realized_bundle)
                conditioning_features = feature_lookup.get(str(row.segment_id))
                if conditioning_features is None:
                    row_payload = row._asdict()
                    conditioning_features, _ = build_feature_vector_from_arrays(
                        row=row_payload,
                        arrays=arrays,
                        config=artifacts.run_config,
                    )
                instances.append(
                    PrimitiveInstance(
                        segment_id=str(row.segment_id),
                        primitive_id=str(row.primitive_id),
                        song_id=str(row.song_id),
                        demo_id=str(row.episode_id),
                        episode_id=str(row.episode_id),
                        split=str(getattr(row, "split", "train") or "train"),
                        start_frame=int(row.onset_step),
                        end_frame=int(row.end_step),
                        duration_steps=max(int(row.duration_steps), 1),
                        control_timestep=float(episode.control_timestep),
                        hand=_optional_string(getattr(row, "hand", None)),
                        start_joint_state=_first_frame(arrays["hand_joints"]),
                        start_joint_velocity=_first_frame(
                            arrays["joint_velocities"]
                            if arrays["joint_velocities"] is not None
                            else _gradient_or_none(arrays["hand_joints"], episode.control_timestep)
                        ),
                        start_fingertip_state=_first_frame(arrays["hand_fingertips"]),
                        intended_keys=intended_bundle.unique_keys,
                        realized_keys_gt=realized_bundle.unique_keys,
                        onset_frames_gt=tuple(int(event.onset_frame) for event in realized_bundle.events),
                        release_frames_gt=tuple(int(event.release_frame) for event in realized_bundle.events),
                        conditioning_features=_as_float_array(conditioning_features),
                        chunk_path=str(getattr(row, "chunk_path", "") or ""),
                        chunk_index=int(getattr(row, "chunk_index", -1)),
                        raw_chunk_path=_optional_string(getattr(row, "raw_chunk_path", None)),
                        raw_chunk_index=_optional_int(getattr(row, "raw_chunk_index", None)),
                        gmr_target_name=str(getattr(row, "gmr_target_name", "actions") or "actions"),
                        primitive_prior_path=_lookup_prior_path(
                            library_df=artifacts.library_df,
                            primitive_id=str(row.primitive_id),
                        ),
                        segment_source=_optional_string(getattr(row, "segment_source", None)),
                        heuristic_family=_optional_string(getattr(row, "heuristic_family", None)),
                        coarse_family=_optional_string(getattr(row, "coarse_family", None)),
                        control_phase=_optional_string(getattr(row, "control_phase", None)),
                        chord_size=_optional_int(getattr(row, "chord_size", None)),
                        key_center=float(getattr(row, "key_center", float("nan"))),
                        intended_events=list(merged["intended_events"]),
                        realized_events_gt=list(merged["realized_events"]),
                        actions_gt=_as_float_array(arrays["actions"]),
                        goals=_as_float_array(arrays["goals"]),
                        piano_states_gt=_as_float_array(arrays["piano_states"]),
                        hand_joints_gt=_as_float_array(arrays["hand_joints"]),
                        hand_fingertips_gt=_as_float_array(arrays["hand_fingertips"]),
                        joint_velocities_gt=_as_float_array(arrays["joint_velocities"]),
                        source_midi_path=_optional_string(manifest_row.get("note_path")),
                        start_piano_state=_first_frame(
                            arrays["piano_states"] if arrays["piano_states"] is not None else arrays["goals"]
                        ),
                        target_source=str(intended_bundle.source),
                        primitive_frame_mode=str(getattr(row, "primitive_frame_mode", "absolute") or "absolute"),
                        relative_wrist_anchor=str(getattr(row, "relative_wrist_anchor", "[]") or "[]"),
                        finger_set=str(getattr(row, "finger_set", getattr(row, "finger_set_id", "unknown")) or "unknown"),
                        target_key_ids=str(getattr(row, "target_key_ids", getattr(row, "target_key_ids_json", "[]")) or "[]"),
                        interval_pattern=str(getattr(row, "interval_pattern", "[]") or "[]"),
                    )
                )
            except Exception as exc:
                failures.append(
                    PrimitiveInstanceBuildFailure(
                        segment_id=str(row.segment_id),
                        primitive_id=str(row.primitive_id),
                        song_id=str(row.song_id),
                        episode_id=str(row.episode_id),
                        error=str(exc),
                    )
                )
    return instances, failures


def infer_instance_key_events(
    *,
    goals: np.ndarray | None,
    piano_states: np.ndarray | None,
    events_config: dict[str, Any],
) -> tuple[KeyEventBundle, KeyEventBundle]:
    use_goals = bool(events_config.get("use_goals", True))
    use_piano_states = bool(events_config.get("use_piano_states", True))
    intended = (
        extract_key_events_from_goals(
            goals,
            key_threshold=float(events_config.get("goal_key_threshold", 0.5)),
            sustain_threshold=float(events_config.get("goal_sustain_threshold", 0.5)),
        )
        if use_goals and goals is not None
        else empty_key_event_bundle(source="goals")
    )
    realized = (
        extract_key_events_from_piano_states(
            piano_states,
            key_threshold=float(events_config.get("piano_state_threshold", 0.5)),
            sustain_threshold=float(events_config.get("piano_sustain_threshold", 0.5)),
        )
        if use_piano_states and piano_states is not None
        else empty_key_event_bundle(source="piano_states")
    )
    if (
        bool(events_config.get("allow_piano_state_intended_fallback", False))
        and not intended.events
        and realized.events
    ):
        intended = KeyEventBundle(
            source="piano_states_as_intended",
            key_roll=realized.key_roll.copy(),
            sustain_roll=realized.sustain_roll.copy(),
            events=list(realized.events),
            unique_keys=tuple(realized.unique_keys),
        )
    if (
        bool(events_config.get("allow_goal_realized_fallback", False))
        and not realized.events
        and intended.events
    ):
        realized = KeyEventBundle(
            source="goals_as_realized",
            key_roll=intended.key_roll.copy(),
            sustain_roll=intended.sustain_roll.copy(),
            events=list(intended.events),
            unique_keys=tuple(intended.unique_keys),
        )
    return intended, realized


def extract_key_events_from_goals(
    goals: np.ndarray | None,
    *,
    key_threshold: float = 0.5,
    sustain_threshold: float = 0.5,
) -> KeyEventBundle:
    return _extract_key_events_from_roll(
        roll=goals,
        key_threshold=key_threshold,
        sustain_threshold=sustain_threshold,
        source="goals",
    )


def extract_key_events_from_piano_states(
    piano_states: np.ndarray | None,
    *,
    key_threshold: float = 0.5,
    sustain_threshold: float = 0.5,
) -> KeyEventBundle:
    return _extract_key_events_from_roll(
        roll=piano_states,
        key_threshold=key_threshold,
        sustain_threshold=sustain_threshold,
        source="piano_states",
    )


def merge_intended_and_realized_events(
    intended: KeyEventBundle,
    realized: KeyEventBundle,
) -> dict[str, Any]:
    intended_keys = set(intended.unique_keys)
    realized_keys = set(realized.unique_keys)
    return {
        "intended_events": list(intended.events),
        "realized_events": list(realized.events),
        "missing_realized_keys": tuple(sorted(intended_keys - realized_keys)),
        "unexpected_realized_keys": tuple(sorted(realized_keys - intended_keys)),
    }


def extract_key_events_from_rollout(
    piano_states: np.ndarray | None,
    *,
    key_threshold: float = 0.5,
    sustain_threshold: float = 0.5,
) -> KeyEventBundle:
    return _extract_key_events_from_roll(
        roll=piano_states,
        key_threshold=key_threshold,
        sustain_threshold=sustain_threshold,
        source="rollout_piano_states",
    )


def match_key_events(
    *,
    predicted_events: Sequence[KeyEvent],
    ground_truth_events: Sequence[KeyEvent],
    onset_tolerance_frames: int,
) -> dict[str, Any]:
    pred = list(predicted_events)
    gt = list(ground_truth_events)
    candidates: list[tuple[int, int, int, int]] = []
    for pred_index, pred_event in enumerate(pred):
        for gt_index, gt_event in enumerate(gt):
            if int(pred_event.key_id) != int(gt_event.key_id):
                continue
            timing_error = int(pred_event.onset_frame) - int(gt_event.onset_frame)
            if abs(timing_error) > int(onset_tolerance_frames):
                continue
            release_gap = abs(pred_event.release_frame - gt_event.release_frame)
            candidates.append((abs(timing_error), release_gap, pred_index, gt_index))
    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))

    matched_pred: set[int] = set()
    matched_gt: set[int] = set()
    timing_errors: list[int] = []
    for _, _, pred_index, gt_index in candidates:
        if pred_index in matched_pred or gt_index in matched_gt:
            continue
        matched_pred.add(pred_index)
        matched_gt.add(gt_index)
        timing_errors.append(pred[pred_index].onset_frame - gt[gt_index].onset_frame)

    true_positives = len(matched_pred)
    false_positive_events = len(pred) - true_positives
    missed_events = len(gt) - true_positives
    precision = _binary_metric(numerator=true_positives, denominator=len(pred))
    recall = _binary_metric(numerator=true_positives, denominator=len(gt))
    f1 = _f1_from_precision_recall(precision, recall)
    unmatched_pred_keys = {int(pred[index].key_id) for index in range(len(pred)) if index not in matched_pred}
    unmatched_gt_keys = {int(gt[index].key_id) for index in range(len(gt)) if index not in matched_gt}
    abs_timing = [abs(item) for item in timing_errors]
    return {
        "true_positives": int(true_positives),
        "false_positive_events": int(false_positive_events),
        "missed_events": int(missed_events),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "timing_errors": timing_errors,
        "mean_abs_timing_error_frames": float(np.mean(abs_timing)) if abs_timing else float("nan"),
        "median_abs_timing_error_frames": float(np.median(abs_timing)) if abs_timing else float("nan"),
        "false_positive_unique_keys": int(len(unmatched_pred_keys)),
        "missed_unique_keys": int(len(unmatched_gt_keys)),
    }


def rollout_primitive_instance(
    *,
    instance: PrimitiveInstance,
    library_entry: PrimitiveLibraryEntry | None,
    runtime: RoboPianistPrimitiveRuntime,
    rollout_config: dict[str, Any],
) -> PrimitiveOnlineRolloutResult:
    causal_config = CausalRolloutConfig.from_mapping(rollout_config.get("causal_eval"))
    restore_mode = str(rollout_config.get("restore_mode", causal_config.restore_mode) or causal_config.restore_mode)
    if restore_mode not in {"unsafe_legacy", "hands_only", "neutral"}:
        return _failed_rollout_result(
            instance=instance,
            source=resolve_instance_rollout_source(
                instance=instance,
                rollout_config=rollout_config,
                robopianist_root=runtime.robopianist_root,
            ),
            status="invalid_restore_mode",
            error=f"Unsupported restore_mode={restore_mode!r}.",
            causal_failure_reason="invalid_restore_mode",
        )
    if restore_mode == "unsafe_legacy":
        causal_config = CausalRolloutConfig.from_mapping({"enabled": False, "restore_mode": "unsafe_legacy"})
        LOGGER.warning(
            "UNSAFE LEGACY PRIMITIVE EVALUATION ENABLED: dataset task/piano/goal state may be restored; "
            "all outputs from this mode are marked causal_validated=false."
        )
    source = resolve_instance_rollout_source(
        instance=instance,
        rollout_config=rollout_config,
        robopianist_root=runtime.robopianist_root,
    )
    if library_entry is None or library_entry.prior_mean is None:
        return _failed_rollout_result(
            instance=instance,
            source=source,
            status="missing_prior",
            error=f"Missing primitive prior for `{instance.primitive_id}`.",
            restore_mode=restore_mode,
            causal_failure_reason="missing_prior",
        )
    if str(instance.gmr_target_name) != "actions":
        return _failed_rollout_result(
            instance=instance,
            source=source,
            status="unsupported_target",
            error=(
                f"Primitive `{instance.primitive_id}` stores `{instance.gmr_target_name}` priors. "
                "Only action-space priors can be replayed online."
            ),
            restore_mode=restore_mode,
            causal_failure_reason="non_action_prior",
        )

    prior_selection = select_primitive_prior(instance=instance, library_entry=library_entry)
    selected_prior = prior_selection["prior_mean"]
    expected_action_dim = int(selected_prior.shape[-1])
    predicted_actions = _resample_sequence(
        np.asarray(selected_prior, dtype=np.float32),
        max(int(instance.duration_steps), 1),
    )
    try:
        env = runtime.get_env(
            environment_name=source["environment_name"],
            midi_file=source["midi_file"],
            control_timestep=float(instance.control_timestep),
            expected_action_dim=expected_action_dim,
        )
        action_dim = int(env.action_spec().shape[0])
        if bool(rollout_config.get("validate_action_dim", True)):
            validate_rollout_action_dim(
                actual_action_dim=action_dim,
                expected_action_dim=expected_action_dim,
                environment_name=str(source["environment_name"]),
            )
        action_diag = _action_diagnostics(predicted_actions, action_dim=action_dim, rollout_config=rollout_config)
        render_video = bool(rollout_config.get("render_video", False))
        rollout_payload = _execute_action_rollout(
            env=env,
            instance=instance,
            actions=predicted_actions,
            action_dim=action_dim,
            restore_mode=restore_mode,
            causal_config=causal_config,
            rollout_config=rollout_config,
            render_video=render_video,
        )
        notes = list(rollout_payload["notes"])
        raw_piano = rollout_payload["raw_piano"]
        raw_hand = rollout_payload["raw_hand"]
        raw_tip = rollout_payload["raw_tip"]
        observed_piano, alignment_mode = align_rollout_sequence(
            observed=raw_piano,
            ground_truth=None if causal_config.enabled else instance.piano_states_gt,
            preferred_mode="post_action" if causal_config.enabled else None,
        )
        observed_hand, _ = align_rollout_sequence(
            observed=raw_hand,
            ground_truth=instance.hand_joints_gt,
            preferred_mode=alignment_mode,
        )
        observed_tip, _ = align_rollout_sequence(
            observed=raw_tip,
            ground_truth=instance.hand_fingertips_gt,
            preferred_mode=alignment_mode,
        )
        observed_bundle = extract_key_events_from_rollout(
            observed_piano,
            key_threshold=float(causal_config.key_activation_threshold),
            sustain_threshold=float(rollout_config.get("piano_sustain_threshold", 0.5)),
        )
        contact_for_metrics = _prepend_initial_contact_frame(
            rollout_payload["contact_roll"],
            steps=raw_piano.shape[0] if raw_piano is not None else 0,
        )
        causal_metrics = contact_gated_keypress_metrics(
            target_key_indices=instance.intended_keys,
            activation_roll=raw_piano,
            contact_roll=contact_for_metrics,
            config=causal_config,
            contact_method=str(rollout_payload["contact_method"]),
        )
        zero_action_metrics: dict[str, Any] = {}
        zero_action_ablation_passed: bool | None = None
        if bool(causal_config.run_zero_action_ablation) and restore_mode != "unsafe_legacy":
            zero_payload = run_zero_action_ablation(
                action_shape=(int(predicted_actions.shape[0]), int(action_dim)),
                rollout_fn=lambda zero_actions: _execute_action_rollout(
                    env=env,
                    instance=instance,
                    actions=zero_actions,
                    action_dim=action_dim,
                    restore_mode=restore_mode,
                    causal_config=causal_config,
                    rollout_config=rollout_config | {"render_video": False},
                    render_video=False,
                ),
            )
            zero_bundle = extract_key_events_from_rollout(
                zero_payload["observed_piano"],
                key_threshold=float(causal_config.key_activation_threshold),
                sustain_threshold=float(rollout_config.get("piano_sustain_threshold", 0.5)),
            )
            zero_keys = sorted({int(event.key_id) for event in zero_bundle.events})
            same_target = bool(set(zero_keys).intersection(set(int(key) for key in instance.intended_keys)))
            zero_action_metrics = {
                "zero_action_pressed_keys": zero_keys,
                "zero_action_key_events": [list(event.as_tuple()) for event in zero_bundle.events],
                "zero_action_any_key_pressed": bool(zero_keys),
                "zero_action_same_target_pressed": same_target,
            }
            zero_action_ablation_passed = not same_target
            if same_target:
                causal_metrics = mark_uncausal_result(
                    causal_metrics,
                    reason="zero_action_pressed_target",
                    status="contaminated_zero_action",
                )
        elif restore_mode == "unsafe_legacy":
            zero_action_ablation_passed = False
        if restore_mode == "unsafe_legacy":
            causal_metrics = mark_uncausal_result(
                causal_metrics,
                reason="unsafe_legacy_eval",
                status="unsafe_legacy_eval",
            )
            notes.append(
                "WARNING: restore_mode=unsafe_legacy restores task/piano/goal state and is not causal-valid."
            )
        video_path = None
        video_format = None
        video_warning = rollout_payload.get("video_warning")
        if render_video and rollout_payload.get("render_frames"):
            final_overlay = {
                "causal_validated": bool(causal_metrics.get("causal_validated", False)),
                "action_max_abs": float(action_diag["action_max_abs"]),
                "initial_active_key_indices": causal_metrics.get("initial_active_key_indices", []),
                "physical_pressed_key_indices": sorted({event.key_id for event in observed_bundle.events}),
                "contact_key_indices": causal_metrics.get("contact_key_indices", []),
                "activation_without_contact_key_indices": causal_metrics.get("activation_without_contact_key_indices", []),
                "zero_action_same_target_pressed": zero_action_metrics.get("zero_action_same_target_pressed"),
            }
            try:
                video_path, video_format, video_warning = _write_primitive_video(
                    frames=_overlay_final_on_frames(
                        rollout_payload["render_frames"],
                        instance=instance,
                        restore_mode=restore_mode,
                        payload=final_overlay,
                    ),
                    instance=instance,
                    rollout_config=rollout_config,
                )
            except Exception as exc:
                video_warning = str(exc)
                notes.append(f"Video export failed: {exc}")
        status = str(causal_metrics.get("status", rollout_payload["status"]))
        if status == "ok":
            status = str(rollout_payload["status"])
        success = bool(causal_metrics.get("contact_gated_success", False))
        if not bool(causal_metrics.get("causal_validated", False)):
            success = False
        return PrimitiveOnlineRolloutResult(
            segment_id=instance.segment_id,
            primitive_id=instance.primitive_id,
            status=status,
            success=success,
            error=rollout_payload["error"],
            restore_mode=restore_mode,
            alignment_mode=alignment_mode,
            predicted_actions=predicted_actions.astype(np.float32),
            raw_observed_piano_states=raw_piano,
            observed_piano_states=observed_piano,
            raw_observed_hand_joints=raw_hand,
            observed_hand_joints=observed_hand,
            raw_observed_hand_fingertips=raw_tip,
            observed_hand_fingertips=observed_tip,
            rollout_source_mode=str(source["source_mode"]),
            rollout_source_label=_optional_string(source.get("source_label")),
            rollout_environment_name=str(source["environment_name"]),
            rollout_midi_path=str(source["midi_file"]) if source.get("midi_file") is not None else None,
            observed_key_events=list(observed_bundle.events),
            observed_key_roll=observed_bundle.key_roll,
            contact_roll=rollout_payload["contact_roll"],
            contact_method=str(causal_metrics.get("contact_method", rollout_payload["contact_method"])),
            causal_metrics=causal_metrics,
            zero_action_metrics=zero_action_metrics,
            causal_validated=bool(causal_metrics.get("causal_validated", False)),
            causal_failure_reason=causal_metrics.get("causal_failure_reason"),
            neutral_start_passed=bool(rollout_payload["neutral_start_passed"]),
            zero_action_ablation_passed=zero_action_ablation_passed,
            contact_gate_passed=bool(causal_metrics.get("contact_gate_passed", False)),
            target_source=instance.target_source,
            observed_source="simulator_piano_activation",
            action_max_abs=float(action_diag["action_max_abs"]),
            hand_action_max_abs=float(action_diag["hand_action_max_abs"]),
            pedal_action_max_abs=float(action_diag["pedal_action_max_abs"]),
            pedal_action_indices=tuple(action_diag["pedal_action_indices"]),
            video_path=video_path,
            video_format=video_format,
            video_warning=video_warning,
            notes=list(notes),
            condition_features_used=tuple(prior_selection["condition_features_used"]),
            condition_feature_norm=float(instance.conditioning_feature_norm),
            conditioned_prior_used=bool(prior_selection["conditioned_prior_used"]),
            unconditional_fallback_used=bool(prior_selection["unconditional_fallback_used"]),
            primitive_frame_mode=str(instance.primitive_frame_mode),
            relative_wrist_anchor=str(instance.relative_wrist_anchor),
            finger_set=str(instance.finger_set),
            target_key_ids=str(instance.target_key_ids),
            interval_pattern=str(instance.interval_pattern),
        )
    except Exception as exc:  # pragma: no cover
        return PrimitiveOnlineRolloutResult(
            segment_id=instance.segment_id,
            primitive_id=instance.primitive_id,
            status="rollout_failed",
            success=False,
            error=str(exc),
            restore_mode="none",
            alignment_mode="none",
            predicted_actions=predicted_actions,
            raw_observed_piano_states=None,
            observed_piano_states=None,
            raw_observed_hand_joints=None,
            observed_hand_joints=None,
            raw_observed_hand_fingertips=None,
            observed_hand_fingertips=None,
            rollout_source_mode=str(source["source_mode"]),
            rollout_source_label=_optional_string(source.get("source_label")),
            rollout_environment_name=str(source["environment_name"]),
            rollout_midi_path=str(source["midi_file"]) if source.get("midi_file") is not None else None,
            causal_validated=False,
            causal_failure_reason=str(exc),
            target_source=instance.target_source,
            condition_features_used=tuple(prior_selection.get("condition_features_used", ())) if "prior_selection" in locals() else (),
            condition_feature_norm=float(instance.conditioning_feature_norm),
            conditioned_prior_used=bool(prior_selection.get("conditioned_prior_used", False)) if "prior_selection" in locals() else False,
            unconditional_fallback_used=bool(prior_selection.get("unconditional_fallback_used", True)) if "prior_selection" in locals() else True,
            primitive_frame_mode=str(instance.primitive_frame_mode),
            relative_wrist_anchor=str(instance.relative_wrist_anchor),
            finger_set=str(instance.finger_set),
            target_key_ids=str(instance.target_key_ids),
            interval_pattern=str(instance.interval_pattern),
        )


def _failed_rollout_result(
    *,
    instance: PrimitiveInstance,
    source: dict[str, Any],
    status: str,
    error: str,
    restore_mode: str = "none",
    causal_failure_reason: str,
) -> PrimitiveOnlineRolloutResult:
    return PrimitiveOnlineRolloutResult(
        segment_id=instance.segment_id,
        primitive_id=instance.primitive_id,
        status=status,
        success=False,
        error=error,
        restore_mode=restore_mode,
        alignment_mode="none",
        predicted_actions=None,
        raw_observed_piano_states=None,
        observed_piano_states=None,
        raw_observed_hand_joints=None,
        observed_hand_joints=None,
        raw_observed_hand_fingertips=None,
        observed_hand_fingertips=None,
        rollout_source_mode=str(source["source_mode"]),
        rollout_source_label=_optional_string(source.get("source_label")),
        rollout_environment_name=str(source["environment_name"]),
        rollout_midi_path=str(source["midi_file"]) if source.get("midi_file") is not None else None,
        causal_validated=False,
        causal_failure_reason=causal_failure_reason,
        target_source=instance.target_source,
        condition_feature_norm=float(instance.conditioning_feature_norm),
        primitive_frame_mode=str(instance.primitive_frame_mode),
        relative_wrist_anchor=str(instance.relative_wrist_anchor),
        finger_set=str(instance.finger_set),
        target_key_ids=str(instance.target_key_ids),
        interval_pattern=str(instance.interval_pattern),
    )


def _execute_action_rollout(
    *,
    env: Any,
    instance: PrimitiveInstance,
    actions: np.ndarray,
    action_dim: int,
    restore_mode: str,
    causal_config: CausalRolloutConfig,
    rollout_config: dict[str, Any],
    render_video: bool,
) -> dict[str, Any]:
    notes: list[str] = []
    env.reset()
    restore_label, restore_notes = restore_instance_state(
        env=env,
        instance=instance,
        restore_mode=restore_mode,
        causal_config=causal_config,
    )
    notes.extend(restore_notes)
    neutral_check = reset_or_validate_neutral_piano(env, causal_config=causal_config)
    if (
        causal_config.enabled
        and causal_config.require_neutral_piano_start
        and causal_config.fail_if_initial_keys_active
        and not neutral_check.passed
    ):
        raw_piano = _stack_frames([_capture_piano_state(env)])
        raw_hand = _stack_frames([_capture_hand_joint_state(env)])
        raw_tip = _stack_frames([_capture_fingertips(env)])
        return {
            "status": "initial_keys_active",
            "error": neutral_check.failure_reason,
            "restore_label": restore_label,
            "notes": notes,
            "raw_piano": raw_piano,
            "observed_piano": raw_piano,
            "raw_hand": raw_hand,
            "raw_tip": raw_tip,
            "contact_roll": None,
            "contact_method": "unavailable",
            "neutral_start_passed": False,
            "render_frames": [],
            "video_warning": None,
        }

    piano_frames = [_capture_piano_state(env)]
    hand_frames = [_capture_hand_joint_state(env)]
    fingertip_frames = [_capture_fingertips(env)]
    contact_frames: list[np.ndarray] = []
    contact_method = "unavailable"
    render_frames: list[np.ndarray] = []
    render_error: str | None = None
    if render_video:
        try:
            render_frames.append(_render_primitive_frame(env, rollout_config=rollout_config, step_index=0, instance=instance, overlay_payload={}))
        except Exception as exc:
            render_error = str(exc)
            notes.append(f"Initial render failed: {exc}")
    status = "completed"
    error = None
    for step_index, action in enumerate(np.asarray(actions, dtype=np.float32), start=1):
        control = np.zeros((int(action_dim),), dtype=np.float32)
        width = min(int(action_dim), int(action.shape[0]))
        control[:width] = action[:width]
        timestep = env.step(control)
        piano_frames.append(_capture_piano_state(env))
        hand_frames.append(_capture_hand_joint_state(env))
        fingertip_frames.append(_capture_fingertips(env))
        contact = collect_fingertip_key_contacts(
            env,
            distance_threshold_m=float(causal_config.fingertip_key_distance_threshold_m),
        )
        if contact.contact_roll is not None:
            contact_frames.append(np.asarray(contact.contact_roll[0], dtype=np.float32))
            if contact_method == "unavailable":
                contact_method = str(contact.contact_method)
        elif contact_method == "unavailable":
            notes.extend(contact.notes)
        if render_video and render_error is None:
            try:
                render_frames.append(
                    _render_primitive_frame(
                        env,
                        rollout_config=rollout_config,
                        step_index=step_index,
                        instance=instance,
                        overlay_payload={},
                    )
                )
            except Exception as exc:
                render_error = str(exc)
                notes.append(f"Render failed: {exc}")
        if timestep.last():
            status = "terminated_early"
            break
    raw_piano = _stack_frames(piano_frames)
    raw_hand = _stack_frames(hand_frames)
    raw_tip = _stack_frames(fingertip_frames)
    observed_piano = raw_piano[1:] if raw_piano is not None and raw_piano.shape[0] > 1 else raw_piano
    contact_roll = _stack_contact_frames(contact_frames, steps=observed_piano.shape[0] if observed_piano is not None else len(contact_frames))
    video_warning = None
    if render_error is not None:
        video_warning = render_error if video_warning is None else f"{video_warning} {render_error}"
    return {
        "status": status,
        "error": error,
        "restore_label": restore_label,
        "notes": notes,
        "raw_piano": raw_piano,
        "observed_piano": observed_piano,
        "raw_hand": raw_hand,
        "raw_tip": raw_tip,
        "contact_roll": contact_roll,
        "contact_method": contact_method,
        "neutral_start_passed": bool(neutral_check.passed),
        "render_frames": render_frames if render_error is None else [],
        "video_warning": video_warning,
    }


def _action_diagnostics(actions: np.ndarray, *, action_dim: int, rollout_config: dict[str, Any]) -> dict[str, Any]:
    array = np.asarray(actions, dtype=np.float32)
    if array.size == 0:
        return {
            "action_max_abs": 0.0,
            "hand_action_max_abs": 0.0,
            "pedal_action_max_abs": 0.0,
            "pedal_action_indices": [],
        }
    pedal_indices = _pedal_action_indices(action_dim=action_dim, rollout_config=rollout_config)
    hand_indices = [index for index in range(min(int(action_dim), array.shape[-1])) if index not in set(pedal_indices)]
    pedal_values = array[:, pedal_indices] if pedal_indices and max(pedal_indices) < array.shape[-1] else np.zeros((array.shape[0], 0), dtype=np.float32)
    hand_values = array[:, hand_indices] if hand_indices else np.zeros((array.shape[0], 0), dtype=np.float32)
    return {
        "action_max_abs": float(np.max(np.abs(array))),
        "hand_action_max_abs": float(np.max(np.abs(hand_values))) if hand_values.size else 0.0,
        "pedal_action_max_abs": float(np.max(np.abs(pedal_values))) if pedal_values.size else 0.0,
        "pedal_action_indices": pedal_indices,
    }


def _pedal_action_indices(*, action_dim: int, rollout_config: dict[str, Any]) -> list[int]:
    configured = rollout_config.get("pedal_action_indices")
    if configured is not None:
        return [int(index) for index in configured if 0 <= int(index) < int(action_dim)]
    if int(action_dim) == 39:
        return [38]
    return []


def _stack_contact_frames(frames: list[np.ndarray], *, steps: int) -> np.ndarray | None:
    if not frames:
        return None
    stacked = np.stack([np.asarray(frame, dtype=np.float32).reshape(-1)[:_NUM_PIANO_KEYS] for frame in frames], axis=0)
    if stacked.shape[0] == int(steps):
        return stacked.astype(np.float32)
    output = np.zeros((int(steps), _NUM_PIANO_KEYS), dtype=np.float32)
    usable = min(output.shape[0], stacked.shape[0])
    output[:usable, : stacked.shape[1]] = stacked[:usable]
    return output


def _prepend_initial_contact_frame(contact_roll: np.ndarray | None, *, steps: int) -> np.ndarray | None:
    if contact_roll is None:
        return None
    array = np.asarray(contact_roll, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    output = np.zeros((int(steps), _NUM_PIANO_KEYS), dtype=np.float32)
    usable = min(max(int(steps) - 1, 0), array.shape[0])
    if usable > 0:
        output[1 : 1 + usable, : min(array.shape[1], _NUM_PIANO_KEYS)] = array[:usable, :_NUM_PIANO_KEYS]
    return output


def _render_primitive_frame(
    env: Any,
    *,
    rollout_config: dict[str, Any],
    step_index: int,
    instance: PrimitiveInstance,
    overlay_payload: dict[str, Any],
) -> np.ndarray:
    from sonata.evaluation.rollout import _render_frame

    frame = _render_frame(
        env,
        height=int(rollout_config.get("video_height", 480)),
        width=int(rollout_config.get("video_width", 640)),
    )
    return _overlay_primitive_debug_text(
        frame,
        lines=[
            f"step={step_index} primitive={instance.primitive_id} segment={instance.segment_id}",
            f"restore={rollout_config.get('restore_mode', rollout_config.get('causal_eval', {}).get('restore_mode', 'hands_only'))}",
            f"target_key_indices={list(instance.intended_keys)}",
            *[f"{key}={value}" for key, value in overlay_payload.items()],
        ],
    )


def _overlay_primitive_debug_text(frame: np.ndarray, *, lines: list[str]) -> np.ndarray:
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return frame
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    draw = ImageDraw.Draw(image, "RGBA")
    height = 8 + 16 * len(lines)
    draw.rectangle((0, 0, image.width, height), fill=(0, 0, 0, 165))
    for index, line in enumerate(lines):
        draw.text((8, 6 + index * 16), str(line)[:160], fill=(255, 255, 255, 255))
    return np.asarray(image, dtype=np.uint8)


def _overlay_final_on_frames(
    frames: list[np.ndarray],
    *,
    instance: PrimitiveInstance,
    restore_mode: str,
    payload: dict[str, Any],
) -> list[np.ndarray]:
    lines = [
        f"primitive_id={instance.primitive_id} segment_id={instance.segment_id} restore_mode={restore_mode}",
        f"causal_validated={payload.get('causal_validated')} action_max_abs={payload.get('action_max_abs')}",
        f"initial_active_key_indices={payload.get('initial_active_key_indices')}",
        f"target_key_indices={list(instance.intended_keys)}",
        f"physical_pressed_key_indices={payload.get('physical_pressed_key_indices')}",
        f"contact_key_indices={payload.get('contact_key_indices')}",
        f"activation_without_contact_key_indices={payload.get('activation_without_contact_key_indices')}",
        f"zero_action_same_target_pressed={payload.get('zero_action_same_target_pressed')}",
    ]
    warning = (
        bool(payload.get("initial_active_key_indices"))
        or bool(payload.get("activation_without_contact_key_indices"))
        or bool(payload.get("zero_action_same_target_pressed"))
        or not bool(payload.get("causal_validated"))
    )
    if warning:
        lines.append("WARNING: causal_validated=false or contact/zero-action contamination detected")
    return [_overlay_primitive_debug_text(frame, lines=lines) for frame in frames]


def _write_primitive_video(
    *,
    frames: list[np.ndarray],
    instance: PrimitiveInstance,
    rollout_config: dict[str, Any],
) -> tuple[str | None, str | None, str | None]:
    from sonata.evaluation.rollout import _relative_output_path, _safe_filename, _write_rollout_video

    videos_dir = Path(str(rollout_config.get("videos_dir", "."))).resolve()
    output_root = videos_dir.parent
    frames_tmp = Path(str(rollout_config.get("frames_tmp_dir", output_root / "frames_tmp"))).resolve()
    video_path, video_format, warning = _write_rollout_video(
        frames=frames,
        output_path=videos_dir / f"{_safe_filename(instance.primitive_id)}_{_safe_filename(instance.segment_id)}.mp4",
        fps=int(rollout_config.get("video_fps", 20)),
        temp_root=frames_tmp,
        audio_events=None,
    )
    return _relative_output_path(video_path, output_root), video_format, warning


def restore_instance_state(env: Any, instance: PrimitiveInstance, *, restore_mode: str, causal_config: CausalRolloutConfig) -> tuple[str, list[str]]:
    if restore_mode == "unsafe_legacy":
        return restore_instance_state_unsafe_legacy(env=env, instance=instance)
    notes: list[str] = []
    task = getattr(env, "task", None)
    physics = getattr(env, "physics", None)
    if task is None or physics is None:
        return "env_reset_only", ["Environment does not expose task/physics handles for state restoration."]
    if restore_mode == "neutral":
        return "neutral", notes
    if restore_mode == "hands_only":
        forbid_dataset_piano_state_restore(
            restore_mode=restore_mode,
            causal_config=causal_config,
            source="piano_state",
        )
        if _restore_hand_state(task=task, physics=physics, instance=instance):
            if hasattr(physics, "forward"):
                physics.forward()
            return "hands_only", notes
        return "hands_only_unavailable", ["Hand joint state was unavailable or incompatible; used env reset state."]
    raise ValueError(f"Unsupported restore_mode={restore_mode!r}")


def restore_instance_state_unsafe_legacy(env: Any, instance: PrimitiveInstance) -> tuple[str, list[str]]:
    notes: list[str] = []
    task = getattr(env, "task", None)
    physics = getattr(env, "physics", None)
    if task is None or physics is None:
        return "env_reset_only", ["Environment does not expose task/physics handles for state restoration."]

    restore_mode = "env_reset_only"
    note_count = len(getattr(task, "_notes", []))
    if hasattr(task, "_t_idx"):
        target_idx = int(instance.start_frame)
        if note_count > 0:
            clamped = int(np.clip(target_idx, 0, note_count - 1))
            if clamped != target_idx:
                notes.append(
                    f"Clamped task timestep from {target_idx} to {clamped} because the MIDI note sequence was shorter."
                )
            target_idx = clamped
        task._t_idx = target_idx
        restore_mode = "task_timestep"

    if hasattr(task, "_should_terminate"):
        task._should_terminate = False
    if hasattr(task, "_discount"):
        task._discount = 1.0

    if _restore_hand_state(task=task, physics=physics, instance=instance):
        restore_mode = "direct_hand_state_restore"
    if _restore_piano_state_unsafe_legacy(task=task, physics=physics, instance=instance):
        restore_mode = "direct_state_restore"
    if hasattr(physics, "forward"):
        physics.forward()
    piano = getattr(task, "piano", None)
    if piano is not None and hasattr(piano, "_update_key_state"):
        piano._update_key_state(physics)
        if hasattr(piano, "_update_key_color"):
            piano._update_key_color(physics)
    if hasattr(task, "_update_goal_state"):
        task._update_goal_state()
        goal_state = getattr(task, "_goal_state", None)
        if goal_state is not None and np.asarray(goal_state).size:
            task._goal_current = np.asarray(goal_state[0], dtype=np.float64)
    return restore_mode, notes


def build_instance_result_row(
    *,
    instance: PrimitiveInstance,
    rollout: PrimitiveOnlineRolloutResult,
    event_config: dict[str, Any],
    debug_artifact_path: Path | None,
) -> dict[str, Any]:
    realized_metrics = _empty_event_metrics()
    intended_metrics = _empty_event_metrics()
    if rollout.observed_piano_states is not None:
        realized_metrics = match_key_events(
            predicted_events=rollout.observed_key_events,
            ground_truth_events=instance.realized_events_gt,
            onset_tolerance_frames=int(event_config.get("onset_tolerance_frames", 1)),
        )
        intended_metrics = match_key_events(
            predicted_events=rollout.observed_key_events,
            ground_truth_events=instance.intended_events,
            onset_tolerance_frames=int(event_config.get("onset_tolerance_frames", 1)),
        )

    piano_state_mse = _aligned_mse(rollout.observed_piano_states, instance.piano_states_gt)
    hand_joint_mse = _aligned_mse(rollout.observed_hand_joints, instance.hand_joints_gt)
    fingertip_mse = _aligned_mse(rollout.observed_hand_fingertips, instance.hand_fingertips_gt)
    action_mse = _aligned_mse(rollout.predicted_actions, instance.actions_gt)
    observed_key_count = len({event.key_id for event in rollout.observed_key_events})
    causal_metrics = dict(rollout.causal_metrics or {})
    zero_metrics = dict(rollout.zero_action_metrics or {})
    intended_key_center = _mean_key_center(instance.intended_keys)
    realized_key_center = _mean_key_center(instance.realized_keys_gt)
    predicted_key_center = _mean_key_center(tuple(sorted({event.key_id for event in rollout.observed_key_events})))
    invalid_motion_flag = _invalid_motion_flag(
        rollout.observed_piano_states,
        rollout.observed_hand_joints,
        rollout.observed_hand_fingertips,
    )
    toggle_count = _activation_toggle_count(rollout.observed_key_roll)
    row = {
        "segment_id": instance.segment_id,
        "primitive_id": instance.primitive_id,
        "song_id": instance.song_id,
        "episode_id": instance.episode_id,
        "demo_id": instance.demo_id,
        "split": instance.split,
        "start_frame": int(instance.start_frame),
        "end_frame": int(instance.end_frame),
                        "duration_steps": int(instance.duration_steps),
                        "control_timestep": float(instance.control_timestep),
                        "hand": instance.hand or "unknown",
                        "segment_source": instance.segment_source,
                        "heuristic_family": instance.heuristic_family,
                        "coarse_family": instance.coarse_family,
                        "control_phase": instance.control_phase,
                        "chord_size": instance.chord_size,
                        "key_center": instance.key_center,
                        "chunk_path": instance.chunk_path,
                        "chunk_index": int(instance.chunk_index),
        "raw_chunk_path": instance.raw_chunk_path,
        "raw_chunk_index": instance.raw_chunk_index,
        "gmr_target_name": instance.gmr_target_name,
        "primitive_prior_path": instance.primitive_prior_path,
        "conditioning_feature_norm": float(instance.conditioning_feature_norm),
        "conditioning_feature_dim": int(instance.conditioning_features.shape[0]) if instance.conditioning_features is not None else 0,
        "condition_features_used": json.dumps(list(rollout.condition_features_used)),
        "condition_feature_norm": float(rollout.condition_feature_norm),
        "conditioned_prior_used": bool(rollout.conditioned_prior_used),
        "unconditional_fallback_used": bool(rollout.unconditional_fallback_used),
        "primitive_frame_mode": rollout.primitive_frame_mode,
        "relative_wrist_anchor": rollout.relative_wrist_anchor,
        "finger_set": rollout.finger_set,
        "target_key_ids": rollout.target_key_ids,
        "interval_pattern": rollout.interval_pattern,
        "intended_key_count": int(len(instance.intended_keys)),
        "realized_key_count_gt": int(len(instance.realized_keys_gt)),
        "predicted_key_count": int(observed_key_count),
        "intended_event_count": int(len(instance.intended_events)),
        "realized_event_count_gt": int(len(instance.realized_events_gt)),
        "predicted_event_count": int(len(rollout.observed_key_events)),
        "intended_key_center": float(intended_key_center),
        "realized_key_center_gt": float(realized_key_center),
        "predicted_key_center": float(predicted_key_center),
        "status": rollout.status,
        "success": bool(rollout.success),
        "error": rollout.error,
        "restore_mode": rollout.restore_mode,
        "alignment_mode": rollout.alignment_mode,
        "rollout_source_mode": rollout.rollout_source_mode,
        "rollout_source_label": rollout.rollout_source_label,
        "rollout_environment_name": rollout.rollout_environment_name,
        "rollout_midi_path": rollout.rollout_midi_path,
        "onset_precision": float(realized_metrics["precision"]),
        "onset_recall": float(realized_metrics["recall"]),
        "onset_f1": float(realized_metrics["f1"]),
        "legacy_state_onset_precision": float(realized_metrics["precision"]),
        "legacy_state_onset_recall": float(realized_metrics["recall"]),
        "legacy_state_onset_f1": float(realized_metrics["f1"]),
        "state_onset_f1": float(realized_metrics["f1"]),
        "false_positive_key_events": int(realized_metrics["false_positive_events"]),
        "missed_key_events": int(realized_metrics["missed_events"]),
        "false_positive_unique_keys": int(realized_metrics["false_positive_unique_keys"]),
        "missed_unique_keys": int(realized_metrics["missed_unique_keys"]),
        "matched_onset_count": int(realized_metrics["true_positives"]),
        "mean_abs_timing_error_frames": float(realized_metrics["mean_abs_timing_error_frames"]),
        "median_abs_timing_error_frames": float(realized_metrics["median_abs_timing_error_frames"]),
        "intended_onset_precision": float(intended_metrics["precision"]),
        "intended_onset_recall": float(intended_metrics["recall"]),
        "intended_onset_f1": float(intended_metrics["f1"]),
        "intended_false_positive_key_events": int(intended_metrics["false_positive_events"]),
        "intended_missed_key_events": int(intended_metrics["missed_events"]),
        "piano_state_mse": float(piano_state_mse),
        "hand_joint_mse": float(hand_joint_mse),
        "fingertip_mse": float(fingertip_mse),
        "action_mse": float(action_mse),
        "activation_toggle_count": int(toggle_count),
        "invalid_motion_flag": bool(invalid_motion_flag),
        "rollout_diverged": bool(not rollout.success or rollout.status not in {"completed", "terminated_early"}),
        "causal_validated": bool(rollout.causal_validated),
        "causal_failure_reason": rollout.causal_failure_reason,
        "zero_action_ablation_passed": rollout.zero_action_ablation_passed,
        "neutral_start_passed": bool(rollout.neutral_start_passed),
        "contact_gate_passed": bool(rollout.contact_gate_passed),
        "target_source": rollout.target_source,
        "observed_source": rollout.observed_source,
        "causal_true_positive_events": int(causal_metrics.get("causal_true_positive_events", 0)),
        "causal_false_positive_events": int(causal_metrics.get("causal_false_positive_events", 0)),
        "causal_missed_events": int(causal_metrics.get("causal_missed_events", 0)),
        "causal_precision": float(causal_metrics.get("causal_precision", float("nan"))),
        "causal_recall": float(causal_metrics.get("causal_recall", float("nan"))),
        "causal_f1": float(causal_metrics.get("causal_f1", float("nan"))),
        "contact_gated_success": bool(causal_metrics.get("contact_gated_success", False)),
        "contact_method": str(causal_metrics.get("contact_method", rollout.contact_method)),
        "action_max_abs": float(rollout.action_max_abs),
        "hand_action_max_abs": float(rollout.hand_action_max_abs),
        "pedal_action_max_abs": float(rollout.pedal_action_max_abs),
        "video_path": rollout.video_path,
        "video_format": rollout.video_format,
        "video_warning": rollout.video_warning,
        "debug_artifact_path": str(debug_artifact_path.resolve()) if debug_artifact_path is not None else None,
        "source_midi_path": instance.source_midi_path,
        "target_key_indices_json": json.dumps(list(instance.intended_keys)),
        "physical_pressed_key_indices_json": json.dumps(sorted({event.key_id for event in rollout.observed_key_events})),
        "contact_key_indices_json": json.dumps(list(causal_metrics.get("contact_key_indices", []))),
        "activation_key_indices_json": json.dumps(list(causal_metrics.get("activation_key_indices", []))),
        "initial_active_key_indices_json": json.dumps(list(causal_metrics.get("initial_active_key_indices", []))),
        "activation_without_contact_key_indices_json": json.dumps(
            list(causal_metrics.get("activation_without_contact_key_indices", []))
        ),
        "contact_without_activation_key_indices_json": json.dumps(
            list(causal_metrics.get("contact_without_activation_key_indices", []))
        ),
        "robot_midi_key_indices_json": json.dumps([]),
        "reference_midi_key_indices_json": json.dumps(list(instance.intended_keys)),
        "zero_action_pressed_keys_json": json.dumps(list(zero_metrics.get("zero_action_pressed_keys", []))),
        "zero_action_key_events_json": json.dumps(list(zero_metrics.get("zero_action_key_events", []))),
        "zero_action_any_key_pressed": bool(zero_metrics.get("zero_action_any_key_pressed", False)),
        "zero_action_same_target_pressed": bool(zero_metrics.get("zero_action_same_target_pressed", False)),
        "primitive_passes_zero_action_control": bool(rollout.zero_action_ablation_passed)
        if rollout.zero_action_ablation_passed is not None
        else None,
        "pedal_action_indices_json": json.dumps(list(rollout.pedal_action_indices)),
        "intended_keys_json": json.dumps(list(instance.intended_keys)),
        "realized_keys_gt_json": json.dumps(list(instance.realized_keys_gt)),
        "predicted_keys_json": json.dumps(sorted({event.key_id for event in rollout.observed_key_events})),
        "intended_events_json": json.dumps([list(event.as_tuple()) for event in instance.intended_events]),
        "realized_events_gt_json": json.dumps([list(event.as_tuple()) for event in instance.realized_events_gt]),
        "predicted_events_json": json.dumps([list(event.as_tuple()) for event in rollout.observed_key_events]),
        "notes_json": json.dumps(list(rollout.notes)),
    }
    return row


def aggregate_per_primitive_reports(result_df: pd.DataFrame) -> pd.DataFrame:
    if result_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for primitive_id, group in result_df.groupby("primitive_id", sort=True):
        usable = group.loc[group["status"].astype(str).isin({"completed", "terminated_early"})].copy()
        causal_usable = group.loc[group.get("causal_validated", pd.Series(False, index=group.index)).astype(bool)].copy()
        representative_success = _representative_segments(causal_usable, metric="causal_f1", ascending=False, top_k=3)
        representative_failure = _representative_failures(group, top_k=3)
        intended_hist = _value_histogram_json(group["intended_key_count"].tolist())
        chord_hist = _value_histogram_json(
            group["chord_size"].tolist() if "chord_size" in group.columns else group["realized_key_count_gt"].tolist()
        )
        mean_center = float(pd.to_numeric(group["intended_key_center"], errors="coerce").mean())
        rows.append(
            {
                "primitive_id": str(primitive_id),
                "num_instances": int(len(group)),
                "num_successful_rollouts": int(group.get("success", pd.Series(False, index=group.index)).astype(bool).sum()),
                "num_failed_rollouts": int(len(group) - group.get("success", pd.Series(False, index=group.index)).astype(bool).sum()),
                "num_causal_validated": int(len(causal_usable)),
                "mean_legacy_state_onset_precision": float(pd.to_numeric(usable["legacy_state_onset_precision"], errors="coerce").mean()) if not usable.empty and "legacy_state_onset_precision" in usable else float("nan"),
                "mean_legacy_state_onset_recall": float(pd.to_numeric(usable["legacy_state_onset_recall"], errors="coerce").mean()) if not usable.empty and "legacy_state_onset_recall" in usable else float("nan"),
                "mean_legacy_state_onset_f1": float(pd.to_numeric(usable["legacy_state_onset_f1"], errors="coerce").mean()) if not usable.empty and "legacy_state_onset_f1" in usable else float("nan"),
                "mean_causal_precision": float(pd.to_numeric(causal_usable["causal_precision"], errors="coerce").mean()) if not causal_usable.empty and "causal_precision" in causal_usable else float("nan"),
                "mean_causal_recall": float(pd.to_numeric(causal_usable["causal_recall"], errors="coerce").mean()) if not causal_usable.empty and "causal_recall" in causal_usable else float("nan"),
                "mean_causal_f1": float(pd.to_numeric(causal_usable["causal_f1"], errors="coerce").mean()) if not causal_usable.empty and "causal_f1" in causal_usable else float("nan"),
                "median_causal_f1": float(pd.to_numeric(causal_usable["causal_f1"], errors="coerce").median()) if not causal_usable.empty and "causal_f1" in causal_usable else float("nan"),
                "false_positive_rate": float(pd.to_numeric(usable["false_positive_key_events"], errors="coerce").mean()) if not usable.empty else float("nan"),
                "missed_note_rate": float(pd.to_numeric(usable["missed_key_events"], errors="coerce").mean()) if not usable.empty else float("nan"),
                "mean_abs_timing_error_frames": float(pd.to_numeric(usable["mean_abs_timing_error_frames"], errors="coerce").mean()) if not usable.empty else float("nan"),
                "typical_duration_steps": float(pd.to_numeric(group["duration_steps"], errors="coerce").median()),
                "mean_intended_key_count": float(pd.to_numeric(group["intended_key_count"], errors="coerce").mean()),
                "mean_realized_key_count_gt": float(pd.to_numeric(group["realized_key_count_gt"], errors="coerce").mean()),
                "mean_predicted_key_count": float(pd.to_numeric(causal_usable["predicted_key_count"], errors="coerce").mean()) if not causal_usable.empty else float("nan"),
                "keyboard_region_summary": _keyboard_region_label(mean_center),
                "hand_summary": _mode_or_unknown(group.get("hand", pd.Series(dtype=object))),
                "intended_key_count_histogram_json": intended_hist,
                "chord_size_histogram_json": chord_hist,
                "behavior_hint": infer_primitive_behavior_hint(group),
                "representative_success_examples_json": json.dumps(representative_success),
                "representative_failure_examples_json": json.dumps(representative_failure),
                "representative_best_segment_id": representative_success[0] if representative_success else None,
                "representative_worst_segment_id": representative_failure[0] if representative_failure else None,
            }
        )
    return pd.DataFrame(rows).sort_values("primitive_id", kind="stable").reset_index(drop=True)


def build_aggregate_metrics(
    *,
    result_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    preflight: dict[str, Any],
    runtime_error: str | None,
) -> dict[str, Any]:
    usable = result_df.loc[result_df["status"].astype(str).isin({"completed", "terminated_early"})].copy()
    causal_usable = result_df.loc[result_df.get("causal_validated", pd.Series(False, index=result_df.index)).astype(bool)].copy()
    legacy_state_metrics = {
        "mean_onset_precision": float(pd.to_numeric(usable.get("legacy_state_onset_precision", pd.Series(dtype=float)), errors="coerce").mean()) if not usable.empty else float("nan"),
        "mean_onset_recall": float(pd.to_numeric(usable.get("legacy_state_onset_recall", pd.Series(dtype=float)), errors="coerce").mean()) if not usable.empty else float("nan"),
        "mean_onset_f1": float(pd.to_numeric(usable.get("legacy_state_onset_f1", pd.Series(dtype=float)), errors="coerce").mean()) if not usable.empty else float("nan"),
        "mean_false_positive_key_events": float(pd.to_numeric(usable["false_positive_key_events"], errors="coerce").mean()) if not usable.empty else float("nan"),
        "mean_missed_key_events": float(pd.to_numeric(usable["missed_key_events"], errors="coerce").mean()) if not usable.empty else float("nan"),
    }
    causal_contact_metrics = {
        "num_causal_validated": int(len(causal_usable)),
        "mean_causal_precision": float(pd.to_numeric(causal_usable.get("causal_precision", pd.Series(dtype=float)), errors="coerce").mean()) if not causal_usable.empty else float("nan"),
        "mean_causal_recall": float(pd.to_numeric(causal_usable.get("causal_recall", pd.Series(dtype=float)), errors="coerce").mean()) if not causal_usable.empty else float("nan"),
        "mean_causal_f1": float(pd.to_numeric(causal_usable.get("causal_f1", pd.Series(dtype=float)), errors="coerce").mean()) if not causal_usable.empty else float("nan"),
        "causal_success_rate": float(causal_usable.get("success", pd.Series(False, index=causal_usable.index)).astype(bool).mean()) if not causal_usable.empty else float("nan"),
    }
    return {
        "status": "completed",
        "runtime_error": runtime_error,
        "num_assignment_rows": int(preflight.get("num_assignment_rows", 0)),
        "num_sampled_rows": int(preflight.get("num_sampled_rows", 0)),
        "num_result_rows": int(len(result_df)),
        "num_successful_rollouts": int(result_df.get("success", pd.Series(False, index=result_df.index)).astype(bool).sum()),
        "num_failed_rollouts": int(len(result_df) - result_df.get("success", pd.Series(False, index=result_df.index)).astype(bool).sum()),
        "num_causal_validated": int(len(causal_usable)),
        "num_primitives_evaluated": int(summary_df["primitive_id"].nunique()) if not summary_df.empty else 0,
        "legacy_state_metrics": legacy_state_metrics,
        "causal_contact_metrics": causal_contact_metrics,
        "mean_causal_f1": causal_contact_metrics["mean_causal_f1"],
        "mean_abs_timing_error_frames": float(pd.to_numeric(usable["mean_abs_timing_error_frames"], errors="coerce").mean()) if not usable.empty else float("nan"),
        "mean_piano_state_mse": float(pd.to_numeric(usable["piano_state_mse"], errors="coerce").mean()) if not usable.empty else float("nan"),
        "mean_action_mse": float(pd.to_numeric(usable["action_mse"], errors="coerce").mean()) if not usable.empty else float("nan"),
    }


def load_primitive_library_lookup(
    *,
    library_df: pd.DataFrame,
    primitive_ids: Sequence[str],
    primitive_root: Path | None = None,
) -> dict[str, PrimitiveLibraryEntry]:
    if library_df.empty:
        return {}
    selected = set(str(item) for item in primitive_ids)
    lookup: dict[str, PrimitiveLibraryEntry] = {}
    for row in library_df.itertuples(index=False):
        primitive_id = str(row.primitive_id)
        if selected and primitive_id not in selected:
            continue
        prior_path = Path(str(getattr(row, "prior_path", ""))).resolve() if getattr(row, "prior_path", "") else None
        if primitive_root is not None:
            local_prior = primitive_root / "library" / f"{primitive_id}_prior.npz"
            if local_prior.exists() and (prior_path is None or not prior_path.exists()):
                prior_path = local_prior.resolve()
        prior_mean = None
        prototype_means = None
        prototype_latent_centroids = None
        prototype_weights = None
        condition_feature_names: tuple[str, ...] = ()
        condition_feature_indices: tuple[int, ...] = ()
        condition_mean = None
        condition_std = None
        condition_regression_coef = None
        condition_regression_intercept = None
        supports_conditioned_rollout = bool(getattr(row, "supports_conditioned_rollout", False) or False)
        fallback_unconditional_available = bool(getattr(row, "fallback_unconditional_available", True))
        default_prototype_index = int(getattr(row, "default_prototype_index", 0) or 0)
        if prior_path is not None and prior_path.exists():
            payload = np.load(prior_path, allow_pickle=True)
            prior_mean = np.asarray(payload["prior_mean"], dtype=np.float32)
            if "prototype_means" in payload:
                prototype_means = np.asarray(payload["prototype_means"], dtype=np.float32)
            if "prototype_latent_centroids" in payload:
                prototype_latent_centroids = np.asarray(payload["prototype_latent_centroids"], dtype=np.float32)
            if "prototype_weights" in payload:
                prototype_weights = np.asarray(payload["prototype_weights"], dtype=np.float32)
            if "condition_feature_names" in payload:
                condition_feature_names = tuple(str(item) for item in np.asarray(payload["condition_feature_names"], dtype=object).reshape(-1).tolist())
            if "condition_feature_indices" in payload:
                condition_feature_indices = tuple(int(item) for item in np.asarray(payload["condition_feature_indices"], dtype=np.int64).reshape(-1).tolist())
            if "condition_mean" in payload:
                condition_mean = np.asarray(payload["condition_mean"], dtype=np.float32)
            if "condition_std" in payload:
                condition_std = np.asarray(payload["condition_std"], dtype=np.float32)
            if "condition_regression_coef" in payload:
                condition_regression_coef = np.asarray(payload["condition_regression_coef"], dtype=np.float32)
            if "condition_regression_intercept" in payload:
                condition_regression_intercept = np.asarray(payload["condition_regression_intercept"], dtype=np.float32)
            if "supports_conditioned_rollout" in payload:
                supports_conditioned_rollout = bool(np.asarray(payload["supports_conditioned_rollout"]).reshape(-1)[0])
            if "fallback_unconditional_available" in payload:
                fallback_unconditional_available = bool(np.asarray(payload["fallback_unconditional_available"]).reshape(-1)[0])
        lookup[primitive_id] = PrimitiveLibraryEntry(
            primitive_id=primitive_id,
            prior_path=prior_path,
            prior_mean=prior_mean,
            prototype_means=prototype_means,
            prototype_latent_centroids=prototype_latent_centroids,
            prototype_weights=prototype_weights,
            default_prototype_index=default_prototype_index,
            metadata=row._asdict(),
            condition_feature_names=condition_feature_names,
            condition_feature_indices=condition_feature_indices,
            condition_mean=condition_mean,
            condition_std=condition_std,
            condition_regression_coef=condition_regression_coef,
            condition_regression_intercept=condition_regression_intercept,
            supports_conditioned_rollout=supports_conditioned_rollout,
            fallback_unconditional_available=fallback_unconditional_available,
        )
    return lookup


def select_primitive_prior(*, instance: PrimitiveInstance, library_entry: PrimitiveLibraryEntry) -> dict[str, Any]:
    if (
        library_entry.supports_conditioned_rollout
        and instance.conditioning_features is not None
        and library_entry.condition_mean is not None
        and library_entry.condition_std is not None
        and library_entry.condition_regression_coef is not None
        and library_entry.condition_regression_intercept is not None
    ):
        if library_entry.condition_feature_indices:
            valid_indices = [index for index in library_entry.condition_feature_indices if 0 <= int(index) < int(instance.conditioning_features.shape[0])]
            condition_source = np.asarray(instance.conditioning_features[np.asarray(valid_indices, dtype=np.int64)], dtype=np.float32)
        else:
            condition_source = np.asarray(instance.conditioning_features, dtype=np.float32)
        feature_dim = min(int(condition_source.shape[0]), int(library_entry.condition_mean.shape[0]), int(library_entry.condition_std.shape[0]))
        if feature_dim > 0:
            condition = np.asarray(condition_source[:feature_dim], dtype=np.float32)
            mean = np.asarray(library_entry.condition_mean[:feature_dim], dtype=np.float32)
            std = np.clip(np.asarray(library_entry.condition_std[:feature_dim], dtype=np.float32), 1e-6, None)
            normalized = (condition - mean) / std
            coef = np.asarray(library_entry.condition_regression_coef, dtype=np.float32)
            intercept = np.asarray(library_entry.condition_regression_intercept, dtype=np.float32)
            if coef.ndim == 2 and coef.shape[1] >= feature_dim and intercept.size == coef.shape[0]:
                flat = intercept + coef[:, :feature_dim] @ normalized
                if library_entry.prior_mean is not None:
                    prior_shape = np.asarray(library_entry.prior_mean).shape
                    if flat.size == int(np.prod(prior_shape)):
                        return {
                            "prior_mean": flat.reshape(prior_shape).astype(np.float32),
                            "condition_features_used": library_entry.condition_feature_names[:feature_dim],
                            "conditioned_prior_used": True,
                            "unconditional_fallback_used": False,
                        }
    if not library_entry.fallback_unconditional_available:
        raise ValueError(f"Primitive `{library_entry.primitive_id}` cannot condition rollout and has no unconditional fallback.")
    return {
        "prior_mean": _select_unconditional_or_prototype_prior(instance=instance, library_entry=library_entry),
        "condition_features_used": (),
        "conditioned_prior_used": False,
        "unconditional_fallback_used": True,
    }


def select_primitive_prior_mean(*, instance: PrimitiveInstance, library_entry: PrimitiveLibraryEntry) -> np.ndarray:
    return np.asarray(select_primitive_prior(instance=instance, library_entry=library_entry)["prior_mean"], dtype=np.float32)


def _select_unconditional_or_prototype_prior(*, instance: PrimitiveInstance, library_entry: PrimitiveLibraryEntry) -> np.ndarray:
    if library_entry.prototype_means is None or library_entry.prototype_means.size == 0:
        if library_entry.prior_mean is None:
            raise ValueError(f"Primitive `{library_entry.primitive_id}` is missing a prior mean.")
        return np.asarray(library_entry.prior_mean, dtype=np.float32)
    prototype_means = np.asarray(library_entry.prototype_means, dtype=np.float32)
    if (
        instance.conditioning_features is None
        or library_entry.prototype_latent_centroids is None
        or library_entry.prototype_latent_centroids.size == 0
    ):
        index = int(np.clip(library_entry.default_prototype_index, 0, prototype_means.shape[0] - 1))
        return prototype_means[index]
    centroids = np.asarray(library_entry.prototype_latent_centroids, dtype=np.float32)
    feature_dim = min(int(instance.conditioning_features.shape[0]), int(centroids.shape[1]))
    if feature_dim <= 0:
        index = int(np.clip(library_entry.default_prototype_index, 0, prototype_means.shape[0] - 1))
        return prototype_means[index]
    feature_slice = np.asarray(instance.conditioning_features[:feature_dim], dtype=np.float32)
    centroid_slice = centroids[:, :feature_dim]
    distances = np.linalg.norm(centroid_slice - feature_slice[None, :], axis=1)
    index = int(np.argmin(distances))
    index = int(np.clip(index, 0, prototype_means.shape[0] - 1))
    return prototype_means[index]


def resolve_instance_rollout_source(
    *,
    instance: PrimitiveInstance,
    rollout_config: dict[str, Any] | None = None,
    robopianist_root: str | Path | None = None,
) -> dict[str, Any]:
    rollout_config = dict(rollout_config or {})
    source_mode = str(rollout_config.get("source_mode", "dataset_song") or "dataset_song")
    if source_mode == "example_midi_pool":
        example_sources = _resolve_example_midi_sources(
            rollout_config=rollout_config,
            robopianist_root=robopianist_root,
        )
        if not example_sources:
            raise FileNotFoundError(
                "No example MIDI files were found for primitive online evaluation. "
                "Provide rollout.example_midi_paths or install the Rousseau example MIDI files."
            )
        source = example_sources[_stable_index(instance.primitive_id, len(example_sources))]
        return {
            "environment_name": str(source["environment_name"]),
            "midi_file": Path(source["midi_file"]).resolve(),
            "source_mode": source_mode,
            "source_label": str(source["label"]),
        }
    midi_file = Path(instance.source_midi_path).resolve() if instance.source_midi_path else None
    if midi_file is not None and not midi_file.exists():
        midi_file = None
    environment_name = _ENV_SUFFIX_RE.sub(r"\1", str(instance.song_id))
    return {
        "environment_name": environment_name,
        "midi_file": midi_file,
        "source_mode": source_mode,
        "source_label": None,
    }


def _resolve_example_midi_sources(
    *,
    rollout_config: dict[str, Any],
    robopianist_root: str | Path | None,
) -> list[dict[str, Any]]:
    package_root = _resolve_robopianist_package_root(robopianist_root)
    configured_paths = [str(item).strip() for item in rollout_config.get("example_midi_paths", []) if str(item).strip()]
    configured_env_names = [
        str(item).strip() for item in rollout_config.get("example_environment_names", []) if str(item).strip()
    ]
    configured_labels = [str(item).strip() for item in rollout_config.get("example_midi_labels", []) if str(item).strip()]
    configured_globs = [str(item).strip() for item in rollout_config.get("example_midi_globs", []) if str(item).strip()]
    max_example_midis = int(rollout_config.get("max_example_midis", 0) or 0)
    sources: list[dict[str, Any]] = []
    if configured_paths:
        for index, item in enumerate(configured_paths):
            midi_path = Path(os.path.expandvars(str(item))).expanduser()
            if not midi_path.is_absolute():
                midi_path = (package_root / midi_path).resolve()
            if not midi_path.exists():
                continue
            sources.append(
                {
                    "label": configured_labels[index] if index < len(configured_labels) else midi_path.stem,
                    "environment_name": (
                        configured_env_names[index]
                        if index < len(configured_env_names)
                        else f"RoboPianist-debug-{midi_path.stem}-v0"
                    ),
                    "midi_file": midi_path,
                }
            )
        return sources
    if configured_globs:
        matched_paths: list[Path] = []
        for pattern in configured_globs:
            expanded = Path(os.path.expandvars(str(pattern))).expanduser()
            if not expanded.is_absolute():
                expanded = (package_root / expanded).resolve()
            matched_paths.extend(Path(path).resolve() for path in sorted(glob(str(expanded), recursive=True)))
        deduped_paths: list[Path] = []
        seen: set[str] = set()
        for midi_path in matched_paths:
            if str(midi_path) in seen or not midi_path.exists():
                continue
            seen.add(str(midi_path))
            deduped_paths.append(midi_path)
        if max_example_midis > 0:
            deduped_paths = deduped_paths[:max_example_midis]
        for index, midi_path in enumerate(deduped_paths):
            sources.append(
                {
                    "label": configured_labels[index] if index < len(configured_labels) else midi_path.stem,
                    "environment_name": (
                        configured_env_names[index]
                        if index < len(configured_env_names)
                        else f"RoboPianist-debug-{midi_path.stem}-v0"
                    ),
                    "midi_file": midi_path,
                }
            )
        return sources
    for item in _DEFAULT_EXAMPLE_MIDI_SOURCES:
        midi_path = (package_root / item["relative_path"]).resolve()
        if not midi_path.exists():
            continue
        sources.append(
            {
                "label": str(item["label"]),
                "environment_name": str(item["environment_name"]),
                "midi_file": midi_path,
            }
        )
    return sources


def _resolve_robopianist_package_root(robopianist_root: str | Path | None) -> Path:
    candidates: list[Path] = []
    if robopianist_root is not None:
        candidates.append(Path(robopianist_root).expanduser().resolve())
    candidates.append(Path(__file__).resolve().parents[4] / "robopianist")
    for candidate in candidates:
        if (candidate / "__init__.py").exists() and (candidate / "music").exists():
            return candidate
        package_root = candidate / "robopianist"
        if (package_root / "__init__.py").exists() and (package_root / "music").exists():
            return package_root
    return candidates[0]


def _stable_index(value: str, size: int) -> int:
    if size <= 0:
        return 0
    return sum(ord(char) for char in str(value)) % size


def save_rollout_debug_artifact(
    *,
    output_dir: Path,
    instance: PrimitiveInstance,
    rollout: PrimitiveOnlineRolloutResult,
) -> Path:
    path = output_dir / f"{_safe_filename(instance.segment_id)}.npz"
    save_npz(
        path,
        predicted_actions=_empty_array_if_none(rollout.predicted_actions),
        observed_piano_states=_empty_array_if_none(rollout.observed_piano_states),
        observed_key_roll=_empty_array_if_none(rollout.observed_key_roll),
        contact_roll=_empty_array_if_none(rollout.contact_roll),
        observed_hand_joints=_empty_array_if_none(rollout.observed_hand_joints),
        observed_hand_fingertips=_empty_array_if_none(rollout.observed_hand_fingertips),
        actions_gt=_empty_array_if_none(instance.actions_gt),
        goals_gt=_empty_array_if_none(instance.goals),
        piano_states_gt=_empty_array_if_none(instance.piano_states_gt),
        hand_joints_gt=_empty_array_if_none(instance.hand_joints_gt),
        hand_fingertips_gt=_empty_array_if_none(instance.hand_fingertips_gt),
        intended_keys=np.asarray(instance.intended_keys, dtype=np.int64),
        realized_keys_gt=np.asarray(instance.realized_keys_gt, dtype=np.int64),
    )
    return path


def save_summary_plots(
    *,
    output_dir: Path,
    result_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    max_plot_primitives: int,
) -> None:
    if result_df.empty:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    usable = result_df.loc[result_df.get("causal_validated", pd.Series(False, index=result_df.index)).astype(bool)].copy()
    if usable.empty:
        return
    _plot_histogram(
        output_path=output_dir / "causal_f1_distribution.png",
        values=pd.to_numeric(usable["causal_f1"], errors="coerce").dropna().to_numpy(dtype=np.float32),
        title="Causal Contact F1 Distribution",
        xlabel="Causal F1",
        plt=plt,
    )
    _plot_histogram(
        output_path=output_dir / "intended_key_count_distribution.png",
        values=pd.to_numeric(result_df["intended_key_count"], errors="coerce").dropna().to_numpy(dtype=np.float32),
        title="Intended Key Count Distribution",
        xlabel="Intended Key Count",
        plt=plt,
        bins=np.arange(0, max(int(result_df["intended_key_count"].max()) + 2, 3)),
    )
    if not summary_df.empty:
        top = summary_df.sort_values(
            ["num_instances", "mean_causal_f1"], ascending=[False, False], kind="stable"
        ).head(max(int(max_plot_primitives), 1))
        fig, ax = plt.subplots(figsize=(max(10, 0.35 * len(top) + 4), 5))
        ax.bar(top["primitive_id"].astype(str), pd.to_numeric(top["mean_causal_f1"], errors="coerce").fillna(0.0))
        ax.set_title("Per-Primitive Mean Causal F1")
        ax.set_ylabel("Mean Causal F1")
        ax.set_xlabel("Primitive")
        ax.tick_params(axis="x", rotation=90)
        fig.tight_layout()
        fig.savefig(output_dir / "primitive_mean_causal_f1.png", dpi=200)
        plt.close(fig)
        _plot_primitive_frequency_accuracy_bar_chart(
            output_path=output_dir / "primitive_frequency_vs_accuracy.png",
            summary_df=summary_df,
            max_plot_primitives=max(int(max_plot_primitives), 1),
            plt=plt,
        )


def save_representative_timing_plots(*, output_dir: Path, result_df: pd.DataFrame, top_k: int) -> None:
    usable = result_df.loc[
        result_df["debug_artifact_path"].notna() & result_df["status"].astype(str).isin({"completed", "terminated_early"})
    ].copy()
    if usable.empty:
        return
    ensure_dir(output_dir)
    candidates = pd.concat(
        [
            usable.sort_values("onset_f1", ascending=False, kind="stable").head(max(int(top_k), 1)),
            usable.sort_values("onset_f1", ascending=True, kind="stable").head(max(int(top_k), 1)),
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["segment_id"])
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for row in candidates.itertuples(index=False):
        artifact_path = Path(str(row.debug_artifact_path))
        if not artifact_path.exists():
            continue
        payload = np.load(artifact_path, allow_pickle=True)
        observed = np.asarray(payload["observed_piano_states"], dtype=np.float32)
        ground_truth = np.asarray(payload["piano_states_gt"], dtype=np.float32)
        if observed.size == 0 or ground_truth.size == 0:
            continue
        keys = _keys_to_plot(observed=observed, ground_truth=ground_truth)
        fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
        axes[0].imshow(ground_truth[:, keys].T, aspect="auto", interpolation="nearest", origin="lower")
        axes[0].set_title(f"Ground Truth Keys: {row.segment_id}")
        axes[0].set_ylabel("Key Index")
        axes[1].imshow(observed[:, keys].T, aspect="auto", interpolation="nearest", origin="lower")
        axes[1].set_title("Rollout Keys")
        axes[1].set_xlabel("Frame")
        axes[1].set_ylabel("Key Index")
        fig.tight_layout()
        fig.savefig(output_dir / f"{_safe_filename(str(row.segment_id))}_timing.png", dpi=200)
        plt.close(fig)


def infer_primitive_behavior_hint(group: pd.DataFrame) -> str:
    mean_keys = float(pd.to_numeric(group.get("intended_key_count", pd.Series(dtype=float)), errors="coerce").mean())
    mean_duration = float(pd.to_numeric(group.get("duration_steps", pd.Series(dtype=float)), errors="coerce").mean())
    mean_events = float(pd.to_numeric(group.get("intended_event_count", pd.Series(dtype=float)), errors="coerce").mean())
    if math.isnan(mean_keys) or mean_keys <= 0.25:
        return "move_or_noop"
    if mean_keys >= 2.5:
        return "chord_press"
    if mean_duration >= 24 and mean_events <= 1.25:
        return "hold_or_release"
    if mean_events >= 1.75:
        return "press_sequence"
    return "single_press"


def align_rollout_sequence(
    *,
    observed: np.ndarray | None,
    ground_truth: np.ndarray | None,
    preferred_mode: str | None,
) -> tuple[np.ndarray | None, str]:
    if observed is None:
        return None, "none"
    array = np.asarray(observed, dtype=np.float32)
    if array.ndim != 2:
        return array.reshape(array.shape[0], -1).astype(np.float32), "reshaped"
    if ground_truth is not None:
        target = np.asarray(ground_truth, dtype=np.float32)
        if array.shape[0] == target.shape[0]:
            return array, "direct"
        if array.shape[0] == target.shape[0] + 1:
            pre = array[:-1]
            post = array[1:]
            if preferred_mode == "pre_action":
                return pre, "pre_action"
            if preferred_mode == "post_action":
                return post, "post_action"
            pre_err = float(np.mean((pre - target) ** 2))
            post_err = float(np.mean((post - target) ** 2))
            return (post, "post_action") if post_err < pre_err else (pre, "pre_action")
        steps = min(array.shape[0], target.shape[0])
        return array[:steps], "trimmed"
    if array.shape[0] > 1:
        if preferred_mode == "post_action":
            return array[1:], "post_action"
        return array[:-1], "pre_action"
    return array, "direct"


def _extract_key_events_from_roll(
    *,
    roll: np.ndarray | None,
    key_threshold: float,
    sustain_threshold: float,
    source: str,
) -> KeyEventBundle:
    if roll is None:
        return empty_key_event_bundle(source=source)
    array = np.asarray(roll, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2:
        raise ValueError(f"Expected a rank-2 piano roll, found shape {array.shape}.")
    keys = np.zeros((array.shape[0], _NUM_PIANO_KEYS), dtype=bool)
    width = min(array.shape[1], _NUM_PIANO_KEYS)
    if width > 0:
        keys[:, :width] = array[:, :width] >= float(key_threshold)
    sustain = np.zeros((array.shape[0], 1), dtype=bool)
    if array.shape[1] > _NUM_PIANO_KEYS:
        sustain[:, 0] = array[:, _NUM_PIANO_KEYS] >= float(sustain_threshold)
    events = _events_from_key_roll(keys)
    return KeyEventBundle(
        source=source,
        key_roll=keys.astype(np.float32),
        sustain_roll=sustain.astype(np.float32),
        events=events,
        unique_keys=tuple(sorted({int(event.key_id) for event in events})),
    )


def empty_key_event_bundle(*, source: str) -> KeyEventBundle:
    return KeyEventBundle(
        source=source,
        key_roll=np.zeros((0, _NUM_PIANO_KEYS), dtype=np.float32),
        sustain_roll=np.zeros((0, 1), dtype=np.float32),
        events=[],
        unique_keys=(),
    )


def _events_from_key_roll(key_roll: np.ndarray) -> list[KeyEvent]:
    if key_roll.size == 0:
        return []
    active_roll = np.asarray(key_roll > 0.5, dtype=bool)
    current_onsets: dict[int, int] = {}
    events: list[KeyEvent] = []
    previous = np.zeros((active_roll.shape[1],), dtype=bool)
    for frame_index, frame in enumerate(active_roll):
        onsets = np.flatnonzero(frame & ~previous)
        releases = np.flatnonzero(previous & ~frame)
        for key_id in onsets.tolist():
            current_onsets[int(key_id)] = int(frame_index)
        for key_id in releases.tolist():
            onset = current_onsets.pop(int(key_id), None)
            if onset is not None:
                events.append(KeyEvent(int(key_id), int(onset), int(frame_index)))
        previous = frame
    for key_id, onset in sorted(current_onsets.items()):
        events.append(KeyEvent(int(key_id), int(onset), int(active_roll.shape[0])))
    return sorted(events, key=lambda item: (item.onset_frame, item.key_id, item.release_frame))


def _load_conditioning_feature_lookup(
    *,
    primitive_root: Path,
    assignments_df: pd.DataFrame,
    stage1_config: dict[str, Any],
) -> dict[str, np.ndarray]:
    if assignments_df.empty or "chunk_path" not in assignments_df.columns or "chunk_index" not in assignments_df.columns:
        return {}
    slim_mask = assignments_df["chunk_path"].astype(str).str.startswith("slim_chunk_")
    if not np.any(slim_mask):
        return {}
    subset = assignments_df.loc[slim_mask, ["segment_id", "chunk_path", "chunk_index"]].copy()
    try:
        matrix, _ = load_feature_matrix_from_store(
            segment_df=subset,
            output_dir=primitive_root,
            config=stage1_config,
        )
    except FileNotFoundError:
        return {}
    return {
        str(segment_id): np.asarray(matrix[index], dtype=np.float32)
        for index, segment_id in enumerate(subset["segment_id"].astype(str).tolist())
    }


def _clear_previous_outputs(output_root: Path) -> None:
    for name in (
        "primitive_instance_metrics.jsonl",
        "primitive_instances_enriched.csv",
        "primitive_instances_enriched.parquet",
        "primitive_instance_metrics.csv",
        "primitive_instance_metrics.parquet",
        "primitive_summary_metrics.csv",
        "primitive_summary_metrics.parquet",
        "aggregate_metrics.json",
        "failure_counts.json",
        "config_snapshot.json",
        "preflight.json",
        "runtime_status.json",
    ):
        path = output_root / name
        if path.exists():
            path.unlink()
    for directory_name in ("plots", "debug", "videos", "frames_tmp"):
        path = output_root / directory_name
        if path.exists():
            shutil.rmtree(path)


def _resolve_eval_config(config: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(config)
    sampling = dict(resolved.get("sampling", {}))
    events = dict(resolved.get("events", {}))
    rollout = dict(resolved.get("rollout", {}))
    causal_eval = default_causal_eval_config(resolved.get("causal_eval") or rollout.get("causal_eval"))
    aggregation = dict(resolved.get("aggregation", {}))
    resolved["sampling"] = sampling
    resolved["events"] = events
    resolved["rollout"] = rollout
    resolved["causal_eval"] = causal_eval
    resolved["aggregation"] = aggregation
    resolved.setdefault("seed", 7)
    resolved.setdefault("resume", True)
    resolved.setdefault("save_plots", True)
    resolved.setdefault("save_debug", False)
    sampling.setdefault("instances_per_primitive", 4)
    sampling.setdefault("primitive_ids", [])
    events.setdefault("use_goals", True)
    events.setdefault("use_piano_states", True)
    events.setdefault("allow_goal_realized_fallback", False)
    events.setdefault("allow_piano_state_intended_fallback", False)
    events.setdefault("goal_key_threshold", 0.5)
    events.setdefault("goal_sustain_threshold", 0.5)
    events.setdefault("piano_state_threshold", 0.5)
    events.setdefault("piano_sustain_threshold", 0.5)
    events.setdefault("onset_tolerance_frames", 1)
    rollout.setdefault("validate_action_dim", True)
    rollout.setdefault("source_mode", "dataset_song")
    rollout.setdefault("restore_mode", causal_eval["restore_mode"])
    rollout.setdefault("video_audio_source", causal_eval["video_audio_source"])
    rollout.setdefault("render_video", False)
    rollout.setdefault("max_render_instances", None)
    rollout.setdefault("video_fps", 20)
    rollout.setdefault("video_height", 480)
    rollout.setdefault("video_width", 640)
    rollout.setdefault("example_midi_paths", [])
    rollout.setdefault("example_environment_names", [])
    rollout.setdefault("example_midi_labels", [])
    rollout.setdefault("piano_state_threshold", events["piano_state_threshold"])
    rollout.setdefault("piano_sustain_threshold", events["piano_sustain_threshold"])
    rollout["causal_eval"] = causal_eval
    aggregation.setdefault("top_k_examples", 3)
    aggregation.setdefault("max_plot_primitives", 24)
    return resolved


def _read_table_base(path_without_suffix: Path) -> pd.DataFrame:
    parquet_path = path_without_suffix.with_suffix(".parquet")
    csv_path = path_without_suffix.with_suffix(".csv")
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    return pd.read_csv(csv_path)


def _slice_episode_field(array: np.ndarray | None, start: int, end: int) -> np.ndarray | None:
    if array is None:
        return None
    return np.asarray(array[start:end], dtype=np.float32)


def _gradient_or_none(array: np.ndarray | None, control_timestep: float) -> np.ndarray | None:
    if array is None or array.shape[0] <= 1:
        return None
    return np.gradient(np.asarray(array, dtype=np.float32), max(float(control_timestep), 1e-6), axis=0).astype(np.float32)


def _first_frame(array: np.ndarray | None) -> np.ndarray | None:
    if array is None or array.size == 0:
        return None
    return np.asarray(array[0], dtype=np.float32).reshape(-1)


def _lookup_prior_path(*, library_df: pd.DataFrame, primitive_id: str) -> str | None:
    if library_df.empty or "primitive_id" not in library_df.columns or "prior_path" not in library_df.columns:
        return None
    rows = library_df.loc[library_df["primitive_id"].astype(str) == str(primitive_id), "prior_path"]
    if rows.empty:
        return None
    text = str(rows.iloc[0]).strip()
    return text or None


def _binary_metric(*, numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0
    return float(numerator) / float(denominator)


def _f1_from_precision_recall(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def _empty_event_metrics() -> dict[str, Any]:
    return {
        "true_positives": 0,
        "false_positive_events": 0,
        "missed_events": 0,
        "precision": float("nan"),
        "recall": float("nan"),
        "f1": float("nan"),
        "timing_errors": [],
        "mean_abs_timing_error_frames": float("nan"),
        "median_abs_timing_error_frames": float("nan"),
        "false_positive_unique_keys": 0,
        "missed_unique_keys": 0,
    }


def _resample_sequence(sequence: np.ndarray, steps: int) -> np.ndarray:
    array = np.asarray(sequence, dtype=np.float32)
    if array.ndim != 2:
        array = array.reshape(array.shape[0], -1)
    if int(steps) <= 0:
        raise ValueError("Cannot resample a sequence to zero steps.")
    if array.shape[0] == int(steps):
        return array.astype(np.float32)
    x_old = np.linspace(0.0, 1.0, array.shape[0], dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, int(steps), dtype=np.float32)
    output = np.zeros((int(steps), array.shape[1]), dtype=np.float32)
    for dim in range(array.shape[1]):
        output[:, dim] = np.interp(x_new, x_old, array[:, dim])
    return output.astype(np.float32)


def _aligned_mse(left: np.ndarray | None, right: np.ndarray | None) -> float:
    if left is None or right is None:
        return float("nan")
    lhs = np.asarray(left, dtype=np.float32)
    rhs = np.asarray(right, dtype=np.float32)
    if lhs.ndim != 2:
        lhs = lhs.reshape(lhs.shape[0], -1)
    if rhs.ndim != 2:
        rhs = rhs.reshape(rhs.shape[0], -1)
    steps = min(lhs.shape[0], rhs.shape[0])
    dims = min(lhs.shape[1], rhs.shape[1])
    if steps == 0 or dims == 0:
        return float("nan")
    return float(np.mean((lhs[:steps, :dims] - rhs[:steps, :dims]) ** 2))


def _mean_key_center(keys: Sequence[int]) -> float:
    if not keys:
        return float("nan")
    return float(np.mean(np.asarray(keys, dtype=np.float32) / max(_NUM_PIANO_KEYS - 1, 1)))


def _invalid_motion_flag(*arrays: np.ndarray | None) -> bool:
    for array in arrays:
        if array is None:
            continue
        if not np.all(np.isfinite(np.asarray(array))):
            return True
    return False


def _activation_toggle_count(key_roll: np.ndarray | None) -> int:
    if key_roll is None or key_roll.shape[0] <= 1:
        return 0
    binary = np.asarray(key_roll > 0.5, dtype=np.int32)
    return int(np.abs(np.diff(binary, axis=0)).sum())


def _build_failure_row_from_build_failure(item: PrimitiveInstanceBuildFailure) -> dict[str, Any]:
    row = {
        "segment_id": item.segment_id,
        "primitive_id": item.primitive_id,
        "song_id": item.song_id,
        "episode_id": item.episode_id,
        "status": "instance_build_failed",
        "success": False,
        "error": item.error,
    }
    row.update(
        {
            "onset_precision": float("nan"),
            "onset_recall": float("nan"),
            "onset_f1": float("nan"),
            "legacy_state_onset_precision": float("nan"),
            "legacy_state_onset_recall": float("nan"),
            "legacy_state_onset_f1": float("nan"),
            "state_onset_f1": float("nan"),
            "false_positive_key_events": float("nan"),
            "missed_key_events": float("nan"),
            "mean_abs_timing_error_frames": float("nan"),
            "causal_validated": False,
            "causal_failure_reason": item.error,
            "restore_mode": "none",
            "zero_action_ablation_passed": False,
            "neutral_start_passed": False,
            "contact_gate_passed": False,
            "target_source": "unavailable",
            "observed_source": "unavailable",
            "causal_true_positive_events": 0,
            "causal_false_positive_events": 0,
            "causal_missed_events": 0,
            "causal_precision": float("nan"),
            "causal_recall": float("nan"),
            "causal_f1": float("nan"),
        }
    )
    return row


def _load_existing_instance_rows(output_root: Path) -> list[dict[str, Any]]:
    path = output_root / "primitive_instance_metrics.jsonl"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            rows.append(json.loads(raw_line))
    return rows


def _append_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_ready(row), sort_keys=True))
        handle.write("\n")


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    return int(value)


def _as_float_array(value: np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value, dtype=np.float32)


def _safe_close(env: Any) -> None:
    if env is None:
        return
    close = getattr(env, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            return


def _restore_hand_state(*, task: Any, physics: Any, instance: PrimitiveInstance) -> bool:
    if instance.start_joint_state is None:
        return False
    right_joints = getattr(getattr(task, "right_hand", None), "joints", None)
    left_joints = getattr(getattr(task, "left_hand", None), "joints", None)
    if right_joints is None or left_joints is None:
        return False
    right_dim = len(right_joints)
    left_dim = len(left_joints)
    qpos = np.asarray(instance.start_joint_state, dtype=np.float32).reshape(-1)
    if qpos.shape[0] < right_dim + left_dim:
        return False
    physics.bind(right_joints).qpos = qpos[:right_dim]
    physics.bind(left_joints).qpos = qpos[right_dim : right_dim + left_dim]
    if instance.start_joint_velocity is not None:
        qvel = np.asarray(instance.start_joint_velocity, dtype=np.float32).reshape(-1)
        if qvel.shape[0] >= right_dim + left_dim:
            physics.bind(right_joints).qvel = qvel[:right_dim]
            physics.bind(left_joints).qvel = qvel[right_dim : right_dim + left_dim]
    return True


def _restore_piano_state_unsafe_legacy(*, task: Any, physics: Any, instance: PrimitiveInstance) -> bool:
    if instance.start_piano_state is None:
        return False
    piano = getattr(task, "piano", None)
    if piano is None or not hasattr(piano, "joints"):
        return False
    state = np.asarray(instance.start_piano_state, dtype=np.float32).reshape(-1)
    key_state = np.zeros((_NUM_PIANO_KEYS,), dtype=np.float32)
    width = min(state.shape[0], _NUM_PIANO_KEYS)
    key_state[:width] = state[:width]
    qpos_range = getattr(piano, "_qpos_range", None)
    if qpos_range is None:
        return False
    physics.bind(piano.joints).qpos = np.asarray(key_state, dtype=np.float32) * np.asarray(qpos_range[:, 1], dtype=np.float32)
    if state.shape[0] > _NUM_PIANO_KEYS and hasattr(piano, "_sustain_state"):
        piano._sustain_state[0] = float(np.clip(state[_NUM_PIANO_KEYS], 0.0, 1.0))
    return True


def _capture_piano_state(env: Any) -> np.ndarray:
    task = env.task
    piano = task.piano
    sustain = np.asarray(piano.sustain_state, dtype=np.float32).reshape(-1)
    return np.concatenate(
        [
            np.asarray(piano.normalized_state, dtype=np.float32).reshape(-1),
            sustain[:1] if sustain.size else np.zeros((1,), dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)


def _capture_hand_joint_state(env: Any) -> np.ndarray:
    task = env.task
    right = np.asarray(env.physics.bind(task.right_hand.joints).qpos, dtype=np.float32).reshape(-1)
    left = np.asarray(env.physics.bind(task.left_hand.joints).qpos, dtype=np.float32).reshape(-1)
    return np.concatenate([right, left], axis=0).astype(np.float32)


def _capture_fingertips(env: Any) -> np.ndarray:
    task = env.task
    right = np.asarray(env.physics.bind(task.right_hand.fingertip_sites).xpos, dtype=np.float32).reshape(-1)
    left = np.asarray(env.physics.bind(task.left_hand.fingertip_sites).xpos, dtype=np.float32).reshape(-1)
    return np.concatenate([right, left], axis=0).astype(np.float32)


def _stack_frames(frames: list[np.ndarray]) -> np.ndarray | None:
    if not frames:
        return None
    return np.stack([np.asarray(frame, dtype=np.float32).reshape(-1) for frame in frames], axis=0).astype(np.float32)


def _value_histogram_json(values: Sequence[Any]) -> str:
    counter = Counter(str(int(value)) for value in values if value is not None and not (isinstance(value, float) and np.isnan(value)))
    return json.dumps(dict(sorted(counter.items())))


def _representative_segments(frame: pd.DataFrame, *, metric: str, ascending: bool, top_k: int) -> list[str]:
    if frame.empty or metric not in frame.columns:
        return []
    ordered = frame.sort_values(metric, ascending=ascending, kind="stable")
    return ordered["segment_id"].astype(str).head(max(int(top_k), 1)).tolist()


def _representative_failures(frame: pd.DataFrame, *, top_k: int) -> list[str]:
    failed = frame.loc[~frame["status"].astype(str).isin({"completed", "terminated_early"})]
    return failed["segment_id"].astype(str).head(max(int(top_k), 1)).tolist()


def _keyboard_region_label(mean_center: float) -> str:
    if math.isnan(mean_center):
        return "unknown"
    if mean_center < 0.33:
        return "low_register"
    if mean_center < 0.66:
        return "mid_register"
    return "high_register"


def _mode_or_unknown(series: pd.Series) -> str:
    if series.empty:
        return "unknown"
    cleaned = [str(item) for item in series.fillna("unknown").astype(str).tolist() if str(item).strip()]
    if not cleaned:
        return "unknown"
    return Counter(cleaned).most_common(1)[0][0]


def _plot_histogram(
    *,
    output_path: Path,
    values: np.ndarray,
    title: str,
    xlabel: str,
    plt: Any,
    bins: np.ndarray | int | None = None,
) -> None:
    if values.size == 0:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values, bins=bins if bins is not None else 20)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_primitive_frequency_accuracy_bar_chart(
    *,
    output_path: Path,
    summary_df: pd.DataFrame,
    max_plot_primitives: int,
    plt: Any,
) -> None:
    """Bar heights = evaluated instance counts; fill color maps mean onset F1 (accuracy)."""
    if summary_df.empty:
        return
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    frame = summary_df.sort_values("num_instances", ascending=False, kind="stable").head(max_plot_primitives)
    labels = frame["primitive_id"].astype(str).tolist()
    counts = pd.to_numeric(frame["num_instances"], errors="coerce").fillna(0).to_numpy(dtype=float)
    accuracy_column = "mean_causal_f1" if "mean_causal_f1" in frame.columns else "mean_legacy_state_onset_f1"
    accuracy = pd.to_numeric(frame[accuracy_column], errors="coerce").to_numpy(dtype=float)

    cmap = plt.cm.viridis
    norm = Normalize(vmin=0.0, vmax=1.0, clip=True)
    colors: list[Any] = []
    no_accuracy_rgba = (0.78, 0.78, 0.78, 1.0)
    for val in accuracy:
        colors.append(no_accuracy_rgba if (val != val or math.isnan(float(val))) else cmap(norm(float(val))))

    fig_w = max(10.0, 0.35 * len(labels) + 5.0)
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    ax.bar(labels, counts, color=colors, edgecolor="black", linewidth=0.3)
    ax.set_title("Primitive frequency in online evaluation (bar color = mean onset F1)")
    ax.set_ylabel("Evaluated instances")
    ax.set_xlabel("Primitive ID")
    ax.tick_params(axis="x", rotation=90)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Mean causal F1")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _keys_to_plot(*, observed: np.ndarray, ground_truth: np.ndarray) -> list[int]:
    observed_binary = np.asarray(observed[:, :_NUM_PIANO_KEYS] > 0.5, dtype=bool)
    gt_binary = np.asarray(ground_truth[:, :_NUM_PIANO_KEYS] > 0.5, dtype=bool)
    active = np.flatnonzero(observed_binary.any(axis=0) | gt_binary.any(axis=0)).tolist()
    return active[:24] if active else list(range(12))


def _empty_array_if_none(array: np.ndarray | None) -> np.ndarray:
    if array is None:
        return np.zeros((0, 0), dtype=np.float32)
    return np.asarray(array, dtype=np.float32)


def _safe_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)
