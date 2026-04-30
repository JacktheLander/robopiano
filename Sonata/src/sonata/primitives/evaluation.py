from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sonata.utils.io import write_json


@dataclass
class EvaluationDecision:
    phase: str
    metrics: dict[str, Any]
    triggers: list[dict[str, Any]] = field(default_factory=list)
    warn: bool = False
    stop: bool = False


class Stage1EarlyStop(RuntimeError):
    pass


class Stage1OnlineEvaluator:
    def __init__(self, config: dict[str, Any], output_dir: Path, logger: logging.Logger | None = None):
        self.config = config
        self.output_dir = Path(output_dir).resolve()
        self.logger = logger
        self.enabled = bool(config.get("enable_stage1_online_eval", True))
        self.warn_only = bool(config.get("stage1_warn_only", True))
        self.early_stop_enabled = bool(config.get("stage1_early_stop_enabled", False))
        self.interval = max(int(config.get("stage1_eval_interval_segments", 4096)), 1)
        self.subsample_size = max(int(config.get("stage1_eval_subsample_size", 2048)), 32)
        self.min_segments_before_stop_check = max(int(config.get("stage1_min_segments_before_stop_check", 2048)), 1)
        self.patience_windows = max(int(config.get("stage1_patience_windows", 3)), 1)

        self.eval_dir = self.output_dir / "evaluation"
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_jsonl_path = self.eval_dir / "stage1_online_metrics.jsonl"
        self.summary_path = self.eval_dir / "stage1_summary.json"
        self.failure_json_path = self.eval_dir / "stage1_failure_summary.json"
        self.failure_txt_path = self.eval_dir / "stage1_failure_summary.txt"

        self.segment_state: dict[str, Any] = {
            "segments": 0,
            "durations": [],
            "boundary_energy_sum": 0.0,
            "boundary_alignment_sum": 0.0,
            "short_count": 0,
            "long_count": 0,
            "family_counts": Counter(),
            "duplicate_iou_sum": 0.0,
            "duplicate_count": 0,
            "merge_count": 0,
            "split_count": 0,
            "recent_rows": [],
            "next_checkpoint": self.interval,
            "windows_without_improvement": 0,
            "best_score": float("-inf"),
        }
        self.clustering_state: dict[str, Any] = {
            "windows_without_improvement": 0,
            "best_score": float("-inf"),
        }
        self.gmr_state: dict[str, Any] = {
            "rows": [],
            "windows_without_improvement": 0,
            "best_score": float("-inf"),
        }
        self.history: list[dict[str, Any]] = []
        self.latest_metrics: dict[str, dict[str, Any]] = {}
        self.failure_payload: dict[str, Any] | None = None

    def observe_segmentation(self, rows: list[dict[str, Any]], batch_stats: dict[str, Any] | None = None) -> EvaluationDecision | None:
        if not self.enabled or not rows:
            return None
        batch_stats = batch_stats or {}
        state = self.segment_state
        durations = [max(int(row.get("duration_steps", 0)), 1) for row in rows]
        state["segments"] += len(rows)
        state["durations"].extend(durations)
        if len(state["durations"]) > self.subsample_size * 4:
            state["durations"] = state["durations"][-self.subsample_size * 4 :]
        state["boundary_energy_sum"] += sum(float(row.get("boundary_energy", 0.0)) for row in rows)
        state["boundary_alignment_sum"] += sum(float(row.get("boundary_alignment_score", 0.0)) for row in rows)
        state["duplicate_iou_sum"] += sum(float(row.get("duplicate_iou", 0.0)) for row in rows)
        state["duplicate_count"] += sum(float(row.get("duplicate_iou", 0.0)) >= float(self.config.get("segment_duplicate_iou_threshold", 0.85)) for row in rows)
        state["short_count"] += sum(duration <= int(self.config.get("segment_min_len", 4)) for duration in durations)
        state["long_count"] += sum(duration >= int(self.config.get("segment_max_len", 64)) for duration in durations)
        state["merge_count"] += int(batch_stats.get("merged_segments", 0))
        state["split_count"] += int(batch_stats.get("split_segments", 0))
        state["family_counts"].update(str(row.get("coarse_family") or row.get("heuristic_family") or "unknown") for row in rows)
        state["recent_rows"].extend(
            {
                "episode_id": str(row.get("episode_id", "")),
                "onset_step": int(row.get("onset_step", 0)),
                "end_step": int(row.get("end_step", 0)),
                "coarse_family": str(row.get("coarse_family") or row.get("heuristic_family") or "unknown"),
                "duplicate_iou": float(row.get("duplicate_iou", 0.0)),
                "target_onset_step": int(row.get("target_onset_step", row.get("onset_step", 0))),
                "target_key_count": float(row.get("target_key_count", row.get("chord_size", 0.0)) or 0.0),
            }
            for row in rows
        )
        if len(state["recent_rows"]) > self.subsample_size:
            state["recent_rows"] = state["recent_rows"][-self.subsample_size :]
        if state["segments"] < state["next_checkpoint"]:
            return None

        metrics = self._build_segmentation_metrics(batch_stats=batch_stats)
        decision = self._finalize_phase_metrics("segmentation", metrics)
        state["next_checkpoint"] += self.interval
        return decision

    def observe_clustering(self, assignments_df: pd.DataFrame, sweep_df: pd.DataFrame) -> EvaluationDecision | None:
        if not self.enabled or assignments_df.empty:
            return None
        cluster_sizes = assignments_df["primitive_id"].value_counts()
        family_column = "coarse_family" if "coarse_family" in assignments_df.columns else "heuristic_family"
        family_counts = assignments_df[family_column].astype(str).value_counts() if family_column in assignments_df.columns else pd.Series(dtype=int)
        metrics = {
            "segments": int(len(assignments_df)),
            "num_primitives": int(cluster_sizes.shape[0]),
            "cluster_size_histogram": cluster_sizes.to_dict(),
            "tiny_cluster_frac": float((cluster_sizes < int(self.config.get("min_segments_per_primitive", 12))).mean()),
            "mean_assignment_confidence": float(assignments_df["assignment_confidence"].mean()),
            "median_assignment_confidence": float(assignments_df["assignment_confidence"].median()),
            "low_confidence_frac": float((assignments_df["assignment_confidence"] < float(self.config.get("stage1_assignment_confidence_threshold", 0.55))).mean()),
            "mean_assignment_entropy": float(assignments_df.get("assignment_entropy", pd.Series([0.0])).mean()),
            "silhouette": float(sweep_df["silhouette"].dropna().iloc[-1]) if "silhouette" in sweep_df.columns and not sweep_df["silhouette"].dropna().empty else float("nan"),
            "family_balance_entropy": _entropy_from_counts(family_counts.to_dict()),
            "dominant_cluster_frac": float(cluster_sizes.iloc[0] / max(cluster_sizes.sum(), 1)) if not cluster_sizes.empty else 0.0,
        }
        return self._finalize_phase_metrics("clustering", metrics)

    def observe_gmr_primitive(self, library_row: dict[str, Any]) -> EvaluationDecision | None:
        if not self.enabled:
            return None
        self.gmr_state["rows"].append(dict(library_row))
        if len(self.gmr_state["rows"]) % max(int(self.config.get("stage1_eval_interval_primitives", 8)), 1) != 0:
            return None
        metrics = self._build_gmr_metrics(pd.DataFrame(self.gmr_state["rows"]))
        return self._finalize_phase_metrics("gmr", metrics)

    def observe_gmr_library(self, library_df: pd.DataFrame) -> EvaluationDecision | None:
        if not self.enabled or library_df.empty:
            return None
        metrics = self._build_gmr_metrics(library_df)
        return self._finalize_phase_metrics("gmr", metrics)

    def finalize(self, payload: dict[str, Any]) -> Path:
        summary = {
            "status": "early_stopped" if self.failure_payload is not None else "completed",
            "latest_metrics": self.latest_metrics,
            "history_length": int(len(self.history)),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        summary.update(payload)
        write_json(_jsonable(summary), self.summary_path)
        return self.summary_path

    def record_failure(self, *, phase: str, decision: EvaluationDecision) -> None:
        payload = {
            "status": "early_stopped",
            "phase": phase,
            "metrics": _jsonable(decision.metrics),
            "triggers": _jsonable(decision.triggers),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.failure_payload = payload
        write_json(payload, self.failure_json_path)
        lines = [
            "Stage 1 early stop triggered.",
            f"Phase: {phase}",
            "Triggers:",
        ]
        lines.extend(f"- {item['name']}: observed={item['observed']} threshold={item['threshold']}" for item in payload["triggers"])
        self.failure_txt_path.write_text("\n".join(lines), encoding="utf-8")

    def _build_segmentation_metrics(self, batch_stats: dict[str, Any]) -> dict[str, Any]:
        state = self.segment_state
        durations = np.asarray(state["durations"][-self.subsample_size :], dtype=np.float32) if state["durations"] else np.zeros((0,), dtype=np.float32)
        families = state["family_counts"]
        recent_rows = pd.DataFrame(state["recent_rows"])
        overlap_rate = 0.0
        if not recent_rows.empty:
            overlap_rate = float(_approximate_overlap_rate(recent_rows))
        total_segments = max(int(state["segments"]), 1)
        score_onsets = (
            recent_rows[["episode_id", "target_onset_step"]].drop_duplicates().shape[0]
            if not recent_rows.empty and {"episode_id", "target_onset_step"}.issubset(recent_rows.columns)
            else len(recent_rows)
        )
        target_key_counts = [float(row.get("target_key_count", row.get("chord_size", 0.0)) or 0.0) for row in state["recent_rows"]]
        return {
            "segments": int(state["segments"]),
            "mean_segment_length": float(durations.mean()) if durations.size else 0.0,
            "median_segment_length": float(np.median(durations)) if durations.size else 0.0,
            "p95_segment_length": float(np.quantile(durations, 0.95)) if durations.size else 0.0,
            "segment_length_distribution": durations.tolist(),
            "short_segment_frac": float(state["short_count"] / total_segments),
            "long_segment_frac": float(state["long_count"] / total_segments),
            "segments_per_score_onset": float(len(recent_rows) / max(int(score_onsets), 1)) if not recent_rows.empty else 0.0,
            "mean_target_key_count": float(np.mean(target_key_counts)) if target_key_counts else 0.0,
            "boundary_motion_energy": float(state["boundary_energy_sum"] / total_segments),
            "boundary_alignment_score": float(state["boundary_alignment_sum"] / total_segments),
            "overlap_redundancy_rate": overlap_rate,
            "duplicate_segment_rate": float(state["duplicate_count"] / total_segments),
            "duplicate_iou_mean": float(state["duplicate_iou_sum"] / total_segments),
            "family_distribution_entropy": _entropy_from_counts(families),
            "family_distribution": dict(families),
            "merged_segments": int(state["merge_count"]),
            "split_segments": int(state["split_count"]),
            "accepted_segments": int(batch_stats.get("accepted_segments", 0)),
            "proposed_segments": int(batch_stats.get("proposed_segments", 0)),
            "duplicate_segments_dropped": int(batch_stats.get("duplicate_segments_dropped", 0)),
        }

    def _build_gmr_metrics(self, library_df: pd.DataFrame) -> dict[str, Any]:
        if library_df.empty:
            return {
                "num_primitives": 0,
                "mean_reconstruction_mse": 0.0,
                "mean_reconstruction_l1": 0.0,
                "mean_weighted_strike_error": 0.0,
                "low_quality_primitive_frac": 0.0,
                "primitive_failure_rate": 0.0,
                "collapse_indicator": 0.0,
            }
        primitive_count = max(len(library_df), 1)
        low_quality_col = library_df["low_quality_flag"].astype(bool) if "low_quality_flag" in library_df.columns else pd.Series([False] * len(library_df))
        return {
            "num_primitives": int(len(library_df)),
            "mean_reconstruction_mse": float(library_df["reconstruction_mse"].mean()) if "reconstruction_mse" in library_df.columns else 0.0,
            "mean_reconstruction_l1": float(library_df["reconstruction_l1"].mean()) if "reconstruction_l1" in library_df.columns else 0.0,
            "mean_weighted_strike_error": float(library_df["weighted_strike_error"].mean()) if "weighted_strike_error" in library_df.columns else 0.0,
            "reconstruction_dispersion": float(library_df["reconstruction_mse"].std()) if "reconstruction_mse" in library_df.columns and len(library_df) > 1 else 0.0,
            "low_quality_primitive_frac": float(low_quality_col.mean()),
            "primitive_failure_rate": float((library_df.get("fit_failed", pd.Series([0] * len(library_df))).astype(bool).mean())),
            "collapse_indicator": float((library_df.get("duplicate_neighbor_distance", pd.Series([1.0] * len(library_df))).astype(float) < float(self.config.get("stage1_duplicate_primitive_distance", 0.02))).mean()),
            "component_count_histogram": library_df.get("component_count", pd.Series(dtype=int)).value_counts().to_dict(),
            "compression_ratio": float(library_df.get("num_segments", pd.Series(dtype=float)).sum() / primitive_count) if "num_segments" in library_df.columns else 0.0,
        }

    def _finalize_phase_metrics(self, phase: str, metrics: dict[str, Any]) -> EvaluationDecision:
        decision = self._evaluate_policy(phase=phase, metrics=metrics)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "phase": phase,
            "metrics": _jsonable(metrics),
            "warn": bool(decision.warn),
            "stop": bool(decision.stop),
            "triggers": _jsonable(decision.triggers),
        }
        self.history.append(record)
        self.latest_metrics[phase] = record
        with self.metrics_jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")
        if decision.triggers and self.logger is not None:
            level = logging.WARNING if decision.warn or decision.stop else logging.INFO
            self.logger.log(level, "%s Stage 1 evaluation: %s", phase, _summarize_triggers(decision.triggers))
        if decision.stop:
            self.record_failure(phase=phase, decision=decision)
        return decision

    def _evaluate_policy(self, *, phase: str, metrics: dict[str, Any]) -> EvaluationDecision:
        triggers: list[dict[str, Any]] = []
        if phase == "segmentation" and int(metrics.get("segments", 0)) >= self.min_segments_before_stop_check:
            self._check_metric(triggers, "short_segment_frac", metrics.get("short_segment_frac"), self.config.get("stage1_max_short_segment_frac"), greater=True)
            self._check_metric(triggers, "family_distribution_entropy", metrics.get("family_distribution_entropy"), self.config.get("stage1_min_family_entropy"), greater=False)
            self._check_metric(triggers, "segments_per_score_onset", metrics.get("segments_per_score_onset"), self.config.get("stage1_max_segments_per_score_onset"), greater=True)
            self._check_metric(triggers, "p95_segment_length", metrics.get("p95_segment_length"), self.config.get("stage1_max_p95_segment_duration_steps"), greater=True)
        if phase == "clustering" and int(metrics.get("segments", 0)) >= self.min_segments_before_stop_check:
            self._check_metric(triggers, "low_confidence_frac", metrics.get("low_confidence_frac"), self.config.get("stage1_max_low_confidence_frac"), greater=True)
        if phase == "gmr":
            self._check_metric(triggers, "low_quality_primitive_frac", metrics.get("low_quality_primitive_frac"), self.config.get("stage1_max_low_quality_primitive_frac"), greater=True)
            self._check_metric(triggers, "mean_weighted_strike_error", metrics.get("mean_weighted_strike_error"), self.config.get("stage1_max_weighted_recon_error"), greater=True)

        score = self._phase_score(phase=phase, metrics=metrics)
        phase_state_name = {
            "segmentation": "segment_state",
            "clustering": "clustering_state",
            "gmr": "gmr_state",
        }[phase]
        state = getattr(self, phase_state_name)
        if score > float(state["best_score"]) + 1e-3:
            state["best_score"] = float(score)
            state["windows_without_improvement"] = 0
        else:
            state["windows_without_improvement"] += 1
        if state["windows_without_improvement"] >= self.patience_windows and int(metrics.get("segments", self.min_segments_before_stop_check)) >= self.min_segments_before_stop_check:
            triggers.append(
                {
                    "name": "no_improvement",
                    "observed": int(state["windows_without_improvement"]),
                    "threshold": int(self.patience_windows),
                    "direction": ">=",
                }
            )

        should_warn = bool(triggers)
        should_stop = bool(triggers) and self.early_stop_enabled and not self.warn_only
        return EvaluationDecision(phase=phase, metrics=metrics, triggers=triggers, warn=should_warn, stop=should_stop)

    def _phase_score(self, *, phase: str, metrics: dict[str, Any]) -> float:
        if phase == "segmentation":
            return (
                float(metrics.get("boundary_alignment_score", 0.0))
                + float(metrics.get("family_distribution_entropy", 0.0))
                - float(metrics.get("short_segment_frac", 0.0))
                - float(metrics.get("duplicate_segment_rate", 0.0))
            )
        if phase == "clustering":
            return (
                float(metrics.get("mean_assignment_confidence", 0.0))
                + float(metrics.get("silhouette", 0.0) if np.isfinite(metrics.get("silhouette", np.nan)) else 0.0)
                - float(metrics.get("low_confidence_frac", 0.0))
                - float(metrics.get("tiny_cluster_frac", 0.0))
            )
        return (
            -float(metrics.get("mean_weighted_strike_error", 0.0))
            -float(metrics.get("low_quality_primitive_frac", 0.0))
            -float(metrics.get("primitive_failure_rate", 0.0))
        )

    def _check_metric(
        self,
        triggers: list[dict[str, Any]],
        name: str,
        observed: Any,
        threshold: Any,
        *,
        greater: bool,
    ) -> None:
        if observed is None or threshold is None:
            return
        try:
            observed_value = float(observed)
            threshold_value = float(threshold)
        except (TypeError, ValueError):
            return
        violates = observed_value > threshold_value if greater else observed_value < threshold_value
        if violates:
            triggers.append(
                {
                    "name": name,
                    "observed": observed_value,
                    "threshold": threshold_value,
                    "direction": ">" if greater else "<",
                }
            )


def _approximate_overlap_rate(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    overlaps = 0
    comparisons = 0
    for _, group in frame.groupby("episode_id", sort=False):
        ordered = group.sort_values(["onset_step", "end_step"], kind="stable")
        last_end = None
        for row in ordered.itertuples(index=False):
            if last_end is not None:
                overlaps += int(int(row.onset_step) < int(last_end))
                comparisons += 1
            last_end = max(int(last_end or row.end_step), int(row.end_step))
    return float(overlaps / max(comparisons, 1))


def _entropy_from_counts(counts: dict[str, Any]) -> float:
    if not counts:
        return 0.0
    values = np.asarray([float(value) for value in counts.values()], dtype=np.float32)
    values = values / max(float(values.sum()), 1.0)
    return float(-(values * np.log(values + 1e-8)).sum())


def _summarize_triggers(triggers: list[dict[str, Any]]) -> str:
    return ", ".join(
        f"{item['name']} {item['direction']} {item['threshold']:.4f} (observed {item['observed']:.4f})"
        if isinstance(item.get("threshold"), float)
        else f"{item['name']} {item['direction']} {item['threshold']} (observed {item['observed']})"
        for item in triggers
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Series,)):
        return _jsonable(value.to_dict())
    if isinstance(value, (pd.DataFrame,)):
        return _jsonable(value.to_dict(orient="records"))
    return value
