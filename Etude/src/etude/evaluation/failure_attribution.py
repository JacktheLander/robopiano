from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

FAILURE_CATEGORIES = (
    "planner_geometry_failure",
    "planner_timing_failure",
    "pd_tracking_failure",
    "contact_depth_failure",
    "wrong_key_contact_failure",
    "release_failure",
    "controller_instability",
    "action_saturation_failure",
    "recovery_failure",
    "unknown_failure",
)

DEFAULT_THRESHOLDS: dict[str, float] = {
    "fingertip_contact_error_cm": 1.0,
    "timing_error_ms": 50.0,
    "wrong_key_rate": 0.05,
    "event_f1_drop": 0.1,
    "action_clip_rate": 0.2,
    "joint_mae": 0.15,
    "joint_mse": 0.05,
    "fingertip_mean_cm": 1.0,
    "contact_depth_cm": 0.5,
    "release_error_rate": 0.05,
    "instability_rate": 0.05,
    "recovery_error_rate": 0.1,
    "residual_action_l2": 0.5,
}

_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "timing_error_ms": ("timing_error_ms", "timing/error_ms", "event/timing_error_ms"),
    "joint_mae": ("joint_mae", "tracking/joint_mae"),
    "joint_mse": ("joint_mse", "tracking/joint_mse"),
    "fingertip_mean_cm": ("fingertip_mean_cm", "tracking/fingertip_mean_cm"),
    "fingertip_contact_error_cm": (
        "fingertip_contact_error_cm",
        "contact/fingertip_error_cm",
        "tracking/fingertip_contact_error_cm",
    ),
    "contact_depth_cm": ("contact_depth_cm", "contact/depth_cm", "contact/depth_error_cm"),
    "wrong_key_rate": ("wrong_key_rate", "piano/wrong_key_rate", "event/wrong_key_rate"),
    "event_f1_drop": ("event_f1_drop", "event/f1_drop"),
    "release_error_rate": ("release_error_rate", "piano/release_error_rate", "event/release_error_rate"),
    "instability_rate": (
        "instability_rate",
        "control/instability_rate",
        "control/unstable_fraction",
    ),
    "action_clip_rate": ("action_clip_rate", "control/action_clip_rate"),
    "recovery_error_rate": ("recovery_error_rate", "control/recovery_error_rate"),
    "residual_action_l2": ("residual_action_l2", "control/residual_action_l2", "control/action_l2"),
}


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        return float(np.asarray(value, dtype=np.float32).mean())
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return float(np.asarray(value, dtype=np.float32).mean())
    if isinstance(value, (int, float, np.floating)):
        return float(value)
    return None


def _metric_value(metrics: Mapping[str, Any] | None, key: str) -> float | None:
    if not metrics:
        return None
    aliases = _METRIC_ALIASES.get(key, (key,))
    for alias in aliases:
        if alias in metrics:
            value = _coerce_float(metrics[alias])
            if value is not None:
                return value
    return None


def _threshold_value(thresholds: Mapping[str, Any], key: str) -> float:
    value = thresholds.get(key, DEFAULT_THRESHOLDS[key])
    coerced = _coerce_float(value)
    if coerced is None:
        return DEFAULT_THRESHOLDS[key]
    return coerced


def _build_evidence(
    source_name: str,
    metrics: Mapping[str, Any] | None,
    key: str,
    threshold_key: str,
    thresholds: Mapping[str, Any],
) -> dict[str, Any] | None:
    value = _metric_value(metrics, key)
    if value is None:
        return None
    return {
        "source": source_name,
        "metric": key,
        "value": value,
        "threshold": _threshold_value(thresholds, threshold_key),
    }


def _score_failure_modes(
    *,
    kinematic_upper_bound_metrics: Mapping[str, Any] | None = None,
    pd_baseline_metrics: Mapping[str, Any] | None = None,
    residual_controller_metrics: Mapping[str, Any] | None = None,
    joint_tracking_metrics: Mapping[str, Any] | None = None,
    fingertip_tracking_metrics: Mapping[str, Any] | None = None,
    event_metrics: Mapping[str, Any] | None = None,
    action_metrics: Mapping[str, Any] | None = None,
    thresholds: Mapping[str, Any] | None = None,
) -> tuple[dict[str, float], dict[str, list[dict[str, Any]]]]:
    thresholds = {**DEFAULT_THRESHOLDS, **(dict(thresholds or {}))}
    scores = {category: 0.0 for category in FAILURE_CATEGORIES}
    evidence = {category: [] for category in FAILURE_CATEGORIES}

    def add_failure(category: str, severity: float, item: dict[str, Any] | None) -> None:
        if severity <= 0.0:
            return
        scores[category] += severity
        if item is not None:
            evidence[category].append(item)

    kin_fingertip = _metric_value(kinematic_upper_bound_metrics, "fingertip_contact_error_cm")
    kin_joint = _metric_value(kinematic_upper_bound_metrics, "joint_mae")
    kin_timing = _metric_value(kinematic_upper_bound_metrics, "timing_error_ms")

    if kin_fingertip is not None:
        threshold = _threshold_value(thresholds, "fingertip_contact_error_cm")
        add_failure(
            "planner_geometry_failure",
            max(0.0, kin_fingertip - threshold),
            _build_evidence(
                "kinematic_upper_bound_metrics",
                kinematic_upper_bound_metrics,
                "fingertip_contact_error_cm",
                "fingertip_contact_error_cm",
                thresholds,
            ),
        )
    if kin_joint is not None:
        threshold = _threshold_value(thresholds, "joint_mae")
        add_failure(
            "planner_geometry_failure",
            max(0.0, kin_joint - threshold),
            _build_evidence("kinematic_upper_bound_metrics", kinematic_upper_bound_metrics, "joint_mae", "joint_mae", thresholds),
        )
    if kin_timing is not None:
        threshold = _threshold_value(thresholds, "timing_error_ms")
        add_failure(
            "planner_timing_failure",
            max(0.0, kin_timing - threshold),
            _build_evidence(
                "kinematic_upper_bound_metrics",
                kinematic_upper_bound_metrics,
                "timing_error_ms",
                "timing_error_ms",
                thresholds,
            ),
        )

    pd_joint = _metric_value(pd_baseline_metrics, "joint_mae")
    pd_fingertip = _metric_value(pd_baseline_metrics, "fingertip_mean_cm")
    if pd_joint is not None:
        threshold = _threshold_value(thresholds, "joint_mae")
        add_failure(
            "pd_tracking_failure",
            max(0.0, pd_joint - threshold),
            _build_evidence("pd_baseline_metrics", pd_baseline_metrics, "joint_mae", "joint_mae", thresholds),
        )
    if pd_fingertip is not None:
        threshold = _threshold_value(thresholds, "fingertip_mean_cm")
        add_failure(
            "pd_tracking_failure",
            max(0.0, pd_fingertip - threshold),
            _build_evidence(
                "pd_baseline_metrics",
                pd_baseline_metrics,
                "fingertip_mean_cm",
                "fingertip_mean_cm",
                thresholds,
            ),
        )

    joint_tracking_joint = _metric_value(joint_tracking_metrics, "joint_mae")
    if joint_tracking_joint is not None:
        threshold = _threshold_value(thresholds, "joint_mae")
        add_failure(
            "pd_tracking_failure",
            max(0.0, joint_tracking_joint - threshold),
            _build_evidence("joint_tracking_metrics", joint_tracking_metrics, "joint_mae", "joint_mae", thresholds),
        )

    fingertip_tracking_mean = _metric_value(fingertip_tracking_metrics, "fingertip_mean_cm")
    if fingertip_tracking_mean is not None:
        threshold = _threshold_value(thresholds, "fingertip_mean_cm")
        add_failure(
            "pd_tracking_failure",
            max(0.0, fingertip_tracking_mean - threshold),
            _build_evidence(
                "fingertip_tracking_metrics",
                fingertip_tracking_metrics,
                "fingertip_mean_cm",
                "fingertip_mean_cm",
                thresholds,
            ),
        )

    contact_depth = _metric_value(fingertip_tracking_metrics, "contact_depth_cm")
    if contact_depth is not None:
        threshold = _threshold_value(thresholds, "contact_depth_cm")
        add_failure(
            "contact_depth_failure",
            max(0.0, contact_depth - threshold),
            _build_evidence(
                "fingertip_tracking_metrics",
                fingertip_tracking_metrics,
                "contact_depth_cm",
                "contact_depth_cm",
                thresholds,
            ),
        )

    wrong_key = _metric_value(event_metrics, "wrong_key_rate")
    if wrong_key is not None:
        threshold = _threshold_value(thresholds, "wrong_key_rate")
        add_failure(
            "wrong_key_contact_failure",
            max(0.0, wrong_key - threshold),
            _build_evidence("event_metrics", event_metrics, "wrong_key_rate", "wrong_key_rate", thresholds),
        )

    release_error = _metric_value(event_metrics, "release_error_rate")
    if release_error is not None:
        threshold = _threshold_value(thresholds, "release_error_rate")
        add_failure(
            "release_failure",
            max(0.0, release_error - threshold),
            _build_evidence("event_metrics", event_metrics, "release_error_rate", "release_error_rate", thresholds),
        )

    instability = _metric_value(residual_controller_metrics, "instability_rate")
    if instability is not None:
        threshold = _threshold_value(thresholds, "instability_rate")
        add_failure(
            "controller_instability",
            max(0.0, instability - threshold),
            _build_evidence(
                "residual_controller_metrics",
                residual_controller_metrics,
                "instability_rate",
                "instability_rate",
                thresholds,
            ),
        )

    clip_rate = _metric_value(action_metrics, "action_clip_rate")
    if clip_rate is not None:
        threshold = _threshold_value(thresholds, "action_clip_rate")
        add_failure(
            "action_saturation_failure",
            max(0.0, clip_rate - threshold),
            _build_evidence("action_metrics", action_metrics, "action_clip_rate", "action_clip_rate", thresholds),
        )

    recovery_error = _metric_value(residual_controller_metrics, "recovery_error_rate")
    if recovery_error is not None:
        threshold = _threshold_value(thresholds, "recovery_error_rate")
        add_failure(
            "recovery_failure",
            max(0.0, recovery_error - threshold),
            _build_evidence(
                "residual_controller_metrics",
                residual_controller_metrics,
                "recovery_error_rate",
                "recovery_error_rate",
                thresholds,
            ),
        )

    event_f1_drop = _metric_value(event_metrics, "event_f1_drop")
    residual_action_l2 = _metric_value(residual_controller_metrics, "residual_action_l2")
    if event_f1_drop is not None and residual_action_l2 is not None:
        f1_threshold = _threshold_value(thresholds, "event_f1_drop")
        residual_threshold = _threshold_value(thresholds, "residual_action_l2")
        severity = max(0.0, event_f1_drop - f1_threshold) + max(0.0, residual_action_l2 - residual_threshold)
        add_failure(
            "controller_instability",
            severity,
            {
                "source": "residual_controller_metrics",
                "metric": "residual_action_l2",
                "value": residual_action_l2,
                "threshold": residual_threshold,
            } if severity > 0.0 else None,
        )

    if not any(score > 0.0 for category, score in scores.items() if category != "unknown_failure"):
        scores["unknown_failure"] = 1.0
        evidence["unknown_failure"].append(
            {
                "source": "attribution",
                "metric": "matched_failure_modes",
                "value": 0.0,
                "threshold": 1.0,
            }
        )

    return scores, evidence


def attribute_failures(
    *,
    kinematic_upper_bound_metrics: Mapping[str, Any] | None = None,
    pd_baseline_metrics: Mapping[str, Any] | None = None,
    residual_controller_metrics: Mapping[str, Any] | None = None,
    joint_tracking_metrics: Mapping[str, Any] | None = None,
    fingertip_tracking_metrics: Mapping[str, Any] | None = None,
    event_metrics: Mapping[str, Any] | None = None,
    action_metrics: Mapping[str, Any] | None = None,
    thresholds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    scores, evidence = _score_failure_modes(
        kinematic_upper_bound_metrics=kinematic_upper_bound_metrics,
        pd_baseline_metrics=pd_baseline_metrics,
        residual_controller_metrics=residual_controller_metrics,
        joint_tracking_metrics=joint_tracking_metrics,
        fingertip_tracking_metrics=fingertip_tracking_metrics,
        event_metrics=event_metrics,
        action_metrics=action_metrics,
        thresholds=thresholds,
    )
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    primary_failure_mode = ranked[0][0]
    secondary_failure_modes = [name for name, score in ranked[1:] if score > 0.0]
    recommended_next_component_to_fix = {
        "planner_geometry_failure": "planner_geometry",
        "planner_timing_failure": "planner_timing",
        "pd_tracking_failure": "pd_controller",
        "contact_depth_failure": "contact_model",
        "wrong_key_contact_failure": "key_contact_assignment",
        "release_failure": "release_logic",
        "controller_instability": "residual_controller",
        "action_saturation_failure": "action_scaling",
        "recovery_failure": "recovery_controller",
        "unknown_failure": "instrumentation",
    }[primary_failure_mode]
    return {
        "primary_failure_mode": primary_failure_mode,
        "secondary_failure_modes": secondary_failure_modes,
        "scores": scores,
        "evidence": evidence,
        "recommended_next_component_to_fix": recommended_next_component_to_fix,
        "inputs_used": {
            "kinematic_upper_bound_metrics": bool(kinematic_upper_bound_metrics),
            "pd_baseline_metrics": bool(pd_baseline_metrics),
            "residual_controller_metrics": bool(residual_controller_metrics),
            "joint_tracking_metrics": bool(joint_tracking_metrics),
            "fingertip_tracking_metrics": bool(fingertip_tracking_metrics),
            "event_metrics": bool(event_metrics),
            "action_metrics": bool(action_metrics),
        },
    }
