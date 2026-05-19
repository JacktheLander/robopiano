from __future__ import annotations

from etude.evaluation.failure_attribution import attribute_failures
from etude.evaluation.failure_report import build_failure_report, format_failure_report


def test_kinematic_failure_maps_to_planner_failure() -> None:
    attribution = attribute_failures(
        kinematic_upper_bound_metrics={
            "fingertip_contact_error_cm": 2.2,
            "timing_error_ms": 10.0,
        },
        thresholds={"fingertip_contact_error_cm": 1.0, "timing_error_ms": 50.0},
    )
    assert attribution["primary_failure_mode"] == "planner_geometry_failure"
    assert attribution["recommended_next_component_to_fix"] == "planner_geometry"


def test_kinematic_good_but_pd_bad_maps_to_pd_tracking_failure() -> None:
    attribution = attribute_failures(
        kinematic_upper_bound_metrics={
            "fingertip_contact_error_cm": 0.2,
            "timing_error_ms": 5.0,
        },
        pd_baseline_metrics={
            "tracking/joint_mae": 0.4,
            "tracking/fingertip_mean_cm": 2.0,
        },
        thresholds={"joint_mae": 0.15, "fingertip_mean_cm": 1.0},
    )
    assert attribution["primary_failure_mode"] == "pd_tracking_failure"
    assert "planner_geometry_failure" not in attribution["secondary_failure_modes"]


def test_pd_good_but_residual_bad_maps_to_controller_instability() -> None:
    attribution = attribute_failures(
        kinematic_upper_bound_metrics={"fingertip_contact_error_cm": 0.2},
        pd_baseline_metrics={"tracking/joint_mae": 0.04},
        residual_controller_metrics={
            "control/instability_rate": 0.4,
            "control/residual_action_l2": 0.9,
        },
        event_metrics={"event_f1_drop": 0.25},
        thresholds={"instability_rate": 0.05, "residual_action_l2": 0.5, "event_f1_drop": 0.1},
    )
    assert attribution["primary_failure_mode"] == "controller_instability"
    report = build_failure_report(
        kinematic_upper_bound_metrics={"fingertip_contact_error_cm": 0.2},
        pd_baseline_metrics={"tracking/joint_mae": 0.04},
        residual_controller_metrics={
            "control/instability_rate": 0.4,
            "control/residual_action_l2": 0.9,
        },
        event_metrics={"event_f1_drop": 0.25},
        thresholds={"instability_rate": 0.05, "residual_action_l2": 0.5, "event_f1_drop": 0.1},
    )
    assert report["recommended_next_component_to_fix"] == "residual_controller"
    assert "controller_instability" in format_failure_report(
        kinematic_upper_bound_metrics={"fingertip_contact_error_cm": 0.2},
        pd_baseline_metrics={"tracking/joint_mae": 0.04},
        residual_controller_metrics={
            "control/instability_rate": 0.4,
            "control/residual_action_l2": 0.9,
        },
        event_metrics={"event_f1_drop": 0.25},
        thresholds={"instability_rate": 0.05, "residual_action_l2": 0.5, "event_f1_drop": 0.1},
    )


def test_high_clip_rate_maps_to_action_saturation_failure() -> None:
    attribution = attribute_failures(
        kinematic_upper_bound_metrics={"fingertip_contact_error_cm": 0.1},
        pd_baseline_metrics={"tracking/joint_mae": 0.05},
        residual_controller_metrics={"control/instability_rate": 0.01},
        action_metrics={"control/action_clip_rate": 0.45},
        thresholds={"action_clip_rate": 0.2},
    )
    assert attribution["primary_failure_mode"] == "action_saturation_failure"
    assert attribution["recommended_next_component_to_fix"] == "action_scaling"
