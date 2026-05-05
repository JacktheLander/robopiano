from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sonata.evaluation.causal_rollout_contract import (  # noqa: E402
    CausalRolloutConfig,
    assert_neutral_piano_start,
    contact_gated_keypress_metrics,
    mark_uncausal_result,
)
from sonata.evaluation.primitive_online_eval import _resolve_eval_config, infer_instance_key_events  # noqa: E402


def test_zero_actions_cannot_pass_causal_eval() -> None:
    result = {"success": True, "causal_validated": True, "status": "completed"}

    contaminated = mark_uncausal_result(result, reason="zero_action_pressed_target", status="contaminated_zero_action")

    assert contaminated["success"] is False
    assert contaminated["causal_validated"] is False
    assert contaminated["status"] == "contaminated_zero_action"


def test_initial_active_keys_fail_neutral_start_validation() -> None:
    piano_state = np.zeros((89,), dtype=np.float32)
    piano_state[12] = 0.25

    check = assert_neutral_piano_start(piano_state, threshold=0.1)

    assert check.passed is False
    assert check.initial_active_key_indices == [12]
    assert check.failure_reason == "initial_keys_active"


def test_target_keys_are_not_observed_without_activation() -> None:
    activation = np.zeros((4, 89), dtype=np.float32)
    contact = np.zeros((4, 88), dtype=np.float32)
    config = CausalRolloutConfig()

    metrics = contact_gated_keypress_metrics(
        target_key_indices=[10],
        activation_roll=activation,
        contact_roll=contact,
        config=config,
        contact_method="distance_proxy",
    )

    assert metrics["causal_true_positive_events"] == 0
    assert metrics["causal_missed_events"] == 1
    assert metrics["contact_gated_success"] is False


def test_goals_cannot_become_realized_events_by_default() -> None:
    goals = np.zeros((3, 89), dtype=np.float32)
    piano_states = np.zeros((3, 89), dtype=np.float32)
    goals[1:, 5] = 1.0

    intended, realized = infer_instance_key_events(
        goals=goals,
        piano_states=piano_states,
        events_config={"use_goals": True, "use_piano_states": True},
    )

    assert intended.unique_keys == (5,)
    assert realized.unique_keys == ()
    assert realized.source == "piano_states"


def test_piano_states_cannot_become_intended_events_by_default() -> None:
    goals = np.zeros((3, 89), dtype=np.float32)
    piano_states = np.zeros((3, 89), dtype=np.float32)
    piano_states[1:, 7] = 1.0

    intended, realized = infer_instance_key_events(
        goals=goals,
        piano_states=piano_states,
        events_config={"use_goals": True, "use_piano_states": True},
    )

    assert intended.unique_keys == ()
    assert intended.source == "goals"
    assert realized.unique_keys == (7,)


def test_contact_gate_rejects_activation_without_contact() -> None:
    activation = np.zeros((4, 89), dtype=np.float32)
    activation[2:, 10] = 1.0
    contact = np.zeros((4, 88), dtype=np.float32)
    config = CausalRolloutConfig(contact_tolerance_frames=2)

    metrics = contact_gated_keypress_metrics(
        target_key_indices=[10],
        activation_roll=activation,
        contact_roll=contact,
        config=config,
        contact_method="distance_proxy",
    )

    assert metrics["causal_true_positive_events"] == 0
    assert metrics["activation_without_contact_key_indices"] == [10]
    assert metrics["contact_gated_success"] is False


def test_contact_gate_accepts_activation_with_preceding_contact() -> None:
    activation = np.zeros((4, 89), dtype=np.float32)
    activation[2:, 10] = 1.0
    contact = np.zeros((4, 88), dtype=np.float32)
    contact[1, 10] = 1.0
    config = CausalRolloutConfig(contact_tolerance_frames=2)

    metrics = contact_gated_keypress_metrics(
        target_key_indices=[10],
        activation_roll=activation,
        contact_roll=contact,
        config=config,
        contact_method="distance_proxy",
    )

    assert metrics["causal_true_positive_events"] == 1
    assert metrics["causal_f1"] == 1.0
    assert metrics["contact_gated_success"] is True


def test_contact_unavailable_fails_when_required() -> None:
    activation = np.zeros((4, 89), dtype=np.float32)
    activation[2:, 10] = 1.0
    config = CausalRolloutConfig(require_contact_for_keypress=True)

    metrics = contact_gated_keypress_metrics(
        target_key_indices=[10],
        activation_roll=activation,
        contact_roll=None,
        config=config,
        contact_method="unavailable",
    )

    assert metrics["status"] == "contact_unavailable"
    assert metrics["causal_validated"] is False
    assert metrics["contact_gated_success"] is False


def test_unsafe_legacy_config_marks_non_causal() -> None:
    config = CausalRolloutConfig.from_mapping({"enabled": False, "restore_mode": "unsafe_legacy"})

    assert config.enabled is False
    assert config.allow_piano_state_restore is True


def test_primitive_online_entrypoint_defaults_to_causal_eval() -> None:
    resolved = _resolve_eval_config({})

    assert resolved["causal_eval"]["enabled"] is True
    assert resolved["causal_eval"]["restore_mode"] == "hands_only"
    assert resolved["causal_eval"]["run_zero_action_ablation"] is True
    assert resolved["causal_eval"]["require_contact_for_keypress"] is True
    assert resolved["rollout"]["video_audio_source"] == "none"
