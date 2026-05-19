from __future__ import annotations

import numpy as np

from etude.core.plan_bundle import PlanBundle
from etude.data.corruption import (
    apply_plan_corruption,
    corrupt_q_ref,
    corrupt_timing,
    drop_lookahead,
    drop_waypoints,
)
from etude.data.corruption_config import PlanCorruptionConfig


def test_corrupt_q_ref_is_deterministic() -> None:
    q_ref = np.linspace(0.0, 1.0, 20, dtype=np.float32).reshape(5, 4)
    config = PlanCorruptionConfig(
        enabled=True,
        q_reference_noise_std=0.1,
        q_smooth_drift_std=0.15,
    )
    first = corrupt_q_ref(q_ref, config, np.random.default_rng(7))
    second = corrupt_q_ref(q_ref, config, np.random.default_rng(7))
    assert first.dtype == np.float32
    assert first.shape == q_ref.shape
    np.testing.assert_allclose(first, second)
    assert not np.allclose(first, q_ref)


def test_corrupt_timing_and_waypoint_dropout_preserve_shape() -> None:
    values = np.arange(12, dtype=np.float32).reshape(6, 2)
    shifted = corrupt_timing(values, 2, np.random.default_rng(3))
    dropped = drop_waypoints(values, 0.6, np.random.default_rng(4))
    assert shifted.shape == values.shape
    assert dropped.shape == values.shape
    np.testing.assert_array_equal(dropped[0], values[0])
    assert any(np.array_equal(dropped[i], dropped[i - 1]) for i in range(1, dropped.shape[0]))


def test_drop_lookahead_zeros_future_slots_only() -> None:
    lookahead = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    dropped = drop_lookahead(lookahead, 1.0, np.random.default_rng(0))
    np.testing.assert_array_equal(dropped[:, 0], lookahead[:, 0])
    np.testing.assert_array_equal(dropped[:, 1:], np.zeros_like(lookahead[:, 1:]))


def test_apply_plan_corruption_keeps_targets_and_is_deterministic() -> None:
    q_ref = np.linspace(0.0, 1.0, 24, dtype=np.float32).reshape(6, 4)
    fingertip_ref = np.linspace(-0.2, 0.3, 36, dtype=np.float32).reshape(6, 6)
    target_keys = np.eye(6, dtype=np.float32)
    phase = np.linspace(0.0, 1.0, 6, dtype=np.float32)
    lookahead = np.ones((6, 3, 2), dtype=np.float32)
    bundle = PlanBundle(
        q_ref=q_ref,
        qdot_ref=np.zeros_like(q_ref),
        dt=0.02,
        target_keys=target_keys,
        fingertip_ref=fingertip_ref,
        phase=phase,
        metadata={"controller_lookahead": lookahead},
    )
    config = PlanCorruptionConfig(
        enabled=True,
        q_reference_noise_std=0.02,
        q_smooth_drift_std=0.03,
        fingertip_xy_noise_std=0.01,
        fingertip_z_noise_std=0.02,
        hover_height_bias=0.01,
        press_depth_bias=0.02,
        timing_jitter_frames=1,
        missing_waypoint_probability=0.25,
        lookahead_dropout_probability=1.0,
    )

    first = apply_plan_corruption(bundle, config, np.random.default_rng(11))
    second = apply_plan_corruption(bundle, config, np.random.default_rng(11))

    np.testing.assert_allclose(first.q_ref, second.q_ref)
    np.testing.assert_allclose(first.qdot_ref, second.qdot_ref)
    np.testing.assert_allclose(first.fingertip_ref, second.fingertip_ref)
    np.testing.assert_allclose(first.phase, second.phase)
    np.testing.assert_array_equal(first.target_keys, target_keys)
    np.testing.assert_array_equal(second.target_keys, target_keys)
    assert not np.allclose(first.q_ref, q_ref)
    assert not np.allclose(first.fingertip_ref, fingertip_ref)
    np.testing.assert_array_equal(first.metadata["controller_lookahead"][:, 0], lookahead[:, 0])
    np.testing.assert_array_equal(
        first.metadata["controller_lookahead"][:, 1:],
        np.zeros_like(lookahead[:, 1:]),
    )


def test_curriculum_scale_can_disable_effects_without_turning_off_config() -> None:
    q_ref = np.ones((4, 3), dtype=np.float32)
    config = PlanCorruptionConfig(
        enabled=True,
        q_reference_noise_std=0.2,
        q_smooth_drift_std=0.2,
        curriculum_scale=0.0,
    )
    corrupted = corrupt_q_ref(q_ref, config, np.random.default_rng(19))
    np.testing.assert_array_equal(corrupted, q_ref)
