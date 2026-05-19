from __future__ import annotations

import numpy as np

from etude.features.key_blocks import KeyFeatureSpec, build_key_features, compute_key_error_masks


def test_key_feature_shape_is_deterministic() -> None:
    spec = KeyFeatureSpec(include_transitions=True)
    target_keys = np.zeros((4, 88), dtype=np.float32)
    target_keys[0, [10, 11]] = 1.0
    target_keys[1, [12]] = 1.0
    key_state = np.zeros((4, 88), dtype=np.float32)
    key_state[0, 10] = 1.0

    features = build_key_features(
        t=0,
        target_keys=target_keys,
        key_state=key_state,
        metadata={"time_to_next_active_key": np.array([0.05, 0.02, 0.0, 0.1], dtype=np.float32)},
        spec=spec,
    )

    expected = (1 + len(spec.lookahead_steps) + 1 + 2 + 1) * spec.key_dim + 2
    assert features.dtype == np.float32
    assert features.shape == (expected,)


def test_missing_optional_key_state_is_zero_filled() -> None:
    spec = KeyFeatureSpec(include_transitions=False, zero_fill_missing=True)
    target_keys = np.zeros((2, 88), dtype=np.float32)
    target_keys[0, [5, 8]] = 1.0

    features = build_key_features(
        t=0,
        target_keys=target_keys,
        key_state=None,
        metadata=None,
        spec=spec,
    )

    key_block_start = (1 + len(spec.lookahead_steps)) * spec.key_dim
    current_state = features[key_block_start : key_block_start + spec.key_dim]
    missed = features[key_block_start + spec.key_dim : key_block_start + (2 * spec.key_dim)]
    assert np.all(current_state == 0.0)
    assert np.count_nonzero(missed) == 2


def test_wrong_key_and_missed_key_masks_are_correct() -> None:
    target_now = np.zeros(88, dtype=np.float32)
    target_now[[1, 4]] = 1.0
    key_state_now = np.zeros(88, dtype=np.float32)
    key_state_now[[1, 7]] = 1.0

    missed, wrong = compute_key_error_masks(target_now, key_state_now)

    assert np.array_equal(np.flatnonzero(missed), np.array([4]))
    assert np.array_equal(np.flatnonzero(wrong), np.array([7]))
