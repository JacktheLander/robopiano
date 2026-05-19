from __future__ import annotations

import numpy as np

from etude.features.fingertip_phase_blocks import (
    FingertipFeatureSpec,
    PhaseFeatureSpec,
    build_fingertip_features,
    build_phase_features,
    infer_phase_from_target_keys,
)


def test_build_fingertip_features_includes_error_weights_and_masks() -> None:
    current = np.array([[0.0, 0.1, 0.2], [1.0, 1.1, 1.2]], dtype=np.float32)
    desired = np.array([[0.5, 0.6, 0.7], [1.5, 1.6, 1.7]], dtype=np.float32)
    weights = np.array([0.2, 0.8], dtype=np.float32)
    active = np.array([1.0, 0.0], dtype=np.float32)
    inactive = 1.0 - active

    features = build_fingertip_features(
        current_fingertips=current,
        desired_fingertips=desired,
        fingertip_weights=weights,
        active_finger_mask=active,
        inactive_finger_mask=inactive,
        spec=FingertipFeatureSpec(flatten=True),
    )

    expected = np.concatenate(
        [
            current.reshape(-1),
            desired.reshape(-1),
            (desired - current).reshape(-1),
            weights,
            active,
            inactive,
        ]
    )
    np.testing.assert_allclose(features, expected.astype(np.float32))


def test_build_fingertip_features_tolerates_missing_inputs_when_enabled() -> None:
    features = build_fingertip_features(
        current_fingertips=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        spec=FingertipFeatureSpec(
            include_desired=False,
            include_error=False,
            include_weights=False,
            include_active_mask=False,
            include_inactive_mask=False,
            flatten=False,
            allow_missing=True,
        ),
    )
    np.testing.assert_allclose(features, np.array([[1.0, 2.0, 3.0]], dtype=np.float32))


def test_build_phase_features_uses_metadata_sequences() -> None:
    metadata = {
        "phase_ids": np.array([0, 1, 2, 4], dtype=np.int64),
        "phase_scalar": np.array([0.0, 0.2, 0.4, 0.8], dtype=np.float32),
        "phase_mask": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
    }
    features = build_phase_features(t=2, metadata=metadata, spec=PhaseFeatureSpec(encode_as="both"))
    expected = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.4, 1.0], dtype=np.float32)
    np.testing.assert_allclose(features, expected)


def test_build_phase_features_can_infer_from_target_keys() -> None:
    target_keys = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )
    press_id = infer_phase_from_target_keys(target_keys, t=1)
    release_id = infer_phase_from_target_keys(target_keys, t=2)
    assert press_id == 2
    assert release_id == 4

    features = build_phase_features(
        t=0,
        target_keys=target_keys,
        spec=PhaseFeatureSpec(encode_as="one_hot", include_mask=True),
    )
    expected = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    np.testing.assert_allclose(features, expected)


def test_build_phase_features_accepts_legacy_phase_aliases() -> None:
    metadata = {"phases": ["attack", "press", "sustain", "release"]}
    features = build_phase_features(t=2, metadata=metadata, spec=PhaseFeatureSpec(encode_as="one_hot"))
    expected = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    np.testing.assert_allclose(features, expected)
