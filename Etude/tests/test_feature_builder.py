from __future__ import annotations

import numpy as np

from etude.data.feature_builder import FeatureSpec, build_tracking_features


def test_feature_builder_includes_lookahead_and_optional_features() -> None:
    q = np.zeros(46, dtype=np.float32)
    qdot = np.zeros(46, dtype=np.float32)
    q_ref = np.ones((12, 46), dtype=np.float32)
    qdot_ref = np.ones((12, 46), dtype=np.float32) * 0.5
    prev = np.zeros(10, dtype=np.float32)
    target_keys = np.zeros((12, 88), dtype=np.float32)
    fingertips = np.zeros((12, 30), dtype=np.float32)
    features = build_tracking_features(
        q=q,
        qdot=qdot,
        q_ref=q_ref,
        qdot_ref=qdot_ref,
        t=0,
        previous_action=prev,
        target_keys=target_keys,
        fingertips=fingertips,
        spec=FeatureSpec(lookahead_steps=(1, 5, 10)),
    )
    assert features.dtype == np.float32
    assert features.shape == (46 * 7 + 10 + 88 + 30,)
