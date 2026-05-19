from __future__ import annotations

import numpy as np

from etude.core.plan_bundle import PlanBundle
from etude.data.feature_builder import FeatureSpec, build_tracking_features
from etude.features.registry import resolve_feature_block


def test_plan_bundle_and_feature_registry_contracts() -> None:
    bundle = PlanBundle(
        q_ref=np.zeros((4, 46), dtype=np.float32),
        dt=0.01,
        target_keys=np.zeros((4, 88), dtype=np.float32),
        key_state=np.zeros((4, 88), dtype=np.float32),
        planner_confidence=np.linspace(0.2, 0.8, 4, dtype=np.float32),
    )
    bundle.validate_step_aligned()
    assert bundle.current_key_state is bundle.key_state
    assert bundle.desired_keys is bundle.target_keys
    assert bundle.goal_keys is bundle.target_keys
    block = resolve_feature_block("etude.data.feature_builder:build_tracking_features")
    features = block(
        q=np.zeros(46, dtype=np.float32),
        qdot=np.zeros(46, dtype=np.float32),
        q_ref=bundle.q_ref,
        qdot_ref=np.zeros_like(bundle.q_ref),
        t=0,
        previous_action=np.zeros(10, dtype=np.float32),
        spec=FeatureSpec(lookahead_steps=(1,)),
    )
    assert features.shape == (46 * 5 + 10,)
