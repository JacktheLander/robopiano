from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from etude.data.feature_builder import FeatureSpec, build_tracking_features
from etude.data.trajectory_io import finite_difference


@dataclass(frozen=True, slots=True)
class SyntheticTrackingBatch:
    q_ref: np.ndarray
    qdot_ref: np.ndarray
    features: np.ndarray
    actions: np.ndarray
    dt: float


def make_synthetic_tracking_batch(
    *,
    timesteps: int = 8,
    dof: int = 46,
    action_dim: int = 10,
    dt: float = 0.005,
    feature_spec: FeatureSpec | None = None,
) -> SyntheticTrackingBatch:
    if timesteps < 2:
        raise ValueError("timesteps must be at least 2")
    t = np.linspace(0.0, 1.0, timesteps, dtype=np.float32)[:, None]
    basis = np.linspace(0.1, 1.0, dof, dtype=np.float32)[None, :]
    q_ref = np.sin(2.0 * np.pi * t * basis).astype(np.float32)
    qdot_ref = finite_difference(q_ref, dt)
    prev = np.zeros(action_dim, dtype=np.float32)
    feature_rows = [
        build_tracking_features(
            q=q_ref[i],
            qdot=qdot_ref[i],
            q_ref=q_ref,
            qdot_ref=qdot_ref,
            t=i,
            previous_action=prev,
            spec=feature_spec,
        )
        for i in range(timesteps)
    ]
    actions = np.tanh(q_ref[:, :action_dim]).astype(np.float32)
    return SyntheticTrackingBatch(
        q_ref=q_ref,
        qdot_ref=qdot_ref,
        features=np.stack(feature_rows).astype(np.float32),
        actions=actions,
        dt=float(dt),
    )
