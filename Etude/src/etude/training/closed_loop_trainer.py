from __future__ import annotations

import numpy as np


def perturb_state(obs: dict[str, np.ndarray], std: float, rng: np.random.Generator | None = None) -> dict:
    """Apply small q/qdot perturbations for closed-loop recovery training."""
    rng = rng or np.random.default_rng()
    out = {key: np.array(value, copy=True) for key, value in obs.items()}
    for key in ("q", "qdot"):
        if key in out:
            out[key] = out[key] + rng.normal(0.0, std, size=out[key].shape).astype(np.float32)
    return out
