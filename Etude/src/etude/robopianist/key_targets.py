from __future__ import annotations

import numpy as np


def active_key_windows(target_keys: np.ndarray, radius: int = 2) -> np.ndarray:
    """Dilate target key activations by a small temporal window."""
    keys = np.asarray(target_keys, dtype=bool)
    if keys.ndim != 2:
        raise ValueError(f"target_keys must have shape [T, K], got {keys.shape}")
    out = keys.copy()
    for offset in range(1, radius + 1):
        out[offset:] |= keys[:-offset]
        out[:-offset] |= keys[offset:]
    return out
