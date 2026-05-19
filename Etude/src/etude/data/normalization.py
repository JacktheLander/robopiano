from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class NormalizationStats:
    mean: np.ndarray
    std: np.ndarray


def compute_normalization_stats(data: np.ndarray, epsilon: float = 1e-6) -> NormalizationStats:
    array = np.asarray(data, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"data must have shape [N, D], got {array.shape}")
    mean = array.mean(axis=0).astype(np.float32)
    std = np.maximum(array.std(axis=0).astype(np.float32), epsilon)
    return NormalizationStats(mean=mean, std=std)


def normalize(data: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    array = np.asarray(data, dtype=np.float32)
    return ((array - stats.mean) / stats.std).astype(np.float32)
