from __future__ import annotations

import numpy as np

from partita.primitives.clustering import transform_features


def assign_nearest_primitives(features: np.ndarray, library: dict) -> np.ndarray:
    transformed = transform_features(features, library["scaler"], library.get("pca"))
    return library["clusterer"].predict(transformed).astype(int)
