from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def fit_kmeans_features(features: np.ndarray, num_primitives: int, pca_dim: int, random_seed: int):
    if features.ndim != 2 or features.shape[0] < 1:
        raise RuntimeError(f"Expected 2D feature matrix, got {features.shape}")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    max_pca = min(int(pca_dim), scaled.shape[0], scaled.shape[1])
    pca = None
    transformed = scaled
    if max_pca >= 2 and max_pca < scaled.shape[1]:
        pca = PCA(n_components=max_pca, random_state=random_seed)
        transformed = pca.fit_transform(scaled)
    k = max(1, min(int(num_primitives), transformed.shape[0]))
    clusterer = KMeans(n_clusters=k, random_state=random_seed, n_init=10)
    labels = clusterer.fit_predict(transformed)
    return scaler, pca, clusterer, labels, transformed


def transform_features(features: np.ndarray, scaler, pca):
    scaled = scaler.transform(features)
    if pca is not None:
        return pca.transform(scaled)
    return scaled
