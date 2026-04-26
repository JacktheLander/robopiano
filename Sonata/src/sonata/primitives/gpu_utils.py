from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA


@dataclass
class GpuBackend:
    name: str
    active: bool
    runtime: dict[str, Any] = field(default_factory=dict)
    details: str = ""

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "active": bool(self.active),
            "details": self.details,
        }


def resolve_gpu_backend(config: dict[str, Any], logger: logging.Logger | None = None) -> GpuBackend:
    if not bool(config.get("gpu_acceleration", False)):
        return GpuBackend(name="cpu", active=False, details="gpu_acceleration_disabled")

    preferences = _normalize_preferences(config.get("gpu_backend_preference", ["rapids", "cupy", "torch"]))
    for preference in preferences:
        if preference == "rapids":
            backend = _try_rapids_backend()
        elif preference == "cupy":
            backend = _try_cupy_backend()
        elif preference == "torch":
            backend = _try_torch_backend()
        else:
            continue
        if backend.active:
            if logger is not None:
                logger.info("Stage 1 GPU acceleration active via %s.", backend.details or backend.name)
            return backend
    if logger is not None:
        logger.info("Stage 1 GPU acceleration requested but no supported backend was available; using CPU.")
    return GpuBackend(name="cpu", active=False, details="no_supported_backend")


def fit_pca_embedding(
    *,
    train_scaled: np.ndarray,
    all_scaled: np.ndarray,
    n_components: int,
    random_state: int,
    backend: GpuBackend,
    subsample_limit: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    max_components = min(int(n_components), train_scaled.shape[0], train_scaled.shape[1])
    n_components = max(max_components, 1)
    if not backend.active:
        pca = PCA(n_components=n_components, random_state=random_state)
        train_reduced = pca.fit_transform(train_scaled)
        all_reduced = pca.transform(all_scaled)
        return train_reduced, all_reduced, {"type": "sklearn_pca", "model": pca, "backend": backend.summary()}

    if backend.name == "rapids":
        try:
            cp = backend.runtime["cupy"]
            cu_pca = backend.runtime["cuml_pca"]
            fit_source = _maybe_subsample(train_scaled, subsample_limit, random_state)
            fit_matrix = cp.asarray(fit_source, dtype=cp.float32)
            transform_matrix = cp.asarray(all_scaled, dtype=cp.float32)
            model = cu_pca(n_components=n_components, random_state=random_state)
            model.fit(fit_matrix)
            train_reduced = np.asarray(model.transform(cp.asarray(train_scaled, dtype=cp.float32)).get(), dtype=np.float32)
            all_reduced = np.asarray(model.transform(transform_matrix).get(), dtype=np.float32)
            return train_reduced, all_reduced, {
                "type": "rapids_pca",
                "backend": backend.summary(),
                "components": np.asarray(model.components_.get(), dtype=np.float32),
                "mean": np.asarray(model.mean_.get(), dtype=np.float32),
                "fit_rows": int(fit_source.shape[0]),
            }
        except Exception:
            pass

    if backend.name == "cupy":
        try:
            cp = backend.runtime["cupy"]
            return _fit_svd_embedding_with_cupy(
                train_scaled=train_scaled,
                all_scaled=all_scaled,
                n_components=n_components,
                random_state=random_state,
                cp=cp,
                backend=backend,
                subsample_limit=subsample_limit,
            )
        except Exception:
            pass

    if backend.name == "torch":
        try:
            torch = backend.runtime["torch"]
            return _fit_svd_embedding_with_torch(
                train_scaled=train_scaled,
                all_scaled=all_scaled,
                n_components=n_components,
                random_state=random_state,
                torch=torch,
                backend=backend,
                subsample_limit=subsample_limit,
            )
        except Exception:
            pass

    pca = PCA(n_components=n_components, random_state=random_state)
    train_reduced = pca.fit_transform(train_scaled)
    all_reduced = pca.transform(all_scaled)
    return train_reduced, all_reduced, {"type": "sklearn_pca_fallback", "model": pca, "backend": backend.summary()}


def screen_kmeans_candidates(
    *,
    reduced_matrix: np.ndarray,
    k_candidates: list[int],
    random_state: int,
    backend: GpuBackend,
    subsample_limit: int,
) -> list[dict[str, Any]]:
    if reduced_matrix.size == 0:
        return []
    subset = _maybe_subsample(reduced_matrix, subsample_limit, random_state)
    rows: list[dict[str, Any]] = []
    for k in sorted({max(int(k), 1) for k in k_candidates}):
        if subset.shape[0] <= k:
            continue
        inertia = _compute_kmeans_inertia(subset=subset, k=k, random_state=random_state, backend=backend)
        rows.append({"k": int(k), "screen_inertia": float(inertia), "screen_rows": int(subset.shape[0])})
    return rows


def _compute_kmeans_inertia(
    *,
    subset: np.ndarray,
    k: int,
    random_state: int,
    backend: GpuBackend,
) -> float:
    if backend.name == "rapids":
        try:
            cp = backend.runtime["cupy"]
            cu_kmeans = backend.runtime["cuml_kmeans"]
            model = cu_kmeans(n_clusters=int(k), random_state=random_state, init="k-means||", max_iter=64)
            data = cp.asarray(subset, dtype=cp.float32)
            model.fit(data)
            return float(model.inertia_)
        except Exception:
            pass
    model = MiniBatchKMeans(
        n_clusters=int(k),
        random_state=random_state,
        batch_size=min(max(len(subset) // 4, 64), 4096),
        n_init=3,
    )
    model.fit(subset)
    return float(model.inertia_)


def _fit_svd_embedding_with_cupy(
    *,
    train_scaled: np.ndarray,
    all_scaled: np.ndarray,
    n_components: int,
    random_state: int,
    cp: Any,
    backend: GpuBackend,
    subsample_limit: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    del random_state
    fit_source = _maybe_subsample(train_scaled, subsample_limit, 0)
    fit_matrix = cp.asarray(fit_source, dtype=cp.float32)
    mean = fit_matrix.mean(axis=0, keepdims=True)
    centered_fit = fit_matrix - mean
    _, _, vh = cp.linalg.svd(centered_fit, full_matrices=False)
    components = vh[:n_components]
    train_reduced = np.asarray(((cp.asarray(train_scaled, dtype=cp.float32) - mean) @ components.T).get(), dtype=np.float32)
    all_reduced = np.asarray(((cp.asarray(all_scaled, dtype=cp.float32) - mean) @ components.T).get(), dtype=np.float32)
    return train_reduced, all_reduced, {
        "type": "cupy_svd",
        "backend": backend.summary(),
        "components": np.asarray(components.get(), dtype=np.float32),
        "mean": np.asarray(mean.get(), dtype=np.float32).reshape(-1),
        "fit_rows": int(fit_source.shape[0]),
    }


def _fit_svd_embedding_with_torch(
    *,
    train_scaled: np.ndarray,
    all_scaled: np.ndarray,
    n_components: int,
    random_state: int,
    torch: Any,
    backend: GpuBackend,
    subsample_limit: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    del random_state
    fit_source = _maybe_subsample(train_scaled, subsample_limit, 0)
    device = backend.runtime.get("device", "cuda")
    fit_tensor = torch.as_tensor(fit_source, dtype=torch.float32, device=device)
    mean = fit_tensor.mean(dim=0, keepdim=True)
    centered_fit = fit_tensor - mean
    _, _, vh = torch.linalg.svd(centered_fit, full_matrices=False)
    components = vh[:n_components]
    train_tensor = torch.as_tensor(train_scaled, dtype=torch.float32, device=device)
    all_tensor = torch.as_tensor(all_scaled, dtype=torch.float32, device=device)
    train_reduced = ((train_tensor - mean) @ components.T).detach().cpu().numpy().astype(np.float32)
    all_reduced = ((all_tensor - mean) @ components.T).detach().cpu().numpy().astype(np.float32)
    return train_reduced, all_reduced, {
        "type": "torch_svd",
        "backend": backend.summary(),
        "components": components.detach().cpu().numpy().astype(np.float32),
        "mean": mean.detach().cpu().numpy().astype(np.float32).reshape(-1),
        "fit_rows": int(fit_source.shape[0]),
    }


def _maybe_subsample(array: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if int(limit) <= 0 or array.shape[0] <= int(limit):
        return np.asarray(array, dtype=np.float32)
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(array.shape[0], size=int(limit), replace=False))
    return np.asarray(array[indices], dtype=np.float32)


def _normalize_preferences(value: Any) -> list[str]:
    if isinstance(value, str):
        parts = [item.strip().lower() for item in value.split(",")]
        return [item for item in parts if item]
    if isinstance(value, (list, tuple)):
        return [str(item).strip().lower() for item in value if str(item).strip()]
    return ["rapids", "cupy", "torch"]


def _try_rapids_backend() -> GpuBackend:
    try:
        import cupy as cp
        from cuml.cluster import KMeans as CuKMeans
        from cuml.decomposition import PCA as CuPCA
    except Exception:
        return GpuBackend(name="rapids", active=False, details="rapids_unavailable")
    return GpuBackend(
        name="rapids",
        active=True,
        runtime={"cupy": cp, "cuml_pca": CuPCA, "cuml_kmeans": CuKMeans},
        details="rapids/cuml",
    )


def _try_cupy_backend() -> GpuBackend:
    try:
        import cupy as cp
    except Exception:
        return GpuBackend(name="cupy", active=False, details="cupy_unavailable")
    return GpuBackend(name="cupy", active=True, runtime={"cupy": cp}, details="cupy")


def _try_torch_backend() -> GpuBackend:
    try:
        import torch
    except Exception:
        return GpuBackend(name="torch", active=False, details="torch_unavailable")
    if not torch.cuda.is_available():
        return GpuBackend(name="torch", active=False, details="torch_cuda_unavailable")
    return GpuBackend(
        name="torch",
        active=True,
        runtime={"torch": torch, "device": "cuda"},
        details=f"torch/{torch.cuda.get_device_name(0)}",
    )
