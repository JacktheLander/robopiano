from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from sonata.primitives.guards import primitive_count_guard
from sonata.utils.io import write_table

LOGGER = logging.getLogger(__name__)

NOISE_PRIMITIVE_ID = "primitive_noise"


def fit_hdbscan_then_local_gmm(
    *,
    segment_df: pd.DataFrame,
    feature_matrix: np.ndarray,
    feature_names: list[str],
    config: dict[str, Any],
    primitive_root: Path,
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], np.ndarray]:
    log = logger or LOGGER
    try:
        import hdbscan  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "hdbscan is required for primitive_discovery_method=hdbscan_then_local_gmm. "
            "Install with: pip install hdbscan"
        ) from exc

    train_mask = segment_df["split"].astype(str).to_numpy() == "train"
    if not np.any(train_mask):
        train_mask[:] = True

    scaler = StandardScaler()
    train_x = np.asarray(feature_matrix[train_mask], dtype=np.float64)
    scaler.fit(train_x)
    all_emb = scaler.transform(np.asarray(feature_matrix, dtype=np.float64)).astype(np.float32)

    cap = int(config.get("embedding_dim_cap", config.get("pca_components", 64)))
    n_train = int(train_x.shape[0])
    n_dim = int(all_emb.shape[1])
    pca_components = min(cap, max(1, n_train - 1), max(1, n_dim))
    pca = PCA(n_components=pca_components, random_state=int(config["seed"]))
    train_pca = pca.fit_transform(all_emb[train_mask])
    all_pca = pca.transform(all_emb).astype(np.float32)

    max_fit = int(config.get("max_fit_rows", 200000))
    rng = np.random.default_rng(int(config["seed"]))
    train_idx_all = np.flatnonzero(train_mask)
    if n_train > max_fit:
        sub_idx = rng.choice(train_idx_all, size=max_fit, replace=False)
        log.warning("Subsampled %d / %d train rows for HDBSCAN fit.", max_fit, n_train)
    else:
        sub_idx = train_idx_all
    fit_matrix = all_pca[sub_idx]

    min_cluster_size = int(config.get("hdbscan_min_cluster_size", 48))
    min_samples = int(config.get("hdbscan_min_samples", 12))
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    clusterer.fit(fit_matrix)
    labels_all, strengths = hdbscan.approximate_predict(clusterer, all_pca)
    labels_all = np.asarray(labels_all, dtype=np.int64)
    strengths = np.asarray(strengths, dtype=np.float32)
    if strengths.size != labels_all.size:
        strengths = np.ones_like(labels_all, dtype=np.float32)

    labels_all, split_diag = _local_gmm_refine_large_clusters(
        all_pca=all_pca,
        labels=labels_all,
        config=config,
        rng=rng,
        logger=log,
    )

    labels_all, merge_diag = _merge_to_cap(all_pca, labels_all, int(config["max_clusters_total"]), rng)
    uniq_pos = {int(x) for x in labels_all.tolist() if int(x) >= 0}
    primitive_count_guard(count=len(uniq_pos), max_clusters_total=int(config["max_clusters_total"]), context="(HDBSCAN+merge)")

    assignments_df, id_map = _labels_to_assignments(segment_df, labels_all, strengths, all_pca=all_pca, id_prefix="primitive")

    noise_mask = assignments_df["primitive_id"].astype(str) == NOISE_PRIMITIVE_ID
    noise_df = assignments_df.loc[noise_mask].copy()
    if not noise_df.empty:
        noise_path = primitive_root / "clustering" / "noise_segments"
        noise_path.parent.mkdir(parents=True, exist_ok=True)
        write_table(noise_df, noise_path)
        log.info("Wrote %d noise rows to %s", len(noise_df), noise_path.with_suffix(".csv"))

    kept_mask = ~noise_mask
    log.info(
        "HDBSCAN path: labeled_primitive_count=%d noise_rows=%d (noise excluded from tokens/GMR)",
        int(assignments_df.loc[kept_mask, "primitive_id"].nunique()),
        int(noise_mask.sum()),
    )

    sweep_rows = [
        {
            "method": "hdbscan_then_local_gmm",
            "hdbscan_min_cluster_size": min_cluster_size,
            "hdbscan_min_samples": min_samples,
            "pca_components": pca_components,
            "local_gmm_diag_json": json.dumps(split_diag, default=str),
            "merge_diag_json": json.dumps(merge_diag, default=str),
            "primitive_map_json": json.dumps(id_map, default=str),
        }
    ]
    sweep_df = pd.DataFrame(sweep_rows)
    bundle: dict[str, Any] = {
        "method": "hdbscan_then_local_gmm",
        "scaler": scaler,
        "pca": pca,
        "feature_names": feature_names,
        "hdbscan_clusterer": clusterer,
        "noise_primitive_id": NOISE_PRIMITIVE_ID,
    }
    return assignments_df, sweep_df, bundle, all_pca


def _local_gmm_refine_large_clusters(
    *,
    all_pca: np.ndarray,
    labels: np.ndarray,
    config: dict[str, Any],
    rng: np.random.Generator,
    logger: logging.Logger,
) -> tuple[np.ndarray, dict[str, Any]]:
    labels = np.asarray(labels, dtype=np.int64).copy()
    large_thr = int(config.get("large_cluster_min_size", 256))
    max_k = int(config.get("max_large_cluster_refine_k", 3))
    cov_type = str(config.get("local_gmm_covariance_type", "diag"))
    diag: dict[str, Any] = {"refined": []}
    unique = sorted({int(x) for x in labels.tolist() if int(x) >= 0})
    next_label = int(labels.max()) + 1
    for lab in unique:
        mask = labels == lab
        count = int(mask.sum())
        if count < large_thr:
            continue
        points = all_pca[mask]
        if points.shape[0] < 2 * int(config.get("min_cluster_size", 96)):
            continue
        best_bic = float("inf")
        best_model = None
        best_k = 1
        for k in (2, 3):
            if k > max_k:
                break
            if points.shape[0] < k * 8:
                continue
            gm = GaussianMixture(
                n_components=k,
                covariance_type=cov_type,
                reg_covar=float(config.get("gmm_reg_covar", 1e-4)),
                random_state=int(rng.integers(0, 2**31 - 1)),
            )
            gm.fit(points)
            bic = float(gm.bic(points))
            if bic < best_bic:
                best_bic = bic
                best_model = gm
                best_k = k
        if best_model is None or best_k < 2:
            continue
        sub = best_model.predict(points).astype(np.int64)
        idxs = np.flatnonzero(mask)
        for sub_k in range(best_k):
            sub_mask = sub == sub_k
            new_lab = next_label
            next_label += 1
            labels[idxs[sub_mask]] = new_lab
        diag["refined"].append({"parent": lab, "k": best_k, "bic": best_bic})
        logger.info("Local GMM split cluster %s into %d subcomponents (bic=%.2f).", lab, best_k, best_bic)
    return labels, diag


def _merge_to_cap(
    all_pca: np.ndarray,
    labels: np.ndarray,
    max_clusters: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    labels = np.asarray(labels, dtype=np.int64).copy()
    merge_steps = 0
    while True:
        uniq = sorted({int(x) for x in labels.tolist() if int(x) >= 0})
        if len(uniq) <= max_clusters:
            break
        centroids = []
        for u in uniq:
            m = all_pca[labels == u].mean(axis=0)
            centroids.append((u, m / (np.linalg.norm(m) + 1e-8)))
        best_pair = None
        best_sim = -1.0
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                sim = float(np.dot(centroids[i][1], centroids[j][1]))
                if sim > best_sim:
                    best_sim = sim
                    best_pair = (centroids[i][0], centroids[j][0])
        if best_pair is None:
            break
        a, b = best_pair
        size_a = int((labels == a).sum())
        size_b = int((labels == b).sum())
        survivor, victim = (a, b) if size_a >= size_b else (b, a)
        labels[labels == victim] = survivor
        merge_steps += 1
    return labels, {"merge_steps": merge_steps}


def _labels_to_assignments(
    segment_df: pd.DataFrame,
    labels: np.ndarray,
    strengths: np.ndarray,
    *,
    all_pca: np.ndarray,
    id_prefix: str,
) -> tuple[pd.DataFrame, dict[int, str]]:
    out = segment_df.copy()
    uniq = sorted({int(x) for x in labels.tolist() if int(x) >= 0})
    id_map: dict[int, str] = {u: f"{id_prefix}_{u:03d}" for u in uniq}
    prim_ids = []
    prim_labels = []
    confidences = []
    for lab, conf in zip(labels.tolist(), strengths.tolist(), strict=False):
        if int(lab) < 0:
            prim_ids.append(NOISE_PRIMITIVE_ID)
            prim_labels.append(-1)
            confidences.append(float(min(1.0, max(0.0, conf))))
        else:
            pid = id_map[int(lab)]
            prim_ids.append(pid)
            prim_labels.append(int(lab))
            confidences.append(float(min(1.0, max(0.0, conf))))
    out["primitive_id"] = prim_ids
    out["primitive_label"] = prim_labels
    out["assignment_confidence"] = confidences
    out["discovery_partition"] = "hdbscan_safe"
    out["gmm_best_k"] = 0
    out["hybrid_selection_score"] = 0.0
    latent_dim = int(all_pca.shape[1]) if all_pca.ndim == 2 else 0
    for idx in range(latent_dim):
        out[f"latent_{idx:03d}"] = all_pca[:, idx]
    out["embedding_norm"] = np.linalg.norm(all_pca, axis=1) if latent_dim > 0 else 0.0
    return out, id_map


def prune_clusters_pre_gmr(
    assignments_df: pd.DataFrame,
    feature_matrix: np.ndarray,
    *,
    config: dict[str, Any],
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Merge tiny / low-confidence clusters into nearest centroid neighbors (embedding space)."""
    log = logger or LOGGER
    frame = assignments_df.copy()
    train_mask = frame["split"].astype(str).to_numpy() == "train"
    noise = frame["primitive_id"].astype(str) == NOISE_PRIMITIVE_ID
    active = (~noise) & train_mask
    if not np.any(active):
        return frame, {"pruned": 0}

    labels = frame["primitive_id"].astype(str).to_numpy()
    unique_ids = sorted(set(labels.tolist()) - {NOISE_PRIMITIVE_ID})
    if not unique_ids:
        return frame, {"pruned": 0}

    X = np.asarray(feature_matrix, dtype=np.float32)
    min_conf = float(config.get("min_assignment_confidence", 0.12))
    min_size = int(config.get("min_cluster_size", 96))
    cos_thr = float(config.get("primitive_merge_cosine_threshold", 0.985))

    centroids: dict[str, np.ndarray] = {}
    sizes: dict[str, int] = {}
    for pid in unique_ids:
        mask = (labels == pid) & active
        if not np.any(mask):
            sizes[pid] = 0
            continue
        rows = np.flatnonzero(mask)
        centroids[pid] = X[rows].mean(axis=0)
        sizes[pid] = int(len(rows))

    merges = 0
    for pid in list(unique_ids):
        if pid not in sizes or sizes[pid] < min_size:
            target = _nearest_primitive_centroid(pid, centroids, cos_thr)
            if target is None:
                continue
            frame.loc[frame["primitive_id"].astype(str) == pid, "primitive_id"] = target
            merges += 1
            log.info("Pruned small/low-support cluster %s -> %s", pid, target)

    labels2 = frame["primitive_id"].astype(str)
    low_conf = (~noise) & (frame["assignment_confidence"].astype(float) < min_conf)
    for pid in unique_ids:
        sel = low_conf & (labels2 == pid)
        if not sel.any():
            continue
        target = _nearest_primitive_centroid(pid, centroids, cos_thr)
        if target is not None and target != pid:
            frame.loc[sel, "primitive_id"] = target
            merges += int(sel.sum())

    diag = {"pruned": merges, "pre_prune_primitive_count": len(unique_ids)}
    return frame, diag


def _nearest_primitive_centroid(
    pid: str,
    centroids: dict[str, np.ndarray],
    cos_thr: float,
) -> str | None:
    if pid not in centroids:
        return None
    v = centroids[pid]
    v = v / (np.linalg.norm(v) + 1e-8)
    best = None
    best_sim = -2.0
    for other, c in centroids.items():
        if other == pid:
            continue
        u = c / (np.linalg.norm(c) + 1e-8)
        sim = float(np.dot(v, u))
        if sim > best_sim:
            best_sim = sim
            best = other
    if best is not None and best_sim >= cos_thr:
        return best
    return best
