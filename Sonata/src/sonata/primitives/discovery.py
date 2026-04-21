from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from sonata.primitives.control_latent import learn_control_latent
from sonata.primitives.guards import estimate_dir_bytes, normalize_primitive_storage_aliases, storage_guard
from sonata.primitives.safe_clustering import NOISE_PRIMITIVE_ID, fit_hdbscan_then_local_gmm, prune_clusters_pre_gmr
from sonata.data.indexer import scan_dataset
from sonata.data.loading import load_manifest
from sonata.primitives.features import extract_segment_features, resolve_gmr_resample_steps
from sonata.primitives.gmr import PhaseGMR
from sonata.primitives.segmenters import load_segment_arrays_from_bundle, run_segmentation
from sonata.primitives.slim_cache import (
    build_gmr_target,
    gmr_target_chunk_path,
    is_slim_chunk_name,
    resolve_online_storage_format,
    resolve_slim_cache_paths,
    save_raw_segment_chunks_enabled,
    summarize_slim_cache,
)
from sonata.primitives.tokenization import add_token_columns, build_vocabulary_payload
from sonata.primitives.visualization import plot_gmr_reconstruction, plot_primitive_frequency, plot_usage_entropy
from sonata.utils.io import read_json, read_table, save_npz, write_json, write_table
from sonata.utils.wandb import WandbRun

def _read_stage_status(manifest_path: Path) -> str:
    if not manifest_path.exists():
        return "missing"
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return str(payload.get("status", "unknown"))

def run_primitive_pipeline(config: dict[str, Any], logger: logging.Logger) -> dict[str, Path]:
    config = normalize_primitive_storage_aliases(dict(config))
    config.setdefault("online_segment_processing", True if config.get("write_slim_cache", True) else False)
    config.setdefault("save_raw_segment_chunks", save_raw_segment_chunks_enabled(config))
    config.setdefault("online_storage_format", resolve_online_storage_format(config))
    config.setdefault("gmr_resample_steps", resolve_gmr_resample_steps(config))
    data_output = Path(config["data_output_root"]).resolve()
    primitive_root = Path(config["output_root"]).resolve()
    primitive_root.mkdir(parents=True, exist_ok=True)
    write_json(config, primitive_root / "run_config.json")
    run_name = f"{config['experiment_name']}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    wandb_run = WandbRun(
        config.get("wandb"),
        run_name=run_name,
        config_payload=config,
        logger=logger,
        job_type="primitives",
        tags=["sonata", "primitives"],
    )
    try:
        manifest_base = data_output / str(config["data_manifest_name"])
        if not manifest_base.with_suffix(".csv").exists():
            scan_dataset(config=config["data_config"], logger=logger)
        manifest_df = load_manifest(manifest_base)
        
        segment_outputs = run_segmentation(manifest_df=manifest_df, output_dir=primitive_root, config=config)
        if _read_stage_status(segment_outputs["manifest_path"]) != "completed":
            raise RuntimeError("Segmentation did not complete successfully; resume Stage 1 before continuing.")
        compact_store_manifest = segment_outputs.get("compact_store_manifest_path")
        if compact_store_manifest is not None and compact_store_manifest.exists():
            store_summary = read_json(compact_store_manifest)
        else:
            store_summary = summarize_slim_cache(resolve_slim_cache_paths(primitive_root, config))

        segment_df = read_table(segment_outputs["segment_table_base"])

        feature_outputs = extract_segment_features(
            segment_df=segment_df,
            segments_dir=primitive_root / "segments",
            output_dir=primitive_root,
            config=config,
        )
        if _read_stage_status(feature_outputs["manifest_path"]) != "completed":
            raise RuntimeError("Feature extraction did not complete successfully; resume Stage 1 before continuing.")
            
        feature_bundle = np.load(feature_outputs["feature_bundle_path"], allow_pickle=True)
        feature_matrix = np.asarray(feature_bundle["feature_matrix"], dtype=np.float32)
        feature_names = [str(item) for item in feature_bundle["feature_names"].tolist()]
        wandb_run.summary(
            {
                "stage": "primitives",
                "output_root": str(primitive_root),
                "data_output_root": str(data_output),
                "dataset/num_rows": int(len(manifest_df)),
                "segments/count": int(len(segment_df)),
                "features/dim": int(feature_matrix.shape[1]) if feature_matrix.ndim == 2 else 0,
                "gmr_target/steps": int(store_summary.get("gmr_target_steps", store_summary.get("gmr_horizon", 0))),
                "gmr_target/dim": int(store_summary.get("gmr_target_dim", store_summary.get("gmr_dim", 0))),
                "stage1_storage/bytes": int(store_summary.get("total_bytes_on_disk", 0)),
            }
        )

        discovery_method = str(config.get("primitive_discovery_method", "gmm_sweep"))
        discovery_diag: dict[str, Any] = {}
        if discovery_method == "hdbscan_then_local_gmm":
            assignments_df, sweep_df, bundle, _ = fit_hdbscan_then_local_gmm(
                segment_df=segment_df,
                feature_matrix=feature_matrix,
                feature_names=feature_names,
                config=config,
                primitive_root=primitive_root,
                logger=logger,
            )
            pre_prune = int(assignments_df.loc[assignments_df["primitive_id"].astype(str) != NOISE_PRIMITIVE_ID, "primitive_id"].nunique())
            assignments_df, prune_diag = prune_clusters_pre_gmr(
                assignments_df, feature_matrix, config=config, logger=logger
            )
            discovery_diag["pre_prune_primitive_count"] = pre_prune
            discovery_diag["prune_diag"] = prune_diag
            discovery_diag["noise_rows"] = int((assignments_df["primitive_id"].astype(str) == NOISE_PRIMITIVE_ID).sum())
        else:
            assignments_df, sweep_df, bundle = fit_primitive_gmm(
                segment_df=segment_df,
                feature_matrix=feature_matrix,
                feature_names=feature_names,
                config=config,
            )
        clustering_dir = primitive_root / "clustering"
        clustering_dir.mkdir(parents=True, exist_ok=True)
        assignments_base = clustering_dir / "segment_assignments"
        sweep_base = clustering_dir / "gmm_sweep"
        write_table(assignments_df, assignments_base)
        write_table(sweep_df, sweep_base)
        joblib.dump(bundle, clustering_dir / "primitive_model_bundle.joblib")

        soft_b = config.get("max_storage_bytes_soft")
        hard_b = config.get("max_storage_bytes_hard")
        if hard_b is not None:
            storage_guard(
                bytes_written=estimate_dir_bytes(primitive_root),
                soft_limit=int(soft_b or hard_b),
                hard_limit=int(hard_b),
                logger=logger,
            )

        library_df, gmr_bundle, gmr_diag = fit_gmr_library(
            assignments_df=assignments_df,
            segments_dir=primitive_root / "segments",
            output_dir=primitive_root,
            config=config,
            logger=logger,
        )
        write_table(library_df, primitive_root / "library" / "primitive_library")
        joblib.dump(gmr_bundle, primitive_root / "library" / "primitive_gmr_bundle.joblib")
        online_selection_summary = None
        if bool(dict(config.get("online_selection", {})).get("enabled", False)):
            assignments_df, library_df, online_selection_summary = apply_online_selection_and_cleanup(
                primitive_root=primitive_root,
                assignments_df=assignments_df,
                library_df=library_df,
                config=config,
                logger=logger,
            )
            write_table(assignments_df, assignments_base)
            library_df, gmr_bundle, gmr_diag = fit_gmr_library(
                assignments_df=assignments_df,
                segments_dir=primitive_root / "segments",
                output_dir=primitive_root,
                config=config,
                logger=logger,
            )
            write_table(library_df, primitive_root / "library" / "primitive_library")
            joblib.dump(gmr_bundle, primitive_root / "library" / "primitive_gmr_bundle.joblib")

        token_assignments = assignments_df.loc[assignments_df["primitive_id"].astype(str) != NOISE_PRIMITIVE_ID].copy()
        if token_assignments.empty or library_df.empty:
            raise RuntimeError(
                "Primitive tokenization requires at least one non-noise assignment and one library row. "
                "Loosen clustering thresholds or increase data coverage."
            )
        token_df = add_token_columns(
            assignments_df=token_assignments.merge(
                library_df[["primitive_id", "reconstruction_mse"]], on="primitive_id", how="left"
            ),
            num_duration_buckets=int(config["num_duration_buckets"]),
            num_dynamics_buckets=int(config["num_dynamics_buckets"]),
        )
        token_base = primitive_root / "tokens" / "primitive_tokens"
        write_table(token_df, token_base)
        write_json(build_vocabulary_payload(token_df), primitive_root / "tokens" / "primitive_vocabulary.json")

        combined_diag = dict(discovery_diag)
        combined_diag.update(gmr_diag)
        metrics = compute_stage1_metrics(
            assignments_df=assignments_df,
            sweep_df=sweep_df,
            library_df=library_df,
            storage_summary=store_summary,
            online_selection_summary=online_selection_summary,
            discovery_diag=combined_diag,
        )
        metrics_dir = primitive_root / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = metrics_dir / "stage1_metrics.json"
        write_json(metrics, metrics_path)
        scalar_metrics = {key: value for key, value in metrics.items() if not isinstance(value, list)}
        wandb_run.log(scalar_metrics)
        wandb_run.summary(scalar_metrics | {"status": "completed", "metrics_path": str(metrics_path)})

        plot_dir = primitive_root / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_primitive_frequency(assignments_df, plot_dir / "primitive_frequency.png")
        if not library_df.empty:
            plot_gmr_reconstruction(library_df, plot_dir / "primitive_gmr_reconstruction.png")
        plot_usage_entropy(assignments_df, plot_dir / "primitive_usage_entropy.png")

        wandb_run.log_artifact_bundle(
            artifact_name=f"{run_name}-outputs",
            artifact_type="dataset",
            entries={
                "clustering": clustering_dir,
                "library": primitive_root / "library",
                "tokens": primitive_root / "tokens",
                "metrics": metrics_dir,
                "plots": plot_dir,
                "run_config.json": primitive_root / "run_config.json",
            },
            aliases=["latest"],
            metadata={"stage": "primitives", "output_root": str(primitive_root)},
        )

        return {
            "segment_table_base": segment_outputs["segment_table_base"],
            "compact_store_manifest_path": segment_outputs.get(
                "compact_store_manifest_path",
                resolve_slim_cache_paths(primitive_root, config).root / "compact_store_manifest.json",
            ),
            "assignments_base": assignments_base,
            "library_base": primitive_root / "library" / "primitive_library",
            "tokens_base": token_base,
            "metrics_path": metrics_path,
        }
    finally:
        wandb_run.finish()


def fit_primitive_gmm(segment_df: pd.DataFrame, feature_matrix: np.ndarray, feature_names: list[str], config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train_mask = segment_df["split"].astype(str).to_numpy() == "train"
    if not np.any(train_mask):
        train_mask[:] = True
    latent_result = learn_control_latent(feature_matrix=feature_matrix, segment_df=segment_df, config=config)
    all_embeddings = np.asarray(latent_result.latent_matrix, dtype=np.float32)
    train_embeddings = all_embeddings[train_mask]
    pca = None
    pca_components = min(int(config["pca_components"]), train_embeddings.shape[0], train_embeddings.shape[1])
    if pca_components > 0 and all_embeddings.shape[1] > pca_components:
        pca = PCA(n_components=max(1, pca_components), random_state=int(config["seed"]))
        train_embeddings = pca.fit_transform(train_embeddings)
        all_embeddings = pca.transform(all_embeddings)

    assignments_df = segment_df.copy()
    latent_dim = int(all_embeddings.shape[1]) if all_embeddings.ndim == 2 else 0
    latent_columns = [f"latent_{idx:03d}" for idx in range(latent_dim)]
    for idx, column in enumerate(latent_columns):
        assignments_df[column] = all_embeddings[:, idx]
    assignments_df["primitive_label"] = -1
    assignments_df["primitive_id"] = ""
    assignments_df["assignment_confidence"] = 0.0
    assignments_df["embedding_norm"] = np.linalg.norm(all_embeddings, axis=1) if latent_dim > 0 else 0.0
    assignments_df["discovery_partition"] = ""
    assignments_df["gmm_best_k"] = 0
    assignments_df["hybrid_selection_score"] = 0.0

    partitions = build_discovery_partitions(assignments_df, config)
    sweep_rows: list[dict[str, Any]] = []
    family_models: dict[str, Any] = {}
    primitive_counter = 0
    criterion = str(config["model_selection_metric"])
    for partition_key, partition_indices in partitions:
        partition_indices = np.asarray(partition_indices, dtype=np.int64)
        partition_embeddings = all_embeddings[partition_indices]
        partition_train_mask = train_mask[partition_indices]
        if not np.any(partition_train_mask):
            partition_train_mask[:] = True
        partition_train_embeddings = partition_embeddings[partition_train_mask]
        if partition_train_embeddings.size == 0:
            continue
        candidate_ks = candidate_k_values(
            num_examples=int(len(partition_train_embeddings)),
            base_candidates=config["gmm_k_candidates"],
            min_examples_per_cluster=int(config.get("min_segments_per_primitive", 4)),
        )
        evaluated: list[tuple[dict[str, Any], GaussianMixture]] = []
        for k in candidate_ks:
            model = GaussianMixture(
                n_components=int(k),
                covariance_type=str(config["gmm_covariance_type"]),
                reg_covar=float(config["gmm_reg_covar"]),
                random_state=int(config["seed"]),
            )
            model.fit(partition_train_embeddings)
            train_labels = model.predict(partition_train_embeddings)
            bic = float(model.bic(partition_train_embeddings))
            aic = float(model.aic(partition_train_embeddings))
            silhouette = _safe_silhouette(
                embedding_matrix=partition_train_embeddings,
                labels=train_labels,
                max_examples=int(config["silhouette_max_examples"]),
                seed=int(config["seed"]),
            )
            partition_frame = assignments_df.iloc[partition_indices[partition_train_mask]].copy().reset_index(drop=True)
            event_proxy = _cluster_event_proxy(
                frame=partition_frame,
                labels=train_labels,
                feature_matrix=feature_matrix[partition_indices[partition_train_mask]],
                feature_names=feature_names,
            )
            cross_song_reuse = _candidate_cross_song_reuse(partition_frame, train_labels)
            dominant_cluster_share = _dominant_cluster_share(train_labels)
            criterion_value = bic if criterion == "bic" else aic
            record = {
                "partition": partition_key,
                "k": int(k),
                "bic": bic,
                "aic": aic,
                "criterion_value": criterion_value,
                "silhouette": silhouette,
                "event_proxy_score": float(event_proxy["event_proxy_score"]),
                "action_consistency_score": float(event_proxy["action_consistency_score"]),
                "key_signature_purity": float(event_proxy["key_signature_purity"]),
                "phase_purity": float(event_proxy["phase_purity"]),
                "cross_song_reuse": float(cross_song_reuse),
                "dominant_cluster_share": float(dominant_cluster_share),
            }
            evaluated.append((record, model))
        shortlisted = shortlist_candidate_models(evaluated, config=config)
        if not shortlisted:
            continue
        for record, _ in shortlisted:
            record["hybrid_score"] = float(
                hybrid_model_score(
                    criterion_value=float(record["criterion_value"]),
                    silhouette=float(record["silhouette"]),
                    event_proxy_score=float(record["event_proxy_score"]),
                    action_consistency=float(record["action_consistency_score"]),
                    cross_song_reuse=float(record["cross_song_reuse"]),
                    dominant_cluster_share=float(record["dominant_cluster_share"]),
                )
            )
        best_record, best_model = max(
            shortlisted,
            key=lambda item: (
                float(item[0]["hybrid_score"]),
                -float(item[0]["criterion_value"]),
            ),
        )
        labels = best_model.predict(partition_embeddings)
        probabilities = best_model.predict_proba(partition_embeddings)
        confidence = probabilities.max(axis=1)
        unique_labels = sorted(set(int(item) for item in labels.tolist()))
        local_to_global = {label: primitive_counter + idx for idx, label in enumerate(unique_labels)}
        primitive_counter += len(unique_labels)
        assigned_ids = np.asarray([f"primitive_{local_to_global[int(label)]:03d}" for label in labels], dtype=object)
        assignments_df.loc[partition_indices, "primitive_label"] = [int(local_to_global[int(label)]) for label in labels]
        assignments_df.loc[partition_indices, "primitive_id"] = assigned_ids
        assignments_df.loc[partition_indices, "assignment_confidence"] = confidence
        assignments_df.loc[partition_indices, "discovery_partition"] = str(partition_key)
        assignments_df.loc[partition_indices, "gmm_best_k"] = int(best_record["k"])
        assignments_df.loc[partition_indices, "hybrid_selection_score"] = float(best_record["hybrid_score"])
        for record, _ in evaluated:
            record["selected_k"] = int(best_record["k"])
            record["selected"] = bool(int(record["k"]) == int(best_record["k"]))
            sweep_rows.append(record)
        family_models[str(partition_key)] = {
            "model": best_model,
            "selected_k": int(best_record["k"]),
            "partition_size": int(len(partition_indices)),
        }

    assignments_df = assignments_df.sort_values(["primitive_id", "segment_id"], kind="stable").reset_index(drop=True)
    sweep_df = pd.DataFrame(sweep_rows)
    bundle = {
        "scaler": latent_result.scaler,
        "pca": pca,
        "family_models": family_models,
        "feature_names": feature_names,
        "latent_mode": latent_result.mode,
        "latent_training_loss": latent_result.training_loss,
        "latent_dim": latent_result.latent_dim,
        "selected_k": int(assignments_df["primitive_id"].nunique()) if not assignments_df.empty else 0,
        "partitions": [partition for partition, _ in partitions],
    }
    return assignments_df, sweep_df, bundle


def build_discovery_partitions(segment_df: pd.DataFrame, config: dict[str, Any]) -> list[tuple[str, np.ndarray]]:
    family_cfg = dict(config.get("family_clustering", {}))
    if not bool(family_cfg.get("enabled", True)) or segment_df.empty:
        return [("all", np.arange(len(segment_df), dtype=np.int64))]
    frame = segment_df.copy()
    region_buckets = max(int(family_cfg.get("region_buckets", 4)), 1)
    coarse = frame.get("coarse_family", frame.get("heuristic_family", pd.Series(dtype=object))).fillna("other").astype(str)
    phase = frame.get("control_phase", pd.Series(["whole_event"] * len(frame))).fillna("whole_event").astype(str)
    key_center = pd.to_numeric(frame.get("key_center", pd.Series(np.zeros((len(frame),), dtype=np.float32))), errors="coerce").fillna(0.0)
    region = np.clip(np.floor(key_center.to_numpy(dtype=np.float32) * region_buckets).astype(np.int64), 0, region_buckets - 1)
    chord_bucket = np.clip(pd.to_numeric(frame.get("chord_size", pd.Series(np.zeros((len(frame),), dtype=np.float32))), errors="coerce").fillna(0).astype(int).to_numpy(), 0, 4)
    keys: list[str] = []
    for coarse_family, control_phase, region_bucket, chord_size in zip(
        coarse.tolist(),
        phase.tolist(),
        region.tolist(),
        chord_bucket.tolist(),
        strict=False,
    ):
        parts = [coarse_family]
        if bool(family_cfg.get("split_by_phase", True)):
            parts.append(control_phase)
        if bool(family_cfg.get("split_by_region", True)):
            parts.append(f"region{int(region_bucket)}")
        if bool(family_cfg.get("split_by_chord_size", True)) and coarse_family in {"single_press", "chord_press", "short_sequence"}:
            parts.append(f"chord{int(chord_size)}")
        keys.append("/".join(parts))
    frame["_partition_key"] = keys
    partitions: list[tuple[str, np.ndarray]] = []
    for partition_key, group in frame.groupby("_partition_key", sort=True):
        partitions.append((str(partition_key), group.index.to_numpy(dtype=np.int64)))
    return partitions


def candidate_k_values(num_examples: int, base_candidates: list[int] | tuple[int, ...], min_examples_per_cluster: int) -> list[int]:
    if num_examples <= 1:
        return [1]
    max_by_size = max(1, num_examples // max(min_examples_per_cluster, 1))
    values = sorted({max(1, min(int(candidate), max_by_size, num_examples)) for candidate in base_candidates})
    if not values:
        values = [1]
    if values[0] != 1:
        values.insert(0, 1)
    return sorted(set(values))


def shortlist_candidate_models(
    evaluated: list[tuple[dict[str, Any], GaussianMixture]],
    config: dict[str, Any],
) -> list[tuple[dict[str, Any], GaussianMixture]]:
    if not evaluated:
        return []
    shortlist_size = max(int(dict(config.get("hybrid_selection", {})).get("shortlist_size", 3)), 1)
    criterion = str(config["model_selection_metric"])
    sorted_by_criterion = sorted(
        evaluated,
        key=lambda item: float(item[0]["bic"] if criterion == "bic" else item[0]["aic"]),
    )
    return sorted_by_criterion[: min(shortlist_size, len(sorted_by_criterion))]


def hybrid_model_score(
    *,
    criterion_value: float,
    silhouette: float,
    event_proxy_score: float,
    action_consistency: float,
    cross_song_reuse: float,
    dominant_cluster_share: float,
) -> float:
    criterion_term = -math.log1p(max(float(criterion_value), 0.0))
    return (
        criterion_term
        + 1.50 * float(silhouette if not math.isnan(silhouette) else 0.0)
        + 2.00 * float(event_proxy_score)
        + 1.00 * float(action_consistency)
        + 0.75 * float(cross_song_reuse)
        - 1.25 * float(dominant_cluster_share)
    )


def _safe_silhouette(*, embedding_matrix: np.ndarray, labels: np.ndarray, max_examples: int, seed: int) -> float:
    if embedding_matrix.shape[0] <= 2 or len(np.unique(labels)) <= 1:
        return float("nan")
    sample_size = min(int(max_examples), len(embedding_matrix))
    if sample_size <= len(np.unique(labels)):
        return float("nan")
    rng = np.random.default_rng(int(seed))
    indices = rng.choice(len(embedding_matrix), size=sample_size, replace=False)
    return float(silhouette_score(embedding_matrix[indices], labels[indices]))


def _dominant_cluster_share(labels: np.ndarray) -> float:
    if labels.size == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    return float(counts.max() / max(counts.sum(), 1))


def _candidate_cross_song_reuse(frame: pd.DataFrame, labels: np.ndarray) -> float:
    if frame.empty or labels.size == 0 or "song_id" not in frame.columns:
        return 0.0
    source = frame.copy()
    source["_candidate_label"] = labels
    grouped = source.groupby("_candidate_label", sort=True)["song_id"].nunique()
    return float((grouped > 1).mean()) if not grouped.empty else 0.0


def _cluster_event_proxy(
    *,
    frame: pd.DataFrame,
    labels: np.ndarray,
    feature_matrix: np.ndarray,
    feature_names: list[str],
) -> dict[str, float]:
    if frame.empty or labels.size == 0:
        return {
            "event_proxy_score": 0.0,
            "action_consistency_score": 0.0,
            "key_signature_purity": 0.0,
            "phase_purity": 0.0,
        }
    source = frame.copy()
    source["_candidate_label"] = labels
    action_indices = [
        idx
        for idx, name in enumerate(feature_names)
        if name.startswith("action_mean_") or name.startswith("action_delta_") or name.startswith("traj_action_")
    ]
    key_purity_scores: list[float] = []
    phase_purity_scores: list[float] = []
    action_consistency_scores: list[float] = []
    for label, group in source.groupby("_candidate_label", sort=True):
        del label
        if "key_signature" in group.columns:
            key_counts = group["key_signature"].astype(str).value_counts(normalize=True)
            key_purity_scores.append(float(key_counts.iloc[0]) if not key_counts.empty else 0.0)
        if "control_phase" in group.columns:
            phase_counts = group["control_phase"].astype(str).value_counts(normalize=True)
            phase_purity_scores.append(float(phase_counts.iloc[0]) if not phase_counts.empty else 0.0)
        if action_indices:
            local_features = feature_matrix[group.index.to_numpy(dtype=np.int64)][:, action_indices]
            centroid = local_features.mean(axis=0, keepdims=True)
            mean_distance = float(np.linalg.norm(local_features - centroid, axis=1).mean())
            action_consistency_scores.append(float(1.0 / (1.0 + mean_distance)))
    key_purity = float(np.mean(key_purity_scores)) if key_purity_scores else 0.0
    phase_purity = float(np.mean(phase_purity_scores)) if phase_purity_scores else 0.0
    action_consistency = float(np.mean(action_consistency_scores)) if action_consistency_scores else 0.0
    return {
        "event_proxy_score": float((0.45 * key_purity) + (0.35 * phase_purity) + (0.20 * action_consistency)),
        "action_consistency_score": action_consistency,
        "key_signature_purity": key_purity,
        "phase_purity": phase_purity,
    }


def fit_gmr_library(
    assignments_df: pd.DataFrame,
    segments_dir: Path,
    output_dir: Path,
    config: dict[str, Any],
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    log = logger or logging.getLogger(__name__)
    library_dir = output_dir / "library"
    library_dir.mkdir(parents=True, exist_ok=True)
    slim_paths = resolve_slim_cache_paths(output_dir, config)
    slim_cache: dict[str, Any] = {}
    raw_cache: dict[str, Any] = {}
    grouped = list(assignments_df.groupby("primitive_id", sort=True))
    grouped = [(pid, g) for pid, g in grouped if str(pid) != NOISE_PRIMITIVE_ID]
    max_gmr = int(config.get("max_gmr_primitives_to_fit", 10**9))
    if len(grouped) > max_gmr:
        grouped.sort(key=lambda item: int((item[1]["split"].astype(str) == "train").sum()), reverse=True)
        grouped = grouped[:max_gmr]
        log.warning("GMR fit capped to max_gmr_primitives_to_fit=%d (largest train counts).", max_gmr)

    library_rows: list[dict[str, Any]] = []
    gmr_payload: dict[str, Any] = {"models": {}}
    prior_cfg = dict(config.get("prior_selection", {}))
    max_prototypes = max(int(prior_cfg.get("max_prototypes", 1)), 1)
    min_segments_per_prototype = max(int(prior_cfg.get("min_segments_per_prototype", 6)), 2)
    max_per_prim = int(config.get("max_segments_per_primitive_for_gmr", 10**9))
    min_cluster = int(config.get("min_cluster_size", 96))
    max_disp = float(config.get("max_cluster_dispersion", 1e9))
    max_mse_keep = float(config.get("max_reconstruction_mse_for_keep", 1e9))
    rng = np.random.default_rng(int(config["seed"]))
    candidate = len(grouped)
    skipped = 0
    for primitive_id, group in grouped:
        train_group = group[group["split"] == "train"]
        source_group = train_group if len(train_group) >= int(config["min_segments_per_primitive"]) else group
        if len(source_group) < min_cluster:
            skipped += 1
            log.info("Skipping GMR for %s: only %d segments (< min_cluster_size=%d).", primitive_id, len(source_group), min_cluster)
            continue
        rows_list = list(source_group.itertuples(index=False))
        if len(rows_list) > max_per_prim:
            idx = rng.choice(len(rows_list), size=max_per_prim, replace=False)
            rows_list = [rows_list[i] for i in sorted(idx.tolist())]
        trajectories: list[np.ndarray] = []
        segment_rows: list[dict[str, Any]] = []
        for row in rows_list:
            trajectory = load_gmr_trajectory(
                row=row,
                slim_paths=slim_paths,
                segments_dir=segments_dir,
                config=config,
                slim_cache=slim_cache,
                raw_cache=raw_cache,
            )
            if trajectory is not None:
                trajectories.append(trajectory.astype(np.float32))
                segment_rows.append(row._asdict())
        if not trajectories:
            skipped += 1
            continue
        stacked = np.stack(trajectories, axis=0)
        mean_traj = stacked.mean(axis=0)
        dispersion = float(np.mean(np.linalg.norm(stacked - mean_traj[None, :, :], axis=(1, 2))))
        if dispersion > max_disp:
            skipped += 1
            log.info("Skipping GMR for %s: dispersion %.4f > max_cluster_dispersion.", primitive_id, dispersion)
            continue
        mean_conf = float(pd.to_numeric(group.get("assignment_confidence", pd.Series(1.0)), errors="coerce").mean())
        prototypes = fit_primitive_prototypes(
            trajectories=stacked,
            segment_rows=segment_rows,
            gmr_components=int(config["gmr_components"]),
            gmr_reg_covar=float(config["gmr_reg_covar"]),
            seed=int(config["seed"]),
            max_prototypes=max_prototypes,
            min_segments_per_prototype=min_segments_per_prototype,
        )
        predicted_mean = np.asarray(prototypes["default_prior_mean"], dtype=np.float32)
        reconstruction_mse = float(prototypes["mean_reconstruction_mse"])
        if reconstruction_mse > max_mse_keep:
            skipped += 1
            log.info("Skipping GMR for %s: reconstruction_mse %.4f too high.", primitive_id, reconstruction_mse)
            continue
        prior_path = library_dir / f"{primitive_id}_prior.npz"
        save_npz(
            prior_path,
            prior_mean=predicted_mean.astype(np.float32),
            prototype_means=np.asarray(prototypes["prototype_means"], dtype=np.float32),
            prototype_weights=np.asarray(prototypes["prototype_weights"], dtype=np.float32),
            prototype_latent_centroids=np.asarray(prototypes["prototype_latent_centroids"], dtype=np.float32),
            trajectory_examples=stacked[: min(8, len(stacked))],
        )
        gmr_payload["models"][primitive_id] = {
            "prototype_payloads": list(prototypes["prototype_payloads"]),
            "prototype_weights": list(prototypes["prototype_weights"]),
            "prototype_latent_centroids": np.asarray(prototypes["prototype_latent_centroids"], dtype=np.float32),
            "default_prototype_index": int(prototypes["default_prototype_index"]),
        }
        library_rows.append(
            {
                "primitive_id": primitive_id,
                "num_segments": int(len(group)),
                "num_train_segments": int(len(train_group)),
                "num_songs": int(group["song_id"].nunique()),
                "mean_duration_steps": float(group["duration_steps"].mean()),
                "mean_motion_energy": float(group["motion_energy"].mean()),
                "mean_chord_size": float(group["chord_size"].mean()),
                "reconstruction_mse": reconstruction_mse,
                "cluster_dispersion": dispersion,
                "mean_assignment_confidence": mean_conf,
                "num_prototypes": int(len(prototypes["prototype_means"])),
                "default_prototype_index": int(prototypes["default_prototype_index"]),
                "prototype_weight_entropy": float(_weight_entropy(np.asarray(prototypes["prototype_weights"], dtype=np.float32))),
                "prior_path": str(prior_path.resolve()),
            }
        )
    library_df = pd.DataFrame(library_rows).sort_values("primitive_id").reset_index(drop=True)
    gmr_diag = {
        "num_candidate_clusters": int(candidate),
        "num_kept_clusters": int(len(library_df)),
        "num_skipped_clusters": int(skipped),
        "final_primitive_count": int(library_df["primitive_id"].nunique()) if not library_df.empty else 0,
    }
    log.info(
        "GMR summary: candidates=%d kept=%d skipped=%d final_primitive_count=%d",
        candidate,
        gmr_diag["num_kept_clusters"],
        skipped,
        gmr_diag["final_primitive_count"],
    )
    return library_df, gmr_payload, gmr_diag


def fit_primitive_prototypes(
    *,
    trajectories: np.ndarray,
    segment_rows: list[dict[str, Any]],
    gmr_components: int,
    gmr_reg_covar: float,
    seed: int,
    max_prototypes: int,
    min_segments_per_prototype: int,
) -> dict[str, Any]:
    latent_matrix = _latent_matrix_from_rows(segment_rows)
    num_examples = int(trajectories.shape[0])
    num_prototypes = 1
    if max_prototypes > 1 and num_examples >= max(min_segments_per_prototype * 2, 4):
        num_prototypes = min(max_prototypes, max(1, num_examples // min_segments_per_prototype))
    if num_prototypes <= 1:
        gmr = PhaseGMR(
            n_components=int(gmr_components),
            reg_covar=float(gmr_reg_covar),
            random_state=int(seed),
        ).fit(trajectories)
        phases = np.linspace(0.0, 1.0, trajectories.shape[1], dtype=np.float32)
        predicted_mean, _ = gmr.predict(phases)
        reconstruction_mse = float(np.mean((trajectories - predicted_mean[None, :, :]) ** 2))
        return {
            "prototype_means": [predicted_mean.astype(np.float32)],
            "prototype_weights": [1.0],
            "prototype_payloads": [gmr.to_payload()],
            "prototype_latent_centroids": [_mean_or_zeros(latent_matrix)],
            "default_prototype_index": 0,
            "default_prior_mean": predicted_mean.astype(np.float32),
            "mean_reconstruction_mse": reconstruction_mse,
        }

    prototype_model = GaussianMixture(
        n_components=int(num_prototypes),
        covariance_type="full",
        reg_covar=1e-4,
        random_state=int(seed),
    )
    prototype_labels = prototype_model.fit_predict(latent_matrix if latent_matrix.size else trajectories.reshape(num_examples, -1))
    prototype_means: list[np.ndarray] = []
    prototype_weights: list[float] = []
    prototype_payloads: list[dict[str, Any]] = []
    prototype_centroids: list[np.ndarray] = []
    prototype_mse: list[float] = []
    phases = np.linspace(0.0, 1.0, trajectories.shape[1], dtype=np.float32)
    for label in sorted(set(int(item) for item in prototype_labels.tolist())):
        member_mask = prototype_labels == int(label)
        member_trajectories = trajectories[member_mask]
        gmr = PhaseGMR(
            n_components=int(gmr_components),
            reg_covar=float(gmr_reg_covar),
            random_state=int(seed) + int(label),
        ).fit(member_trajectories)
        predicted_mean, _ = gmr.predict(phases)
        prototype_means.append(predicted_mean.astype(np.float32))
        prototype_weights.append(float(member_mask.mean()))
        prototype_payloads.append(gmr.to_payload())
        prototype_centroids.append(_mean_or_zeros(latent_matrix[member_mask] if latent_matrix.size else np.zeros((0, 1), dtype=np.float32)))
        prototype_mse.append(float(np.mean((member_trajectories - predicted_mean[None, :, :]) ** 2)))
    default_prototype_index = int(np.argmax(np.asarray(prototype_weights, dtype=np.float32)))
    return {
        "prototype_means": prototype_means,
        "prototype_weights": prototype_weights,
        "prototype_payloads": prototype_payloads,
        "prototype_latent_centroids": prototype_centroids,
        "default_prototype_index": default_prototype_index,
        "default_prior_mean": prototype_means[default_prototype_index],
        "mean_reconstruction_mse": float(np.mean(prototype_mse)) if prototype_mse else 0.0,
    }


def _latent_matrix_from_rows(segment_rows: list[dict[str, Any]]) -> np.ndarray:
    if not segment_rows:
        return np.zeros((0, 1), dtype=np.float32)
    latent_keys = sorted(key for key in segment_rows[0].keys() if str(key).startswith("latent_"))
    if not latent_keys:
        return np.zeros((len(segment_rows), 1), dtype=np.float32)
    return np.asarray(
        [[float(row.get(key, 0.0)) for key in latent_keys] for row in segment_rows],
        dtype=np.float32,
    )


def _mean_or_zeros(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros((1,), dtype=np.float32)
    return np.asarray(values.mean(axis=0), dtype=np.float32)


def _weight_entropy(weights: np.ndarray) -> float:
    if weights.size == 0:
        return 0.0
    normalized = weights / max(float(weights.sum()), 1e-8)
    return float(-(normalized * np.log(normalized + 1e-8)).sum())


def apply_online_selection_and_cleanup(
    *,
    primitive_root: Path,
    assignments_df: pd.DataFrame,
    library_df: pd.DataFrame,
    config: dict[str, Any],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any] | None]:
    selection_cfg = dict(config.get("online_selection", {}))
    if assignments_df.empty or library_df.empty:
        return assignments_df, library_df, None
    from sonata.evaluation.primitive_online_eval import evaluate_primitives_online

    selection_output_root = primitive_root / "online_selection"
    eval_config = {
        "primitive_root": str(primitive_root),
        "output_root": str(selection_output_root),
        "robopianist_root": selection_cfg.get("robopianist_root", config.get("robopianist_root")),
        "seed": int(config.get("seed", 0)),
        "resume": False,
        "force": True,
        "save_plots": bool(selection_cfg.get("save_plots", True)),
        "save_debug": bool(selection_cfg.get("save_debug", False)),
        "sampling": {
            "instances_per_primitive": int(selection_cfg.get("instances_per_primitive", 1)),
            "max_instances": selection_cfg.get("max_instances"),
            "split": selection_cfg.get("split"),
            "min_chord_size": selection_cfg.get("min_chord_size"),
            "max_chord_size": selection_cfg.get("max_chord_size"),
            "min_duration_steps": selection_cfg.get("min_duration_steps"),
            "max_duration_steps": selection_cfg.get("max_duration_steps"),
        },
        "events": {
            "use_goals": True,
            "use_piano_states": True,
            "goal_key_threshold": 0.5,
            "goal_sustain_threshold": 0.5,
            "piano_state_threshold": 0.5,
            "piano_sustain_threshold": 0.5,
            "onset_tolerance_frames": int(selection_cfg.get("onset_tolerance_frames", 1)),
        },
        "rollout": dict(selection_cfg.get("rollout", {})),
        "aggregation": {
            "top_k_examples": int(selection_cfg.get("top_k_examples", 3)),
            "max_plot_primitives": int(selection_cfg.get("max_plot_primitives", 24)),
        },
    }
    payload = evaluate_primitives_online(config=eval_config, logger=logger)
    if payload.get("status") != "completed":
        return assignments_df, library_df, {"status": str(payload.get("status", "unknown")), "changes_applied": False}
    summary_df = read_table(selection_output_root / "primitive_summary_metrics")
    if summary_df.empty:
        return assignments_df, library_df, {"status": "empty", "changes_applied": False}

    prune_ids = identify_prunable_primitives(summary_df=summary_df, selection_cfg=selection_cfg)
    merge_map = identify_merge_candidates(assignments_df=assignments_df, library_df=library_df, selection_cfg=selection_cfg)
    updated = remap_primitive_assignments(assignments_df=assignments_df, prune_ids=prune_ids, merge_map=merge_map)
    summary = {
        "status": "completed",
        "changes_applied": bool(not updated.equals(assignments_df)),
        "pruned_primitives": sorted(prune_ids),
        "merged_primitives": dict(sorted(merge_map.items())),
        "num_pruned": int(len(prune_ids)),
        "num_merged": int(len(merge_map)),
        "mean_online_onset_f1": float(pd.to_numeric(summary_df["mean_onset_f1"], errors="coerce").mean()),
        "selection_output_root": str(selection_output_root),
    }
    return updated, library_df, summary


def identify_prunable_primitives(*, summary_df: pd.DataFrame, selection_cfg: dict[str, Any]) -> set[str]:
    min_onset_f1 = float(selection_cfg.get("min_onset_f1", 0.12))
    max_false_positive = float(selection_cfg.get("max_false_positive_rate", 12.0))
    max_missed = float(selection_cfg.get("max_missed_note_rate", 24.0))
    prune_ids: set[str] = set()
    for row in summary_df.itertuples(index=False):
        onset_f1 = float(getattr(row, "mean_onset_f1", float("nan")))
        false_positive_rate = float(getattr(row, "false_positive_rate", float("nan")))
        missed_note_rate = float(getattr(row, "missed_note_rate", float("nan")))
        successful = int(getattr(row, "num_successful_rollouts", 0))
        if successful <= 0:
            prune_ids.add(str(row.primitive_id))
            continue
        if not math.isnan(onset_f1) and onset_f1 < min_onset_f1:
            prune_ids.add(str(row.primitive_id))
            continue
        if not math.isnan(false_positive_rate) and false_positive_rate > max_false_positive:
            prune_ids.add(str(row.primitive_id))
            continue
        if not math.isnan(missed_note_rate) and missed_note_rate > max_missed:
            prune_ids.add(str(row.primitive_id))
    return prune_ids


def identify_merge_candidates(
    *,
    assignments_df: pd.DataFrame,
    library_df: pd.DataFrame,
    selection_cfg: dict[str, Any],
) -> dict[str, str]:
    if library_df.empty:
        return {}
    merge_threshold = float(selection_cfg.get("merge_prior_distance_threshold", 0.20))
    family_lookup = (
        assignments_df.groupby("primitive_id")["coarse_family"].agg(lambda values: str(values.mode().iloc[0]) if not values.mode().empty else "other")
        if "coarse_family" in assignments_df.columns
        else pd.Series(dtype=object)
    )
    segment_counts = assignments_df["primitive_id"].astype(str).value_counts().to_dict()
    priors: dict[str, np.ndarray] = {}
    for row in library_df.itertuples(index=False):
        prior_path = Path(str(row.prior_path))
        if prior_path.exists():
            payload = np.load(prior_path, allow_pickle=True)
            priors[str(row.primitive_id)] = np.asarray(payload["prior_mean"], dtype=np.float32)
    merge_map: dict[str, str] = {}
    primitive_ids = sorted(priors)
    for index, primitive_id in enumerate(primitive_ids):
        if primitive_id in merge_map:
            continue
        for candidate_id in primitive_ids[index + 1 :]:
            if candidate_id in merge_map:
                continue
            if str(family_lookup.get(primitive_id, "other")) != str(family_lookup.get(candidate_id, "other")):
                continue
            distance = prior_mean_distance(priors[primitive_id], priors[candidate_id])
            if distance > merge_threshold:
                continue
            keep_id = primitive_id if int(segment_counts.get(primitive_id, 0)) >= int(segment_counts.get(candidate_id, 0)) else candidate_id
            merge_id = candidate_id if keep_id == primitive_id else primitive_id
            merge_map[merge_id] = keep_id
    return merge_map


def prior_mean_distance(prior_a: np.ndarray, prior_b: np.ndarray) -> float:
    a = np.asarray(prior_a, dtype=np.float32).reshape(-1)
    b = np.asarray(prior_b, dtype=np.float32).reshape(-1)
    width = min(len(a), len(b))
    if width <= 0:
        return float("inf")
    return float(np.linalg.norm(a[:width] - b[:width]) / max(width, 1))


def remap_primitive_assignments(
    *,
    assignments_df: pd.DataFrame,
    prune_ids: set[str],
    merge_map: dict[str, str],
) -> pd.DataFrame:
    if assignments_df.empty or (not prune_ids and not merge_map):
        return assignments_df
    frame = assignments_df.copy()
    latent_columns = [column for column in frame.columns if str(column).startswith("latent_")]
    centroids = (
        frame.loc[~frame["primitive_id"].astype(str).isin(prune_ids)]
        .groupby("primitive_id", sort=True)[latent_columns]
        .mean()
        if latent_columns
        else pd.DataFrame()
    )
    family_lookup = (
        frame.groupby("primitive_id")["coarse_family"].agg(lambda values: str(values.mode().iloc[0]) if not values.mode().empty else "other")
        if "coarse_family" in frame.columns
        else pd.Series(dtype=object)
    )
    remap = dict(merge_map)
    for primitive_id in sorted(prune_ids):
        if primitive_id in remap:
            continue
        family_name = str(family_lookup.get(primitive_id, "other"))
        candidates = [
            candidate
            for candidate in centroids.index.astype(str).tolist()
            if candidate != primitive_id and str(family_lookup.get(candidate, "other")) == family_name
        ]
        if not candidates:
            continue
        if latent_columns and primitive_id in frame["primitive_id"].astype(str).values:
            source_rows = frame.loc[frame["primitive_id"].astype(str) == primitive_id, latent_columns]
            if not source_rows.empty:
                source_centroid = source_rows.to_numpy(dtype=np.float32).mean(axis=0)
                distances = [
                    (
                        float(np.linalg.norm(centroids.loc[candidate].to_numpy(dtype=np.float32) - source_centroid)),
                        candidate,
                    )
                    for candidate in candidates
                ]
                distances.sort(key=lambda item: (item[0], item[1]))
                remap[primitive_id] = str(distances[0][1])
                continue
        remap[primitive_id] = str(candidates[0])
    if not remap:
        return frame
    frame["primitive_id"] = frame["primitive_id"].astype(str).map(lambda value: remap.get(value, value))
    primitive_names = sorted(frame["primitive_id"].astype(str).unique().tolist())
    primitive_to_index = {name: idx for idx, name in enumerate(primitive_names)}
    frame["primitive_label"] = frame["primitive_id"].astype(str).map(primitive_to_index).astype(int)
    return frame.sort_values(["primitive_id", "segment_id"], kind="stable").reset_index(drop=True)


def load_gmr_trajectory(
    row,
    slim_paths,
    segments_dir: Path,
    config: dict[str, Any],
    slim_cache: dict[str, Any],
    raw_cache: dict[str, Any],
) -> np.ndarray | None:
    chunk_name = str(row.chunk_path)
    if chunk_name and is_slim_chunk_name(chunk_name):
        target_path = gmr_target_chunk_path(slim_paths, chunk_name)
        if target_path.exists():
            bundle = slim_cache.get(chunk_name)
            if bundle is None:
                bundle = np.load(target_path, allow_pickle=True)
                slim_cache[chunk_name] = bundle
            index = int(row.chunk_index)
            segment_ids = np.asarray(bundle["segment_ids"], dtype=object)
            if index >= len(segment_ids) or str(segment_ids[index]) != str(row.segment_id):
                raise ValueError(f"Slim GMR target segment id mismatch in {chunk_name}")
            return np.asarray(bundle["gmr_targets"][index], dtype=np.float32)

    raw_chunk_name = str(getattr(row, "raw_chunk_path", "") or chunk_name)
    raw_index = int(getattr(row, "raw_chunk_index", -1))
    if raw_index < 0:
        raw_index = int(row.chunk_index)
    raw_path = segments_dir / raw_chunk_name
    if not raw_chunk_name or not raw_path.exists():
        return None
    bundle = raw_cache.get(raw_chunk_name)
    if bundle is None:
        bundle = np.load(raw_path, allow_pickle=True)
        raw_cache[raw_chunk_name] = bundle
    arrays = load_segment_arrays_from_bundle(bundle, raw_index)
    trajectory, _ = build_gmr_target(arrays=arrays, config=config)
    return trajectory.astype(np.float32)


def compute_stage1_metrics(
    assignments_df: pd.DataFrame,
    sweep_df: pd.DataFrame,
    library_df: pd.DataFrame,
    storage_summary: dict[str, Any] | None = None,
    online_selection_summary: dict[str, Any] | None = None,
    discovery_diag: dict[str, Any] | None = None,
) -> dict[str, Any]:
    probabilities = assignments_df["primitive_id"].value_counts(normalize=True).to_numpy(dtype=np.float32)
    usage_entropy = float(-(probabilities * np.log(probabilities + 1e-8)).sum())
    cross_song_reuse = float((library_df["num_songs"] > 1).mean()) if not library_df.empty else 0.0
    selected_rows = sweep_df.loc[sweep_df.get("selected", pd.Series(dtype=bool)).fillna(False)] if not sweep_df.empty else pd.DataFrame()
    metrics = {
        "num_primitives": int(library_df["primitive_id"].nunique()) if not library_df.empty else 0,
        "selected_k": int(assignments_df["primitive_id"].nunique()) if not assignments_df.empty else 0,
        "mean_assignment_confidence": float(assignments_df["assignment_confidence"].mean()),
        "usage_entropy": usage_entropy,
        "primitive_reuse_across_songs": cross_song_reuse,
        "mean_reconstruction_mse": float(library_df["reconstruction_mse"].mean()) if not library_df.empty else 0.0,
        "median_reconstruction_mse": float(library_df["reconstruction_mse"].median()) if not library_df.empty else 0.0,
        "mean_num_prototypes": float(library_df["num_prototypes"].mean()) if "num_prototypes" in library_df.columns and not library_df.empty else 1.0,
        "num_discovery_partitions": int(assignments_df["discovery_partition"].nunique()) if "discovery_partition" in assignments_df.columns else 0,
        "mean_hybrid_selection_score": float(pd.to_numeric(assignments_df.get("hybrid_selection_score", pd.Series(dtype=float)), errors="coerce").mean()) if "hybrid_selection_score" in assignments_df.columns else float("nan"),
        "mean_action_consistency_score": float(pd.to_numeric(selected_rows.get("action_consistency_score", pd.Series(dtype=float)), errors="coerce").mean()) if not selected_rows.empty else float("nan"),
        "mean_event_proxy_score": float(pd.to_numeric(selected_rows.get("event_proxy_score", pd.Series(dtype=float)), errors="coerce").mean()) if not selected_rows.empty else float("nan"),
        "selected_partition_rows": selected_rows.to_dict(orient="records") if not selected_rows.empty else [],
        "gmm_sweep": sweep_df.to_dict(orient="records"),
    }
    if storage_summary:
        metrics.update(
            {
                "segment_store_bytes": int(storage_summary.get("total_bytes_on_disk", 0)),
                "segment_store_bytes_per_1000_segments": float(storage_summary.get("bytes_per_1000_segments", 0.0)),
                "feature_dim": int(storage_summary.get("feature_dim", 0)),
                "gmr_target_steps": int(storage_summary.get("gmr_target_steps", storage_summary.get("gmr_horizon", 0))),
                "gmr_target_dim": int(storage_summary.get("gmr_target_dim", storage_summary.get("gmr_dim", 0))),
                "estimated_storage_reduction_vs_legacy": storage_summary.get("estimated_storage_reduction_vs_legacy"),
            }
        )
    if online_selection_summary:
        metrics["online_selection"] = online_selection_summary
    if discovery_diag is not None:
        for key, value in discovery_diag.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                metrics[key] = value
        if "final_primitive_count" not in metrics:
            metrics["final_primitive_count"] = int(library_df["primitive_id"].nunique()) if not library_df.empty else 0
        metrics.setdefault(
            "pre_prune_primitive_count",
            int(assignments_df.loc[assignments_df["primitive_id"].astype(str) != NOISE_PRIMITIVE_ID, "primitive_id"].nunique())
            if not assignments_df.empty
            else 0,
        )
        noise_rows = int((assignments_df["primitive_id"].astype(str) == NOISE_PRIMITIVE_ID).sum()) if not assignments_df.empty else 0
        metrics.setdefault("noise_rows", noise_rows)
    return metrics
