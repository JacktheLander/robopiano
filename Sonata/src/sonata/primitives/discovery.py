from __future__ import annotations

import logging
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

from sonata.data.indexer import scan_dataset
from sonata.data.loading import load_manifest
from sonata.primitives.features import load_feature_matrix_from_store, resolve_gmr_resample_steps
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


def run_primitive_pipeline(config: dict[str, Any], logger: logging.Logger) -> dict[str, Path]:
    config = dict(config)
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
        segment_outputs = run_segmentation(manifest_df=manifest_df, output_dir=primitive_root, config=config, logger=logger)
        segment_df = read_table(segment_outputs["segment_table_base"])
        feature_matrix, feature_names = load_feature_matrix_from_store(
            segment_df=segment_df,
            output_dir=primitive_root,
            config=config,
            segments_dir=primitive_root / "segments",
        )
        store_summary = (
            read_json(segment_outputs["compact_store_manifest_path"])
            if "compact_store_manifest_path" in segment_outputs and Path(segment_outputs["compact_store_manifest_path"]).exists()
            else summarize_slim_cache(resolve_slim_cache_paths(primitive_root, config))
        )
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

        library_df, gmr_bundle = fit_gmr_library(assignments_df=assignments_df, segments_dir=primitive_root / "segments", output_dir=primitive_root, config=config)
        write_table(library_df, primitive_root / "library" / "primitive_library")
        joblib.dump(gmr_bundle, primitive_root / "library" / "primitive_gmr_bundle.joblib")

        token_df = add_token_columns(
            assignments_df=assignments_df.merge(library_df[["primitive_id", "reconstruction_mse"]], on="primitive_id", how="left"),
            num_duration_buckets=int(config["num_duration_buckets"]),
            num_dynamics_buckets=int(config["num_dynamics_buckets"]),
        )
        token_base = primitive_root / "tokens" / "primitive_tokens"
        write_table(token_df, token_base)
        write_json(build_vocabulary_payload(token_df), primitive_root / "tokens" / "primitive_vocabulary.json")

        metrics = compute_stage1_metrics(assignments_df=assignments_df, sweep_df=sweep_df, library_df=library_df, storage_summary=store_summary)
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

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(feature_matrix[train_mask])
    all_scaled = scaler.transform(feature_matrix)
    pca_components = min(int(config["pca_components"]), train_scaled.shape[0], train_scaled.shape[1])
    pca = PCA(n_components=max(1, pca_components), random_state=int(config["seed"]))
    train_reduced = pca.fit_transform(train_scaled)
    all_reduced = pca.transform(all_scaled)

    sweep_rows: list[dict[str, Any]] = []
    best_model = None
    best_value = float("inf")
    best_k = None
    criterion = str(config["model_selection_metric"])
    for k in config["gmm_k_candidates"]:
        model = GaussianMixture(
            n_components=int(k),
            covariance_type=str(config["gmm_covariance_type"]),
            reg_covar=float(config["gmm_reg_covar"]),
            random_state=int(config["seed"]),
        )
        model.fit(train_reduced)
        bic = float(model.bic(train_reduced))
        aic = float(model.aic(train_reduced))
        value = bic if criterion == "bic" else aic
        sweep_rows.append({"k": int(k), "bic": bic, "aic": aic})
        if value < best_value:
            best_value = value
            best_model = model
            best_k = int(k)

    assert best_model is not None and best_k is not None
    labels = best_model.predict(all_reduced)
    probabilities = best_model.predict_proba(all_reduced)
    confidence = probabilities.max(axis=1)
    primitive_ids = np.asarray([f"primitive_{label:03d}" for label in labels], dtype=object)
    assignments_df = segment_df.copy()
    assignments_df["primitive_label"] = labels
    assignments_df["primitive_id"] = primitive_ids
    assignments_df["assignment_confidence"] = confidence
    assignments_df["embedding_norm"] = np.linalg.norm(all_reduced, axis=1)
    assignments_df["gmm_best_k"] = best_k

    silhouette = float("nan")
    sample_size = min(int(config["silhouette_max_examples"]), len(train_reduced))
    if best_k > 1 and sample_size > best_k:
        sample_indices = np.random.default_rng(int(config["seed"])).choice(len(train_reduced), size=sample_size, replace=False)
        silhouette = float(silhouette_score(train_reduced[sample_indices], best_model.predict(train_reduced[sample_indices])))
    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df["selected_k"] = int(best_k)
    sweep_df["silhouette"] = silhouette
    bundle = {
        "scaler": scaler,
        "pca": pca,
        "gmm": best_model,
        "feature_names": feature_names,
        "selected_k": best_k,
    }
    return assignments_df, sweep_df, bundle


def fit_gmr_library(assignments_df: pd.DataFrame, segments_dir: Path, output_dir: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    library_dir = output_dir / "library"
    library_dir.mkdir(parents=True, exist_ok=True)
    slim_paths = resolve_slim_cache_paths(output_dir, config)
    slim_cache: dict[str, Any] = {}
    raw_cache: dict[str, Any] = {}
    grouped = assignments_df.groupby("primitive_id", sort=True)
    library_rows: list[dict[str, Any]] = []
    gmr_payload: dict[str, Any] = {"models": {}}
    for primitive_id, group in grouped:
        train_group = group[group["split"] == "train"]
        source_group = train_group if len(train_group) >= int(config["min_segments_per_primitive"]) else group
        trajectories: list[np.ndarray] = []
        for row in source_group.itertuples(index=False):
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
        if not trajectories:
            continue
        stacked = np.stack(trajectories, axis=0)
        gmr = PhaseGMR(
            n_components=int(config["gmr_components"]),
            reg_covar=float(config["gmr_reg_covar"]),
            random_state=int(config["seed"]),
        ).fit(stacked)
        phases = np.linspace(0.0, 1.0, stacked.shape[1], dtype=np.float32)
        predicted_mean, _ = gmr.predict(phases)
        reconstruction_mse = float(np.mean((stacked - predicted_mean[None, :, :]) ** 2))
        prior_path = library_dir / f"{primitive_id}_prior.npz"
        save_npz(prior_path, prior_mean=predicted_mean.astype(np.float32), trajectory_examples=stacked[: min(8, len(stacked))])
        gmr_payload["models"][primitive_id] = gmr.to_payload()
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
                "prior_path": str(prior_path.resolve()),
            }
        )
    library_df = pd.DataFrame(library_rows).sort_values("primitive_id").reset_index(drop=True)
    return library_df, gmr_payload


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
) -> dict[str, Any]:
    probabilities = assignments_df["primitive_id"].value_counts(normalize=True).to_numpy(dtype=np.float32)
    usage_entropy = float(-(probabilities * np.log(probabilities + 1e-8)).sum())
    cross_song_reuse = float((library_df["num_songs"] > 1).mean()) if not library_df.empty else 0.0
    metrics = {
        "num_primitives": int(library_df["primitive_id"].nunique()) if not library_df.empty else 0,
        "selected_k": int(sweep_df.loc[sweep_df["selected_k"].notna(), "selected_k"].iloc[0]) if not sweep_df.empty else 0,
        "mean_assignment_confidence": float(assignments_df["assignment_confidence"].mean()),
        "usage_entropy": usage_entropy,
        "primitive_reuse_across_songs": cross_song_reuse,
        "mean_reconstruction_mse": float(library_df["reconstruction_mse"].mean()) if not library_df.empty else 0.0,
        "median_reconstruction_mse": float(library_df["reconstruction_mse"].median()) if not library_df.empty else 0.0,
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
    return metrics
