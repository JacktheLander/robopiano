from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from sonata.data.indexer import scan_dataset
from sonata.data.loading import load_manifest
from sonata.primitives.evaluation import Stage1EarlyStop, Stage1OnlineEvaluator
from sonata.primitives.features import build_condition_feature_vector, extract_segment_features, resolve_gmr_resample_steps, select_feature_indices
from sonata.primitives.gmr import fit_phase_gmr_with_selection
from sonata.primitives.gpu_utils import GpuBackend, fit_pca_embedding, resolve_gpu_backend, screen_kmeans_candidates
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
from sonata.primitives.visualization import (
    plot_cluster_size_hist,
    plot_family_balance,
    plot_gmr_reconstruction,
    plot_primitive_frequency,
    plot_primitive_quality_hist,
    plot_segment_length_hist,
    plot_usage_entropy,
)
from sonata.utils.io import read_json, read_table, save_npz, write_json, write_table
from sonata.utils.wandb import WandbRun

MAX_CONTEXT_FLAT_FINGERTIP_WIDTH = 24


def _read_stage_status(manifest_path: Path) -> str:
    if not manifest_path.exists():
        return "missing"
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return str(payload.get("status", "unknown"))


def run_primitive_pipeline(config: dict[str, Any], logger: logging.Logger) -> dict[str, Path]:
    config = _apply_stage1_defaults(config)
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
    evaluator = Stage1OnlineEvaluator(config=config, output_dir=primitive_root, logger=logger)
    gpu_backend = resolve_gpu_backend(config=config, logger=logger)
    try:
        manifest_base = data_output / str(config["data_manifest_name"])
        if not manifest_base.with_suffix(".csv").exists():
            scan_dataset(config=config["data_config"], logger=logger)
        manifest_df = load_manifest(manifest_base)

        segment_outputs = run_segmentation(
            manifest_df=manifest_df,
            output_dir=primitive_root,
            config=config,
            logger=logger,
            evaluator=evaluator,
        )
        segmentation_status = _read_stage_status(segment_outputs["manifest_path"])
        if segmentation_status != "completed":
            evaluator.finalize(
                {
                    "status": segmentation_status,
                    "phase": "segmentation",
                    "output_root": str(primitive_root),
                }
            )
            raise Stage1EarlyStop(f"Stage 1 segmentation ended with status={segmentation_status}.")

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
                "gpu_backend": gpu_backend.name,
            }
        )

        assignments_df, sweep_df, clustering_bundle = fit_primitive_gmm(
            segment_df=segment_df,
            feature_matrix=feature_matrix,
            feature_names=feature_names,
            config=config,
            gpu_backend=gpu_backend,
            evaluator=evaluator,
        )
        clustering_dir = primitive_root / "clustering"
        clustering_dir.mkdir(parents=True, exist_ok=True)
        assignments_base = clustering_dir / "segment_assignments"
        sweep_base = clustering_dir / "gmm_sweep"
        write_table(assignments_df, assignments_base)
        write_table(sweep_df, sweep_base)
        joblib.dump(clustering_bundle, clustering_dir / "primitive_model_bundle.joblib")

        assignments_df, library_df, gmr_bundle = fit_gmr_library(
            assignments_df=assignments_df,
            segments_dir=primitive_root / "segments",
            output_dir=primitive_root,
            config=config,
            evaluator=evaluator,
            logger=logger,
            feature_matrix=feature_matrix,
            feature_names=feature_names,
        )
        write_table(library_df, primitive_root / "library" / "primitive_library")
        joblib.dump(gmr_bundle, primitive_root / "library" / "primitive_gmr_bundle.joblib")

        token_df = add_token_columns(
            assignments_df=assignments_df.merge(
                library_df[["primitive_id", "reconstruction_mse", "weighted_strike_error", "low_quality_flag"]],
                on="primitive_id",
                how="left",
            ),
            num_duration_buckets=int(config["num_duration_buckets"]),
            num_dynamics_buckets=int(config["num_dynamics_buckets"]),
        )
        token_base = primitive_root / "tokens" / "primitive_tokens"
        write_table(token_df, token_base)
        write_json(build_vocabulary_payload(token_df), primitive_root / "tokens" / "primitive_vocabulary.json")
        write_primitive_diagnostics(assignments_df=assignments_df, library_df=library_df, output_dir=primitive_root)

        metrics = compute_stage1_metrics(
            segment_df=segment_df,
            assignments_df=assignments_df,
            sweep_df=sweep_df,
            library_df=library_df,
            storage_summary=store_summary,
            config=config,
        )
        contract = build_primitive_training_contract(
            primitive_root=primitive_root,
            manifest_df=manifest_df,
            segment_df=segment_df,
            library_df=library_df,
            storage_summary=store_summary,
            config=config,
        )
        contract_path = primitive_root / "validation" / "primitive_training_contract.json"
        contract_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(contract, contract_path)
        if not bool(contract["pass_training_contract"]):
            raise RuntimeError(f"Primitive training contract failed; see {contract_path}")
        metrics_dir = primitive_root / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = metrics_dir / "stage1_metrics.json"
        write_json(metrics, metrics_path)
        scalar_metrics = {key: value for key, value in metrics.items() if not isinstance(value, (list, dict))}
        wandb_run.log(scalar_metrics)
        wandb_run.summary(scalar_metrics | {"status": "completed", "metrics_path": str(metrics_path)})

        plot_dir = primitive_root / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_segment_length_hist(segment_df, plot_dir / "segment_length_hist.png")
        plot_family_balance(segment_df, plot_dir / "family_balance.png")
        plot_primitive_quality_hist(library_df, plot_dir / "primitive_quality_hist.png")
        plot_cluster_size_hist(assignments_df, plot_dir / "cluster_size_hist.png")
        plot_primitive_frequency(assignments_df, plot_dir / "primitive_frequency.png")
        plot_gmr_reconstruction(library_df, plot_dir / "primitive_gmr_reconstruction.png")
        plot_usage_entropy(assignments_df, plot_dir / "primitive_usage_entropy.png")

        summary_path = evaluator.finalize(
            {
                "status": "completed",
                "metrics_path": str(metrics_path),
                "num_segments": int(len(segment_df)),
                "num_primitives": int(library_df["primitive_id"].nunique()) if not library_df.empty else 0,
                "gpu_backend": gpu_backend.summary(),
            }
        )

        wandb_run.log_artifact_bundle(
            artifact_name=f"{run_name}-outputs",
            artifact_type="dataset",
            entries={
                "clustering": clustering_dir,
                "library": primitive_root / "library",
                "tokens": primitive_root / "tokens",
                "metrics": metrics_dir,
                "plots": plot_dir,
                "evaluation": primitive_root / "evaluation",
                "run_config.json": primitive_root / "run_config.json",
            },
            aliases=["latest"],
            metadata={"stage": "primitives", "output_root": str(primitive_root), "evaluation_summary": str(summary_path)},
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
            "evaluation_summary_path": summary_path,
        }
    except Stage1EarlyStop as exc:
        summary_path = evaluator.finalize({"status": "early_stopped", "reason": str(exc), "gpu_backend": gpu_backend.summary()})
        wandb_run.summary({"status": "early_stopped", "reason": str(exc), "evaluation_summary": str(summary_path)})
        raise
    finally:
        wandb_run.finish()


def fit_primitive_gmm(
    *,
    segment_df: pd.DataFrame,
    feature_matrix: np.ndarray,
    feature_names: list[str],
    config: dict[str, Any],
    gpu_backend: GpuBackend,
    evaluator: Stage1OnlineEvaluator | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if segment_df.empty or feature_matrix.size == 0:
        return segment_df.copy(), pd.DataFrame(), {"feature_names": feature_names, "buckets": {}, "gpu_backend": gpu_backend.summary()}

    original_feature_names = list(feature_names)
    feature_matrix, feature_names, clustering_feature_indices = select_clustering_features(
        feature_matrix=feature_matrix,
        feature_names=feature_names,
        config=config,
    )

    train_mask = segment_df["split"].astype(str).to_numpy() == "train"
    if not np.any(train_mask):
        train_mask[:] = True

    family_column = "coarse_family" if bool(config.get("family_aware_clustering", False)) and "coarse_family" in segment_df.columns else None
    if family_column is None:
        bucket_items = [("global", np.arange(len(segment_df), dtype=np.int64))]
    else:
        family_values = segment_df[family_column].fillna("mixed_unknown").astype(str)
        bucket_items = [(family_name, family_values.index[family_values == family_name].to_numpy(dtype=np.int64)) for family_name in sorted(family_values.unique())]

    assignment_frame = segment_df.copy()
    assignment_frame["primitive_label"] = -1
    assignment_frame["primitive_id"] = ""
    assignment_frame["assignment_confidence"] = 0.0
    assignment_frame["assignment_entropy"] = 0.0
    assignment_frame["assignment_margin"] = 0.0
    assignment_frame["embedding_norm"] = 0.0
    assignment_frame["primitive_family_bucket"] = family_column if family_column is not None else "global"
    assignment_frame["gpu_backend"] = gpu_backend.name

    sweep_rows: list[dict[str, Any]] = []
    bundle: dict[str, Any] = {
        "feature_names": feature_names,
        "source_feature_names": original_feature_names,
        "clustering_feature_indices": clustering_feature_indices,
        "clustering_mode": str(config.get("clustering", {}).get("mode", config.get("clustering_mode", "absolute"))),
        "buckets": {},
        "gpu_backend": gpu_backend.summary(),
        "family_aware": bool(family_column),
    }
    global_label_offset = 0

    for bucket_name, indices in bucket_items:
        bucket_features = np.asarray(feature_matrix[indices], dtype=np.float32)
        bucket_train_mask = np.asarray(train_mask[indices], dtype=bool)
        result = _fit_gmm_bucket(
            bucket_name=str(bucket_name),
            feature_matrix=bucket_features,
            train_mask=bucket_train_mask,
            config=config,
            gpu_backend=gpu_backend,
        )
        local_labels = np.asarray(result["labels"], dtype=np.int64)
        probabilities = np.asarray(result["probabilities"], dtype=np.float32)
        confidence = probabilities.max(axis=1)
        entropy = -(probabilities * np.log(probabilities + 1e-8)).sum(axis=1)
        if probabilities.shape[1] > 1:
            sorted_prob = np.sort(probabilities, axis=1)
            margin = sorted_prob[:, -1] - sorted_prob[:, -2]
        else:
            margin = np.ones((probabilities.shape[0],), dtype=np.float32)
        global_labels = local_labels + global_label_offset
        primitive_ids = np.asarray([f"primitive_{label:03d}" for label in global_labels], dtype=object)

        assignment_frame.loc[indices, "primitive_label"] = global_labels
        assignment_frame.loc[indices, "primitive_id"] = primitive_ids
        assignment_frame.loc[indices, "assignment_confidence"] = confidence
        assignment_frame.loc[indices, "assignment_entropy"] = entropy
        assignment_frame.loc[indices, "assignment_margin"] = margin
        assignment_frame.loc[indices, "embedding_norm"] = np.linalg.norm(np.asarray(result["all_reduced"], dtype=np.float32), axis=1)
        assignment_frame.loc[indices, "gmm_best_k"] = int(result["selected_k"])
        assignment_frame.loc[indices, "primitive_family_bucket"] = str(bucket_name)

        for row in result["sweep_rows"]:
            sweep_rows.append(row | {"bucket_name": str(bucket_name), "label_offset": int(global_label_offset)})
        bundle["buckets"][str(bucket_name)] = {
            "scaler": result["scaler"],
            "embedding_model": result["embedding_model"],
            "gmm": result["model"],
            "selected_k": int(result["selected_k"]),
            "selected_covariance_type": str(result["selected_covariance_type"]),
            "silhouette": float(result["silhouette"]),
            "indices": indices.tolist(),
        }
        global_label_offset += int(result["model"].n_components)

    assignment_frame = assignment_frame.sort_index().reset_index(drop=True)
    sweep_df = pd.DataFrame(sweep_rows)
    if evaluator is not None:
        decision = evaluator.observe_clustering(assignment_frame, sweep_df)
        if decision is not None and decision.stop:
            raise Stage1EarlyStop("Stage 1 clustering quality thresholds were exceeded.")
    return assignment_frame, sweep_df, bundle


def _fit_gmm_bucket(
    *,
    bucket_name: str,
    feature_matrix: np.ndarray,
    train_mask: np.ndarray,
    config: dict[str, Any],
    gpu_backend: GpuBackend,
) -> dict[str, Any]:
    train_source = feature_matrix[train_mask] if np.any(train_mask) else feature_matrix
    if train_source.shape[0] == 0:
        raise ValueError(f"Bucket {bucket_name!r} is empty.")
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_source)
    all_scaled = scaler.transform(feature_matrix)
    if train_scaled.shape[0] < 2 or feature_matrix.shape[0] < 2:
        return {
            "labels": np.zeros((feature_matrix.shape[0],), dtype=np.int64),
            "probabilities": np.ones((feature_matrix.shape[0], 1), dtype=np.float32),
            "all_reduced": np.zeros((feature_matrix.shape[0], 1), dtype=np.float32),
            "scaler": scaler,
            "embedding_model": None,
            "model": SimpleNamespace(n_components=1),
            "selected_k": 1,
            "selected_covariance_type": "singleton",
            "silhouette": float("nan"),
            "sweep_rows": [
                {
                    "bucket_name": bucket_name,
                    "stage": "singleton",
                    "k": 1,
                    "covariance_type": "singleton",
                    "bic": float("nan"),
                    "aic": float("nan"),
                    "criterion_value": float("nan"),
                }
            ],
        }
    pca_components = min(int(config["pca_components"]), max(1, train_scaled.shape[0]), train_scaled.shape[1])
    train_reduced, all_reduced, embedding_model = fit_pca_embedding(
        train_scaled=np.asarray(train_scaled, dtype=np.float32),
        all_scaled=np.asarray(all_scaled, dtype=np.float32),
        n_components=max(int(pca_components), 1),
        random_state=int(config["gmm_seed"]),
        backend=gpu_backend,
        subsample_limit=int(config.get("gpu_subsample_limit", 32768)),
    )

    covariance_types = _normalize_covariance_types(config.get("gmm_candidate_covariance_types", [config.get("gmm_covariance_type", "full")]))
    k_candidates = [
        max(int(k), 1)
        for k in _resolve_gmm_k_candidates(config=config, bucket_name=bucket_name)
        if int(k) < max(train_reduced.shape[0], 2)
    ]
    if not k_candidates:
        k_candidates = [1]

    full_candidates = [(int(k), covariance_type) for covariance_type in covariance_types for k in k_candidates]
    sweep_rows: list[dict[str, Any]] = []
    if bool(config.get("gmm_use_staged_k_search", True)) and len(full_candidates) > 1 and train_reduced.shape[0] > 8:
        subset = _subsample_rows(
            train_reduced,
            size=min(int(config.get("gmm_k_screen_subset_size", 8192)), train_reduced.shape[0]),
            seed=int(config["gmm_seed"]),
        )
        kmeans_rows = {row["k"]: row for row in screen_kmeans_candidates(
            reduced_matrix=subset,
            k_candidates=k_candidates,
            random_state=int(config["gmm_seed"]),
            backend=gpu_backend,
            subsample_limit=int(config.get("gpu_subsample_limit", 32768)),
        )}
        screen_candidates: list[tuple[float, tuple[int, str]]] = []
        for k, covariance_type in full_candidates:
            if subset.shape[0] <= k:
                continue
            model = GaussianMixture(
                n_components=int(k),
                covariance_type=str(covariance_type),
                reg_covar=_adaptive_gmm_reg(train_reduced, float(config["gmm_reg_covar"])),
                random_state=int(config["gmm_seed"]),
                n_init=1,
                max_iter=int(config.get("gmm_screen_max_iter", 32)),
            )
            model.fit(subset)
            bic = float(model.bic(subset))
            aic = float(model.aic(subset))
            value = bic if str(config["model_selection_metric"]).lower() == "bic" else aic
            sweep_rows.append(
                {
                    "bucket_name": bucket_name,
                    "stage": "screen",
                    "k": int(k),
                    "covariance_type": str(covariance_type),
                    "bic": bic,
                    "aic": aic,
                    "criterion_value": value,
                    "screen_inertia": kmeans_rows.get(int(k), {}).get("screen_inertia"),
                }
            )
            screen_candidates.append((value, (int(k), str(covariance_type))))
        top_full = max(int(config.get("gmm_top_k_full_fits", 2)), 1)
        selected_pairs = [candidate for _, candidate in sorted(screen_candidates, key=lambda item: item[0])[:top_full]]
        if not selected_pairs:
            selected_pairs = full_candidates
    else:
        selected_pairs = full_candidates

    best_model = None
    best_value = float("inf")
    best_k = 1
    best_covariance = "full"
    criterion = str(config["model_selection_metric"]).lower()
    for k, covariance_type in selected_pairs:
        if train_reduced.shape[0] <= k:
            continue
        model = GaussianMixture(
            n_components=int(k),
            covariance_type=str(covariance_type),
            reg_covar=_adaptive_gmm_reg(train_reduced, float(config["gmm_reg_covar"])),
            random_state=int(config["gmm_seed"]),
            n_init=max(int(config.get("gmm_n_init", 4)), 1),
        )
        model.fit(train_reduced)
        bic = float(model.bic(train_reduced))
        aic = float(model.aic(train_reduced))
        value = bic if criterion == "bic" else aic
        sweep_rows.append(
            {
                "bucket_name": bucket_name,
                "stage": "full",
                "k": int(k),
                "covariance_type": str(covariance_type),
                "bic": bic,
                "aic": aic,
                "criterion_value": value,
            }
        )
        if value < best_value:
            best_value = value
            best_model = model
            best_k = int(k)
            best_covariance = str(covariance_type)

    if best_model is None:
        best_model = GaussianMixture(
            n_components=1,
            covariance_type="full",
            reg_covar=_adaptive_gmm_reg(train_reduced, float(config["gmm_reg_covar"])),
            random_state=int(config["gmm_seed"]),
            n_init=max(int(config.get("gmm_n_init", 4)), 1),
        ).fit(train_reduced)
        best_k = 1
        best_covariance = "full"

    probabilities = np.asarray(best_model.predict_proba(all_reduced), dtype=np.float32)
    labels = np.asarray(best_model.predict(all_reduced), dtype=np.int64)
    silhouette = float("nan")
    sample_size = min(int(config["silhouette_max_examples"]), len(train_reduced))
    if best_k > 1 and sample_size > best_k:
        sample_indices = np.random.default_rng(int(config["gmm_seed"])).choice(len(train_reduced), size=sample_size, replace=False)
        silhouette = float(silhouette_score(train_reduced[sample_indices], best_model.predict(train_reduced[sample_indices])))
    return {
        "labels": labels,
        "probabilities": probabilities,
        "all_reduced": all_reduced,
        "scaler": scaler,
        "embedding_model": embedding_model,
        "model": best_model,
        "selected_k": best_k,
        "selected_covariance_type": best_covariance,
        "silhouette": silhouette,
        "sweep_rows": sweep_rows,
    }


def fit_gmr_library(
    *,
    assignments_df: pd.DataFrame,
    segments_dir: Path,
    output_dir: Path,
    config: dict[str, Any],
    evaluator: Stage1OnlineEvaluator | None = None,
    logger: logging.Logger | None = None,
    feature_matrix: np.ndarray | None = None,
    feature_names: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    refined_assignments = assignments_df.copy()
    library_df, gmr_bundle = _fit_library_models(
        assignments_df=refined_assignments,
        segments_dir=segments_dir,
        output_dir=output_dir,
        config=config,
        evaluator=evaluator,
        feature_matrix=feature_matrix,
        feature_names=feature_names,
    )

    if bool(config.get("primitive_split_merge_refinement", False)) and not library_df.empty:
        refined_assignments = _split_bad_primitives(
            assignments_df=refined_assignments,
            library_df=library_df,
            segments_dir=segments_dir,
            output_dir=output_dir,
            config=config,
        )
        if not refined_assignments["primitive_id"].equals(assignments_df["primitive_id"]):
            if logger is not None:
                logger.info("Stage 1 split refinement updated primitive assignments; refitting GMR library.")
            library_df, gmr_bundle = _fit_library_models(
                assignments_df=refined_assignments,
                segments_dir=segments_dir,
                output_dir=output_dir,
                config=config,
                evaluator=evaluator,
                feature_matrix=feature_matrix,
                feature_names=feature_names,
            )

        merged_assignments = _merge_duplicate_primitives(
            assignments_df=refined_assignments,
            library_df=library_df,
            gmr_bundle=gmr_bundle,
            config=config,
        )
        if not merged_assignments["primitive_id"].equals(refined_assignments["primitive_id"]):
            if logger is not None:
                logger.info("Stage 1 merge refinement updated primitive assignments; refitting GMR library.")
            refined_assignments = merged_assignments
            library_df, gmr_bundle = _fit_library_models(
                assignments_df=refined_assignments,
                segments_dir=segments_dir,
                output_dir=output_dir,
                config=config,
                evaluator=evaluator,
                feature_matrix=feature_matrix,
                feature_names=feature_names,
            )

    if evaluator is not None:
        decision = evaluator.observe_gmr_library(library_df)
        if decision is not None and decision.stop:
            raise Stage1EarlyStop("Stage 1 GMR quality thresholds were exceeded.")
    return refined_assignments, library_df, gmr_bundle


def _fit_library_models(
    *,
    assignments_df: pd.DataFrame,
    segments_dir: Path,
    output_dir: Path,
    config: dict[str, Any],
    evaluator: Stage1OnlineEvaluator | None = None,
    feature_matrix: np.ndarray | None = None,
    feature_names: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    library_dir = output_dir / "library"
    library_dir.mkdir(parents=True, exist_ok=True)
    slim_paths = resolve_slim_cache_paths(output_dir, config)
    slim_cache: dict[str, Any] = {}
    raw_cache: dict[str, Any] = {}
    grouped = assignments_df.groupby("primitive_id", sort=True)
    library_rows: list[dict[str, Any]] = []
    gmr_payload: dict[str, Any] = {"models": {}, "diagnostics": {}, "priors": {}}

    for primitive_id, group in grouped:
        dominant_family = _dominant_label(group.get("coarse_family", group.get("heuristic_family", pd.Series(dtype=object))))
        train_group = group[group["split"] == "train"] if "split" in group.columns else group
        source_group = train_group if len(train_group) >= int(config["min_segments_per_primitive"]) else group
        trajectories: list[np.ndarray] = []
        usable_indices: list[int] = []
        condition_vectors: list[np.ndarray] = []
        condition_feature_names: list[str] = []
        condition_feature_indices: list[int] = []
        for row_index, row in source_group.iterrows():
            trajectory = load_gmr_trajectory(
                row=row,
                slim_paths=slim_paths,
                segments_dir=segments_dir,
                config=config,
                slim_cache=slim_cache,
                raw_cache=raw_cache,
            )
            if trajectory is not None:
                trajectories.append(np.asarray(trajectory, dtype=np.float32))
                usable_indices.append(int(row_index))
                if feature_matrix is not None and feature_names is not None and int(row_index) < int(feature_matrix.shape[0]):
                    condition_vector, names, indices = build_condition_feature_vector(
                        feature_vector=np.asarray(feature_matrix[int(row_index)], dtype=np.float32),
                        feature_names=list(feature_names),
                        config=config,
                    )
                    if condition_vector.size:
                        condition_vectors.append(condition_vector.astype(np.float32))
                        condition_feature_names = condition_feature_names or names
                        condition_feature_indices = condition_feature_indices or indices
        if not trajectories:
            continue

        stacked = np.stack(trajectories, axis=0).astype(np.float32)
        fit_failed = False
        try:
            gmr, diagnostics = fit_phase_gmr_with_selection(
                trajectories=stacked,
                component_candidates=list(config["gmr_component_candidates"]) if bool(config.get("adaptive_gmr_components", True)) else [int(config["gmr_components"])],
                reg_covar=float(config["gmr_reg_covar"]),
                random_state=int(config["gmm_seed"]),
                min_samples_per_component=int(config.get("gmr_min_samples_per_component", 4)),
                strike_weight=float(config.get("gmr_strike_weight", 2.5)),
                covariance_type="full",
                n_init=max(int(config.get("gmm_n_init", 4)), 1),
            )
            phases = np.linspace(0.0, 1.0, stacked.shape[1], dtype=np.float32)
            predicted_mean, _ = gmr.predict(phases)
            selected_metrics = diagnostics["selected"]
        except Exception:
            fit_failed = True
            predicted_mean = np.mean(stacked, axis=0).astype(np.float32)
            selected_metrics = {
                "reconstruction_mse": float(np.mean((stacked - predicted_mean[None, :, :]) ** 2)),
                "reconstruction_l1": float(np.mean(np.abs(stacked - predicted_mean[None, :, :]))),
                "weighted_strike_error": float(np.mean((stacked - predicted_mean[None, :, :]) ** 2)),
                "component_count": 1,
                "reg_covar": float(config["gmr_reg_covar"]),
            }
            gmr = None
            diagnostics = {"selected": selected_metrics, "candidates": []}

        prior_path = library_dir / f"{primitive_id}_prior.npz"
        save_npz(
            prior_path,
            prior_mean=predicted_mean.astype(np.float32),
            trajectory_examples=stacked[: min(8, len(stacked))],
            **_conditional_prior_npz_payload(
                condition_vectors=condition_vectors,
                condition_feature_names=condition_feature_names,
                condition_feature_indices=condition_feature_indices,
                trajectories=stacked,
                predicted_mean=predicted_mean,
                config=config,
            ),
            gmr_target_name=np.asarray(["actions" if bool(config.get("gmr_target_actions", True)) else "hand_joints"], dtype=object),
            causal_segment=np.asarray([bool(config.get("segmentation_strategy") == "prepress_causal" or config.get("segmenter_name") == "prepress_causal")]),
            segment_alignment=np.asarray(["prepress_to_onset" if str(config.get("segmentation_strategy", "")) == "prepress_causal" or str(config.get("segmenter_name", "")) == "prepress_causal" else ""], dtype=object),
        )
        if gmr is not None:
            gmr_payload["models"][primitive_id] = gmr.to_payload()
        gmr_payload["diagnostics"][primitive_id] = diagnostics
        gmr_payload["priors"][primitive_id] = predicted_mean.astype(np.float32)
        conditional_summary = _conditional_prior_summary(
            condition_vectors=condition_vectors,
            condition_feature_names=condition_feature_names,
            condition_feature_indices=condition_feature_indices,
            trajectories=stacked,
            predicted_mean=predicted_mean,
            config=config,
        )

        confidence_values = group["assignment_confidence"].astype(float).to_numpy(dtype=np.float32) if "assignment_confidence" in group.columns else np.ones((len(group),), dtype=np.float32)
        low_quality = bool(
            fit_failed
            or float(selected_metrics["weighted_strike_error"]) > float(config.get("stage1_max_weighted_recon_error", 5.0))
            or float(confidence_values.mean()) < float(config.get("stage1_assignment_confidence_threshold", 0.55))
        )
        row_payload = {
            "primitive_id": primitive_id,
            "dominant_family": dominant_family,
            "num_segments": int(len(group)),
            "num_train_segments": int(len(train_group)),
            "num_songs": int(group["song_id"].nunique()) if "song_id" in group.columns else 0,
            "mean_duration_steps": float(group["duration_steps"].mean()) if "duration_steps" in group.columns else 0.0,
            "duration_variance": float(group["duration_steps"].var(ddof=0)) if "duration_steps" in group.columns else 0.0,
            "mean_motion_energy": float(group["motion_energy"].mean()) if "motion_energy" in group.columns else 0.0,
            "mean_chord_size": float(group["chord_size"].mean()) if "chord_size" in group.columns else 0.0,
            "assignment_confidence_mean": float(confidence_values.mean()),
            "assignment_confidence_std": float(confidence_values.std()),
            "sample_count": int(stacked.shape[0]),
            "component_count": int(selected_metrics.get("component_count", 1)),
            "reconstruction_mse": float(selected_metrics["reconstruction_mse"]),
            "reconstruction_l1": float(selected_metrics["reconstruction_l1"]),
            "weighted_strike_error": float(selected_metrics["weighted_strike_error"]),
            "low_quality_flag": bool(low_quality),
            "fit_failed": bool(fit_failed),
            "prior_path": str(prior_path.resolve()),
            "gmr_target_name": "actions" if bool(config.get("gmr_target_actions", True)) else "hand_joints",
            "target_name": "actions" if bool(config.get("gmr_target_actions", True)) else "hand_joints",
            "target_action_dim": int(stacked.shape[-1]),
            "primitive_family": dominant_family,
            "hand": _dominant_label(group.get("hand_side", pd.Series(dtype=object))) if "hand_side" in group.columns else "unknown",
            "finger_set": _dominant_label(group.get("finger_set", group.get("finger_set_id", pd.Series(dtype=object)))) if ("finger_set" in group.columns or "finger_set_id" in group.columns) else "unknown",
            "finger_set_id": _dominant_label(group.get("finger_set_id", pd.Series(dtype=object))) if "finger_set_id" in group.columns else "unknown",
            "chord_size": int(round(float(group["chord_size"].median()))) if "chord_size" in group.columns and not group.empty else 0,
            "interval_pattern_bucket": _dominant_label(group.get("interval_pattern_bucket", pd.Series(dtype=object))) if "interval_pattern_bucket" in group.columns else "none",
            "condition_feature_names": json.dumps(condition_feature_names),
            "condition_feature_count": int(len(condition_feature_names)),
            "frame_mode": _primitive_frame_mode(config),
            "primitive_frame_mode": _primitive_frame_mode(config),
            "supports_conditioned_rollout": bool(conditional_summary["supports_conditioned_rollout"]),
            "fallback_unconditional_available": True,
            "unconditional_fallback_available": True,
            "conditioned_reconstruction_mse": float(conditional_summary["conditioned_reconstruction_mse"]),
            "conditioned_vs_unconditioned_delta": float(conditional_summary["conditioned_vs_unconditioned_delta"]),
            "causal_segment": bool(config.get("segmentation_strategy") == "prepress_causal" or config.get("segmenter_name") == "prepress_causal"),
            "segment_alignment": "prepress_to_onset" if str(config.get("segmentation_strategy", "")) == "prepress_causal" or str(config.get("segmenter_name", "")) == "prepress_causal" else "",
        }
        library_rows.append(row_payload)
        if evaluator is not None:
            decision = evaluator.observe_gmr_primitive(row_payload)
            if decision is not None and decision.stop:
                raise Stage1EarlyStop("Stage 1 GMR quality thresholds were exceeded.")

    library_df = pd.DataFrame(library_rows).sort_values("primitive_id").reset_index(drop=True) if library_rows else pd.DataFrame()
    if not library_df.empty:
        library_df["duplicate_neighbor_distance"] = _nearest_neighbor_distances(library_df["primitive_id"].tolist(), gmr_payload["priors"])
    return library_df, gmr_payload


def _split_bad_primitives(
    *,
    assignments_df: pd.DataFrame,
    library_df: pd.DataFrame,
    segments_dir: Path,
    output_dir: Path,
    config: dict[str, Any],
) -> pd.DataFrame:
    refined = assignments_df.copy()
    slim_paths = resolve_slim_cache_paths(output_dir, config)
    slim_cache: dict[str, Any] = {}
    raw_cache: dict[str, Any] = {}
    improvement_ratio = float(config.get("primitive_split_error_improvement", 0.9))
    min_segments = max(int(config.get("min_segments_per_primitive", 12)), 2)

    for primitive_id, row in library_df.set_index("primitive_id", drop=False).iterrows():
        if not bool(row.get("low_quality_flag", False)) or int(row.get("sample_count", 0)) < (2 * min_segments):
            continue
        primitive_rows = refined[refined["primitive_id"] == primitive_id]
        trajectories: list[np.ndarray] = []
        usable_indices: list[int] = []
        for row_index, assignment_row in primitive_rows.iterrows():
            trajectory = load_gmr_trajectory(
                row=assignment_row,
                slim_paths=slim_paths,
                segments_dir=segments_dir,
                config=config,
                slim_cache=slim_cache,
                raw_cache=raw_cache,
            )
            if trajectory is not None:
                trajectories.append(np.asarray(trajectory, dtype=np.float32))
                usable_indices.append(int(row_index))
        if len(trajectories) < 2 * min_segments:
            continue
        stacked = np.stack(trajectories, axis=0).reshape(len(trajectories), -1)
        splitter = GaussianMixture(
            n_components=2,
            covariance_type="full",
            reg_covar=float(config["gmr_reg_covar"]),
            random_state=int(config["gmm_seed"]),
            n_init=max(int(config.get("gmm_n_init", 4)), 1),
        )
        splitter.fit(stacked)
        labels = splitter.predict(stacked)
        counts = np.bincount(labels, minlength=2)
        if np.any(counts < min_segments):
            continue
        child_errors = []
        for label in range(2):
            child_traj = np.stack([trajectories[index] for index, value in enumerate(labels) if int(value) == label], axis=0)
            _, diagnostics = fit_phase_gmr_with_selection(
                trajectories=child_traj,
                component_candidates=[int(config.get("gmr_components", 4)), max(int(config.get("gmr_components", 4)) // 2, 1)],
                reg_covar=float(config["gmr_reg_covar"]),
                random_state=int(config["gmm_seed"]) + label,
                min_samples_per_component=int(config.get("gmr_min_samples_per_component", 4)),
                strike_weight=float(config.get("gmr_strike_weight", 2.5)),
            )
            child_errors.append(float(diagnostics["selected"]["weighted_strike_error"]))
        if (sum(child_errors) / len(child_errors)) >= float(row["weighted_strike_error"]) * improvement_ratio:
            continue
        for assignment_index, label in zip(usable_indices, labels.tolist()):
            refined.loc[assignment_index, "primitive_id"] = f"{primitive_id}_split_{label}"
    return _renumber_primitives(refined)


def _merge_duplicate_primitives(
    *,
    assignments_df: pd.DataFrame,
    library_df: pd.DataFrame,
    gmr_bundle: dict[str, Any],
    config: dict[str, Any],
) -> pd.DataFrame:
    if library_df.empty:
        return assignments_df
    priors = gmr_bundle.get("priors", {})
    threshold = float(config.get("primitive_merge_distance_threshold", 0.02))
    dominant_family = library_df.set_index("primitive_id")["dominant_family"].to_dict() if "dominant_family" in library_df.columns else {}
    sizes = assignments_df["primitive_id"].value_counts().to_dict()
    merge_map: dict[str, str] = {}
    primitive_ids = list(library_df["primitive_id"].astype(str))
    for index, left in enumerate(primitive_ids):
        for right in primitive_ids[index + 1 :]:
            if dominant_family.get(left) != dominant_family.get(right):
                continue
            left_prior = np.asarray(priors.get(left), dtype=np.float32)
            right_prior = np.asarray(priors.get(right), dtype=np.float32)
            if left_prior.size == 0 or right_prior.size == 0:
                continue
            distance = float(np.mean((left_prior - right_prior) ** 2))
            if distance >= threshold:
                continue
            target, source = (left, right) if int(sizes.get(left, 0)) >= int(sizes.get(right, 0)) else (right, left)
            merge_map[source] = target
    if not merge_map:
        return assignments_df
    refined = assignments_df.copy()
    refined["primitive_id"] = refined["primitive_id"].astype(str).map(lambda value: merge_map.get(value, value))
    return _renumber_primitives(refined)


def select_clustering_features(
    *,
    feature_matrix: np.ndarray,
    feature_names: list[str],
    config: dict[str, Any],
) -> tuple[np.ndarray, list[str], list[int]]:
    clustering_config = config.get("clustering", {}) if isinstance(config.get("clustering"), dict) else {}
    mode = str(clustering_config.get("mode", config.get("clustering_mode", "")))
    if mode != "reusable_motor_motif":
        return np.asarray(feature_matrix, dtype=np.float32), list(feature_names), list(range(len(feature_names)))

    selectors: list[str] = []
    if bool(clustering_config.get("include_relative_geometry", True)):
        selectors.extend(["relative_geometry"])
    if bool(clustering_config.get("include_finger_set", True)):
        selectors.extend(["finger_set"])
    if bool(clustering_config.get("include_chord_interval_pattern", True)):
        selectors.extend(["chord_interval_pattern"])
    if bool(clustering_config.get("include_motion_family", True)):
        selectors.extend(["motion_family"])
    if bool(clustering_config.get("include_wrist_relative_features", True)):
        selectors.extend(["wrist_relative_features"])
    selectors.extend(["context_duration_dynamics", "joint_rel_", "traj_joint_rel", "traj_joint_velocity_rel", "action_", "traj_action_delta"])
    indices = select_feature_indices(feature_names=feature_names, selectors=selectors)
    if bool(clustering_config.get("exclude_absolute_key_position_from_clustering", True)):
        excluded_prefixes = (
            "target_keyset_",
            "contact_summary_",
            "piano_state_",
            "piano_states_",
            "raw_piano_state_",
            "context_wrist_start_",
            "score_scalar_0005",
            "context_chord_geometry_0001",
        )
        indices = [index for index in indices if not any(feature_names[index].startswith(prefix) for prefix in excluded_prefixes)]
    if not indices:
        return np.asarray(feature_matrix, dtype=np.float32), list(feature_names), list(range(len(feature_names)))
    selected = np.asarray(feature_matrix[:, np.asarray(indices, dtype=np.int64)], dtype=np.float32)
    return selected, [feature_names[index] for index in indices], indices


def _conditional_prior_npz_payload(
    *,
    condition_vectors: list[np.ndarray],
    condition_feature_names: list[str],
    condition_feature_indices: list[int],
    trajectories: np.ndarray,
    predicted_mean: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    model = _fit_conditioned_regressor(
        condition_vectors=condition_vectors,
        trajectories=trajectories,
        predicted_mean=predicted_mean,
        config=config,
    )
    if model is None:
        return {
            "condition_feature_names": np.asarray(condition_feature_names, dtype=object),
            "condition_feature_indices": np.asarray(condition_feature_indices, dtype=np.int64),
            "supports_conditioned_rollout": np.asarray([False]),
            "fallback_unconditional_available": np.asarray([True]),
        }
    return {
        "condition_feature_names": np.asarray(condition_feature_names, dtype=object),
        "condition_feature_indices": np.asarray(condition_feature_indices, dtype=np.int64),
        "condition_mean": np.asarray(model["condition_mean"], dtype=np.float32),
        "condition_std": np.asarray(model["condition_std"], dtype=np.float32),
        "condition_regression_coef": np.asarray(model["coef"], dtype=np.float32),
        "condition_regression_intercept": np.asarray(model["intercept"], dtype=np.float32),
        "conditioned_train_mse": np.asarray([float(model["train_mse"])], dtype=np.float32),
        "unconditional_train_mse": np.asarray([float(model["unconditional_mse"])], dtype=np.float32),
        "supports_conditioned_rollout": np.asarray([True]),
        "fallback_unconditional_available": np.asarray([True]),
    }


def _conditional_prior_summary(
    *,
    condition_vectors: list[np.ndarray],
    condition_feature_names: list[str],
    condition_feature_indices: list[int],
    trajectories: np.ndarray,
    predicted_mean: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    model = _fit_conditioned_regressor(
        condition_vectors=condition_vectors,
        trajectories=trajectories,
        predicted_mean=predicted_mean,
        config=config,
    )
    if model is None:
        return {
            "supports_conditioned_rollout": False,
            "condition_feature_names": condition_feature_names,
            "condition_feature_indices": condition_feature_indices,
            "conditioned_reconstruction_mse": float("nan"),
            "conditioned_vs_unconditioned_delta": float("nan"),
        }
    return {
        "supports_conditioned_rollout": True,
        "condition_feature_names": condition_feature_names,
        "condition_feature_indices": condition_feature_indices,
        "conditioned_reconstruction_mse": float(model["train_mse"]),
        "conditioned_vs_unconditioned_delta": float(model["unconditional_mse"] - model["train_mse"]),
    }


def _fit_conditioned_regressor(
    *,
    condition_vectors: list[np.ndarray],
    trajectories: np.ndarray,
    predicted_mean: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    if not bool(config.get("condition_gmr_on_context", False)):
        return None
    if len(condition_vectors) != int(trajectories.shape[0]) or len(condition_vectors) < 2:
        return None
    condition_matrix = np.stack(condition_vectors, axis=0).astype(np.float32)
    if condition_matrix.ndim != 2 or condition_matrix.shape[1] == 0:
        return None
    target = trajectories.reshape(trajectories.shape[0], -1).astype(np.float32)
    condition_mean = condition_matrix.mean(axis=0)
    condition_std = np.clip(condition_matrix.std(axis=0), 1e-6, None)
    normalized = (condition_matrix - condition_mean[None, :]) / condition_std[None, :]
    alpha = float(config.get("condition_regression_alpha", 1.0))
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(normalized, target)
    predicted = model.predict(normalized).astype(np.float32)
    unconditional = predicted_mean.reshape(1, -1).astype(np.float32)
    return {
        "condition_mean": condition_mean,
        "condition_std": condition_std,
        "coef": np.asarray(model.coef_, dtype=np.float32),
        "intercept": np.asarray(model.intercept_, dtype=np.float32),
        "train_mse": float(np.mean((target - predicted) ** 2)),
        "unconditional_mse": float(np.mean((target - unconditional) ** 2)),
    }


def _primitive_frame_mode(config: dict[str, Any]) -> str:
    frame_config = config.get("primitive_frame", {}) if isinstance(config.get("primitive_frame"), dict) else {}
    return str(frame_config.get("mode", config.get("primitive_frame_mode", "absolute")))


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
    *,
    segment_df: pd.DataFrame,
    assignments_df: pd.DataFrame,
    sweep_df: pd.DataFrame,
    library_df: pd.DataFrame,
    storage_summary: dict[str, Any] | None = None,
    config: dict[str, Any],
) -> dict[str, Any]:
    probabilities = assignments_df["primitive_id"].value_counts(normalize=True).to_numpy(dtype=np.float32) if not assignments_df.empty else np.zeros((0,), dtype=np.float32)
    usage_entropy = float(-(probabilities * np.log(probabilities + 1e-8)).sum()) if probabilities.size else 0.0
    cross_song_reuse = float((library_df["num_songs"] > 1).mean()) if not library_df.empty and "num_songs" in library_df.columns else 0.0
    family_column = "coarse_family" if "coarse_family" in segment_df.columns else "heuristic_family"
    family_entropy = _entropy_from_series(segment_df[family_column]) if family_column in segment_df.columns and not segment_df.empty else 0.0
    score_onsets = int(segment_df[["episode_id", "target_onset_step"]].drop_duplicates().shape[0]) if {"episode_id", "target_onset_step"}.issubset(segment_df.columns) and not segment_df.empty else int(len(segment_df))
    duration_values = pd.to_numeric(segment_df.get("duration_steps", pd.Series(dtype=float)), errors="coerce")
    target_key_values = pd.to_numeric(segment_df.get("target_key_count", segment_df.get("chord_size", pd.Series(dtype=float))), errors="coerce")
    metrics = {
        "num_segments": int(len(segment_df)),
        "num_primitives": int(library_df["primitive_id"].nunique()) if not library_df.empty else 0,
        "selected_k": int(sweep_df.loc[sweep_df["stage"] == "full", "k"].iloc[0]) if not sweep_df.empty and np.any(sweep_df["stage"] == "full") else 0,
        "mean_assignment_confidence": float(assignments_df["assignment_confidence"].mean()) if "assignment_confidence" in assignments_df.columns and not assignments_df.empty else 0.0,
        "low_confidence_frac": float((assignments_df["assignment_confidence"] < float(config.get("stage1_assignment_confidence_threshold", 0.55))).mean()) if "assignment_confidence" in assignments_df.columns and not assignments_df.empty else 0.0,
        "usage_entropy": usage_entropy,
        "primitive_reuse_across_songs": cross_song_reuse,
        "mean_reconstruction_mse": float(library_df["reconstruction_mse"].mean()) if not library_df.empty else 0.0,
        "median_reconstruction_mse": float(library_df["reconstruction_mse"].median()) if not library_df.empty else 0.0,
        "mean_reconstruction_l1": float(library_df["reconstruction_l1"].mean()) if "reconstruction_l1" in library_df.columns and not library_df.empty else 0.0,
        "mean_weighted_strike_error": float(library_df["weighted_strike_error"].mean()) if "weighted_strike_error" in library_df.columns and not library_df.empty else 0.0,
        "low_quality_primitive_frac": float(library_df["low_quality_flag"].astype(bool).mean()) if "low_quality_flag" in library_df.columns and not library_df.empty else 0.0,
        "compression_ratio": float(len(assignments_df) / max(library_df["primitive_id"].nunique(), 1)) if not assignments_df.empty else 0.0,
        "family_entropy": family_entropy,
        "duplicate_primitive_frac": float((library_df["duplicate_neighbor_distance"] < float(config.get("stage1_duplicate_primitive_distance", 0.02))).mean()) if "duplicate_neighbor_distance" in library_df.columns and not library_df.empty else 0.0,
        "metadata_predictability_proxy": _metadata_predictability(assignments_df),
        "segment_duplicate_rate": float((segment_df["duplicate_iou"] >= float(config.get("segment_duplicate_iou_threshold", 0.85))).mean()) if "duplicate_iou" in segment_df.columns and not segment_df.empty else 0.0,
        "segments_per_score_onset": float(len(segment_df) / max(score_onsets, 1)) if not segment_df.empty else 0.0,
        "mean_segment_duration_steps": float(duration_values.mean()) if not duration_values.empty else 0.0,
        "p95_segment_duration_steps": float(duration_values.quantile(0.95)) if not duration_values.empty else 0.0,
        "mean_target_key_count": float(target_key_values.mean()) if not target_key_values.empty else 0.0,
        "target_family_counts": segment_df[family_column].astype(str).value_counts().to_dict() if family_column in segment_df.columns and not segment_df.empty else {},
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
                "budget_enabled": bool(storage_summary.get("budget_enabled", False)),
                "proposed_segments": int(storage_summary.get("proposed_segments", 0)),
                "accepted_segments_before_budget": int(storage_summary.get("accepted_segments_before_budget", 0)),
                "accepted_segments_after_budget": int(storage_summary.get("accepted_segments_after_budget", len(segment_df))),
                "dropped_by_budget": int(storage_summary.get("dropped_by_budget", 0)),
                "mean_segments_per_score_onset_after_budget": float(storage_summary.get("mean_segments_per_score_onset_after_budget", metrics["segments_per_score_onset"])),
                "p95_segments_per_score_onset_after_budget": float(storage_summary.get("p95_segments_per_score_onset_after_budget", 0.0)),
            }
        )
    return metrics


def write_primitive_diagnostics(
    *,
    assignments_df: pd.DataFrame,
    library_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    if assignments_df.empty or "primitive_id" not in assignments_df.columns:
        pd.DataFrame().to_csv(diagnostics_dir / "primitive_key_transposition_coverage.csv", index=False)
        pd.DataFrame().to_csv(diagnostics_dir / "primitive_condition_coverage.csv", index=False)
        pd.DataFrame().to_csv(diagnostics_dir / "primitive_reusability_score.csv", index=False)
        return

    key_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    for primitive_id, group in assignments_df.groupby("primitive_id", sort=True):
        key_centers = pd.to_numeric(group.get("chord_center_key_id", group.get("key_center", pd.Series(dtype=float))), errors="coerce")
        if "key_center" in group.columns and key_centers.max(skipna=True) <= 1.0:
            key_centers = key_centers * 87.0
        key_centers = key_centers.dropna()
        hand_distribution = _distribution_json(group.get("hand_side"))
        finger_distribution = _distribution_json(group.get("finger_set", group.get("finger_set_id")))
        chord_distribution = _distribution_json(group.get("chord_size"))
        interval_distribution = _distribution_json(group.get("interval_pattern_bucket"))
        key_rows.append(
            {
                "primitive_id": str(primitive_id),
                "min_key_center": float(key_centers.min()) if not key_centers.empty else float("nan"),
                "max_key_center": float(key_centers.max()) if not key_centers.empty else float("nan"),
                "key_center_std": float(key_centers.std(ddof=0)) if not key_centers.empty else float("nan"),
                "number_of_unique_key_centers": int(key_centers.round(3).nunique()) if not key_centers.empty else 0,
                "hand_distribution": hand_distribution,
                "finger_set_distribution": finger_distribution,
                "chord_size_distribution": chord_distribution,
                "interval_pattern_distribution": interval_distribution,
            }
        )
        wrist_values = _json_vector_stack(group.get("relative_wrist_anchor"), width=3)
        fingertip_values = _json_vector_stack(group.get("fingertip_to_target_key_offsets"), width=MAX_CONTEXT_FLAT_FINGERTIP_WIDTH)
        duration_distribution = _distribution_json(group.get("duration_bucket"))
        dynamics_distribution = _distribution_json(group.get("dynamics_bucket"))
        condition_rows.append(
            {
                "primitive_id": str(primitive_id),
                "relative_wrist_anchor_mean": json.dumps(np.nanmean(wrist_values, axis=0).round(6).tolist()) if wrist_values.size else "[]",
                "relative_wrist_anchor_std": json.dumps(np.nanstd(wrist_values, axis=0).round(6).tolist()) if wrist_values.size else "[]",
                "fingertip_to_key_offset_mean": json.dumps(np.nanmean(fingertip_values, axis=0).round(6).tolist()) if fingertip_values.size else "[]",
                "fingertip_to_key_offset_std": json.dumps(np.nanstd(fingertip_values, axis=0).round(6).tolist()) if fingertip_values.size else "[]",
                "duration_distribution": duration_distribution,
                "dynamics_distribution": dynamics_distribution,
            }
        )
        key_coverage = min(float(key_centers.nunique()) / 4.0, 1.0) if not key_centers.empty else 0.0
        finger_consistency = _purity(group.get("finger_set", group.get("finger_set_id")))
        interval_consistency = _purity(group.get("interval_pattern_bucket"))
        action_consistency = 1.0 / (1.0 + float(library_df.loc[library_df["primitive_id"].astype(str) == str(primitive_id), "reconstruction_mse"].mean()) if not library_df.empty and "reconstruction_mse" in library_df.columns else 1.0)
        sample_score = min(float(len(group)) / 8.0, 1.0)
        reusability = float(0.25 * key_coverage + 0.25 * finger_consistency + 0.20 * interval_consistency + 0.20 * action_consistency + 0.10 * sample_score)
        score_rows.append(
            {
                "primitive_id": str(primitive_id),
                "reusability_score": reusability,
                "key_coverage": key_coverage,
                "finger_consistency": finger_consistency,
                "chord_interval_consistency": interval_consistency,
                "action_trajectory_consistency": action_consistency,
                "sample_count_score": sample_score,
                "sample_count": int(len(group)),
            }
        )

    pd.DataFrame(key_rows).to_csv(diagnostics_dir / "primitive_key_transposition_coverage.csv", index=False)
    pd.DataFrame(condition_rows).to_csv(diagnostics_dir / "primitive_condition_coverage.csv", index=False)
    pd.DataFrame(score_rows).to_csv(diagnostics_dir / "primitive_reusability_score.csv", index=False)


def build_primitive_training_contract(
    *,
    primitive_root: Path,
    manifest_df: pd.DataFrame,
    segment_df: pd.DataFrame,
    library_df: pd.DataFrame,
    storage_summary: dict[str, Any] | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    del storage_summary
    action_values = pd.to_numeric(manifest_df.get("action_dim", pd.Series(dtype=float)), errors="coerce").dropna()
    action_dim = int(action_values.iloc[0]) if not action_values.empty else int(config.get("fallback_action_dim", 39))
    prior_shapes: dict[str, list[int]] = {}
    any_prior_not_action_dim = False
    any_prior_uses_non_action_target = False
    for row in library_df.itertuples(index=False):
        prior_path = Path(str(getattr(row, "prior_path", "")))
        if not prior_path.exists():
            any_prior_not_action_dim = True
            prior_shapes[str(getattr(row, "primitive_id", "unknown"))] = []
            continue
        payload = np.load(prior_path, allow_pickle=True)
        prior = np.asarray(payload["prior_mean"], dtype=np.float32)
        prior_shapes[str(row.primitive_id)] = [int(dim) for dim in prior.shape]
        if prior.ndim < 2 or int(prior.shape[-1]) != int(action_dim):
            any_prior_not_action_dim = True
        target_names = (
            [str(item) for item in np.asarray(payload["gmr_target_name"], dtype=object).reshape(-1).tolist()]
            if "gmr_target_name" in payload
            else []
        )
        if any(item in {"goals", "piano_states"} for item in target_names):
            any_prior_uses_non_action_target = True
    segment_manifest = read_json(primitive_root / "segments" / "segment_manifest.json") if (primitive_root / "segments" / "segment_manifest.json").exists() else {}
    rejection_counts = segment_manifest.get("rejection_counts", {})
    inactive_start = _percent_bool(segment_df.get("inactive_start"))
    activation_after_start = _percent_bool(segment_df.get("activation_after_start"))
    contact_near_onset = _percent_bool(segment_df.get("contact_near_onset"))
    causal_segment = bool(
        str(config.get("segmentation_strategy", config.get("segmenter_name", ""))) == "prepress_causal"
        or _percent_bool(segment_df.get("causal_segment")) >= 99.0
    )
    action_target = bool(config.get("gmr_target_actions", True)) and not any_prior_not_action_dim
    frame_mode = _primitive_frame_mode(config)
    wrist_relative_mode = frame_mode == "wrist_key_relative"
    required_library_columns = ["finger_set", "hand", "chord_size", "interval_pattern_bucket", "primitive_frame_mode"]
    missing_library_metadata = [column for column in required_library_columns if column not in library_df.columns]
    conditioned_missing = bool(
        wrist_relative_mode
        and bool(config.get("condition_gmr_on_context", False))
        and (
            library_df.empty
            or "condition_feature_count" not in library_df.columns
            or pd.to_numeric(library_df["condition_feature_count"], errors="coerce").fillna(0).le(0).any()
        )
    )
    absolute_key_position_main_clustering_driver = bool(
        wrist_relative_mode
        and isinstance(config.get("clustering"), dict)
        and not bool(config.get("clustering", {}).get("exclude_absolute_key_position_from_clustering", False))
    )
    pass_contract = bool(
        action_target
        and not any_prior_uses_non_action_target
        and not missing_library_metadata
        and not conditioned_missing
        and not absolute_key_position_main_clustering_driver
        and (not causal_segment or (inactive_start >= 99.0 and activation_after_start >= 99.0))
    )
    return {
        "action_dim": int(action_dim),
        "gmr_target_name": "actions" if bool(config.get("gmr_target_actions", True)) else "hand_joints",
        "primitive_frame_mode": frame_mode,
        "wrist_key_relative_frame": bool(wrist_relative_mode),
        "condition_gmr_on_context": bool(config.get("condition_gmr_on_context", False)),
        "condition_features_exist": bool(not conditioned_missing),
        "missing_library_metadata_columns": missing_library_metadata,
        "absolute_key_position_main_clustering_driver": absolute_key_position_main_clustering_driver,
        "no_piano_state_or_goal_target": bool(not any_prior_uses_non_action_target),
        "causal_segment": bool(causal_segment),
        "segment_alignment": "prepress_to_onset" if causal_segment else "",
        "prior_shapes": prior_shapes,
        "num_segments": int(len(segment_df)),
        "num_rejected_segments": int(segment_manifest.get("num_rejected_segments", sum(int(value) for value in rejection_counts.values())) if isinstance(rejection_counts, dict) else 0),
        "rejection_counts": rejection_counts if isinstance(rejection_counts, dict) else {},
        "percent_segments_with_inactive_start": float(inactive_start),
        "percent_segments_with_activation_after_start": float(activation_after_start),
        "percent_segments_with_contact_near_onset": float(contact_near_onset),
        "any_prior_not_action_dim": bool(any_prior_not_action_dim),
        "any_prior_uses_non_action_target": bool(any_prior_uses_non_action_target),
        "pass_training_contract": bool(pass_contract),
    }


def _percent_bool(series: pd.Series | None) -> float:
    if series is None or len(series) == 0:
        return 0.0
    values = pd.Series(series).fillna(False).astype(bool)
    return float(values.mean() * 100.0) if len(values) else 0.0


def _distribution_json(series: pd.Series | None) -> str:
    if series is None or len(series) == 0:
        return "{}"
    counts = pd.Series(series).fillna("unknown").astype(str).value_counts().to_dict()
    return json.dumps({str(key): int(value) for key, value in counts.items()}, sort_keys=True)


def _purity(series: pd.Series | None) -> float:
    if series is None or len(series) == 0:
        return 0.0
    counts = pd.Series(series).fillna("unknown").astype(str).value_counts(normalize=True)
    return float(counts.iloc[0]) if not counts.empty else 0.0


def _json_vector_stack(series: pd.Series | None, width: int) -> np.ndarray:
    if series is None or len(series) == 0:
        return np.zeros((0, int(width)), dtype=np.float32)
    rows: list[np.ndarray] = []
    for item in pd.Series(series).fillna("[]").astype(str).tolist():
        try:
            value = json.loads(item)
        except Exception:
            value = []
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        row = np.zeros((int(width),), dtype=np.float32)
        count = min(row.size, array.size)
        if count:
            row[:count] = array[:count]
        rows.append(row)
    return np.stack(rows, axis=0).astype(np.float32) if rows else np.zeros((0, int(width)), dtype=np.float32)


def _apply_stage1_defaults(config: dict[str, Any]) -> dict[str, Any]:
    output = dict(config)
    output.setdefault("online_segment_processing", True if output.get("write_slim_cache", True) else False)
    output.setdefault("save_raw_segment_chunks", save_raw_segment_chunks_enabled(output))
    output.setdefault("online_storage_format", resolve_online_storage_format(output))
    output.setdefault("gmr_resample_steps", resolve_gmr_resample_steps(output))
    output.setdefault("segmenter_name", output.get("segmentation_strategy", "note_aligned"))
    output.setdefault("segmentation_strategy", output["segmenter_name"])
    output.setdefault("segment_press_pre_steps", min(int(output.get("pre_steps", 2)), 2))
    output.setdefault("segment_press_post_steps", min(int(output.get("post_steps", 6)), 6))
    output.setdefault("segment_truncate_at_next_onset", True)
    output.setdefault("prepress_steps", 12)
    output.setdefault("post_onset_steps", 3)
    output.setdefault("min_inactive_pre_steps", 4)
    output.setdefault("min_hold_steps", 2)
    output.setdefault("activation_threshold", 0.5)
    output.setdefault("max_events_per_onset", None)
    output.setdefault("require_contact_near_onset", False)
    output.setdefault("contact_tolerance_frames", 2)
    output.setdefault("fingertip_key_distance_threshold_m", 0.025)
    output.setdefault("segment_end_mode", "onset_plus_post")
    output.setdefault("segment_start_mode", "fixed_prepress")
    output.setdefault(
        "segment_budget",
        {
            "enabled": False,
            "max_total_segments": None,
            "max_segments_per_song": None,
            "max_segments_per_episode": None,
            "max_segments_per_score_onset": None,
            "max_segments_per_target_signature": None,
            "preserve_family_balance": True,
            "ranking_metric": "causal_press_score",
            "seed": int(output.get("seed", 0)),
        },
    )
    output.setdefault(
        "feature_blocks",
        {
            "include_joint_summary": True,
            "include_joint_trajectory": True,
            "include_velocity_trajectory": True,
            "include_action_summary": True,
            "include_action_trajectory": True,
            "include_wrist_context": True,
            "include_fingertip_context": True,
            "include_score_histogram": True,
            "include_target_keyset_absolute": False,
            "include_target_keyset_relative": True,
        },
    )
    output.setdefault("use_process_pool", True)
    output.setdefault("gpu_acceleration", False)
    output.setdefault("gpu_backend_preference", ["rapids", "cupy", "torch"])
    output.setdefault("gpu_subsample_limit", 32768)
    output.setdefault("family_aware_clustering", True)
    output.setdefault("gmm_n_init", 4)
    output.setdefault("gmm_seed", int(output.get("seed", 0)))
    output.setdefault("gmm_use_staged_k_search", True)
    output.setdefault("gmm_k_screen_subset_size", 8192)
    output.setdefault("gmm_top_k_full_fits", 2)
    output.setdefault("gmm_candidate_covariance_types", [output.get("gmm_covariance_type", "full")])
    output.setdefault("gmm_screen_max_iter", 32)
    output.setdefault("adaptive_gmr_components", True)
    output.setdefault("gmr_component_candidates", [max(int(output.get("gmr_components", 8)) // 2, 1), int(output.get("gmr_components", 8))])
    output.setdefault("gmr_min_samples_per_component", 4)
    output.setdefault("gmr_strike_weight", 2.5)
    output.setdefault("primitive_split_merge_refinement", True)
    output.setdefault("primitive_merge_distance_threshold", 0.02)
    output.setdefault("primitive_split_error_improvement", 0.9)
    output.setdefault("stage1_assignment_confidence_threshold", 0.55)
    output.setdefault("enable_stage1_online_eval", True)
    output.setdefault("stage1_eval_interval_segments", 4096)
    output.setdefault("stage1_eval_subsample_size", 2048)
    output.setdefault("stage1_warn_only", True)
    output.setdefault("stage1_early_stop_enabled", False)
    output.setdefault("stage1_min_segments_before_stop_check", 2048)
    output.setdefault("stage1_max_short_segment_frac", 0.35)
    output.setdefault("stage1_max_segments_per_score_onset", 1.5)
    output.setdefault("stage1_max_p95_segment_duration_steps", 12)
    output.setdefault("stage1_max_low_confidence_frac", 0.45)
    output.setdefault("stage1_max_low_quality_primitive_frac", 0.40)
    output.setdefault("stage1_min_family_entropy", 0.80)
    output.setdefault("stage1_max_weighted_recon_error", 5.0)
    output.setdefault("stage1_patience_windows", 3)
    output.setdefault("stage1_duplicate_primitive_distance", 0.02)
    output.setdefault("relative_wrist_frame", True)
    output.setdefault("relative_key_center_frame", True)
    output.setdefault("hand_specific_normalization", True)
    output.setdefault("primitive_frame", {"mode": "absolute"})
    output.setdefault("clustering", {"mode": "absolute"})
    output.setdefault("condition_gmr_on_context", False)
    output.setdefault("gmr_target_name", "actions" if bool(output.get("gmr_target_actions", True)) else "hand_joints")
    override_workers = os.environ.get("SONATA_PRIMITIVE_NUM_WORKERS", "").strip()
    if override_workers:
        count = max(int(override_workers), 1)
        output["segment_num_workers"] = count
        output["feature_num_workers"] = count
    return output


def _normalize_covariance_types(value: Any) -> list[str]:
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",")]
        return [item for item in items if item]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return ["full"]


def _resolve_gmm_k_candidates(*, config: dict[str, Any], bucket_name: str) -> list[int]:
    by_family = config.get("gmm_k_candidates_by_family")
    if isinstance(by_family, dict):
        candidates = by_family.get(str(bucket_name), by_family.get("default"))
        if candidates is not None:
            return [int(item) for item in candidates]
    return [int(item) for item in config["gmm_k_candidates"]]


def _adaptive_gmm_reg(embedding: np.ndarray, base_reg: float) -> float:
    variance = float(np.mean(np.var(embedding, axis=0))) if embedding.size else 0.0
    return float(max(base_reg, variance * 1e-6))


def _subsample_rows(array: np.ndarray, *, size: int, seed: int) -> np.ndarray:
    if array.shape[0] <= int(size):
        return np.asarray(array, dtype=np.float32)
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(array.shape[0], size=int(size), replace=False))
    return np.asarray(array[indices], dtype=np.float32)


def _renumber_primitives(assignments_df: pd.DataFrame) -> pd.DataFrame:
    frame = assignments_df.copy()
    ordered_ids = sorted(frame["primitive_id"].astype(str).unique().tolist())
    mapping = {primitive_id: f"primitive_{index:03d}" for index, primitive_id in enumerate(ordered_ids)}
    frame["primitive_id"] = frame["primitive_id"].astype(str).map(mapping)
    label_mapping = {primitive_id: index for index, primitive_id in enumerate(ordered_ids)}
    frame["primitive_label"] = frame["primitive_id"].astype(str).map({f"primitive_{index:03d}": index for index in range(len(ordered_ids))}).astype(int)
    return frame.sort_index().reset_index(drop=True)


def _nearest_neighbor_distances(primitive_ids: list[str], priors: dict[str, np.ndarray]) -> list[float]:
    distances: list[float] = []
    for primitive_id in primitive_ids:
        source = np.asarray(priors.get(primitive_id), dtype=np.float32)
        candidates = []
        for other_id, prior in priors.items():
            if other_id == primitive_id:
                continue
            target = np.asarray(prior, dtype=np.float32)
            if source.size and target.size:
                candidates.append(float(np.mean((source - target) ** 2)))
        distances.append(min(candidates) if candidates else 1.0)
    return distances


def _metadata_predictability(assignments_df: pd.DataFrame) -> float:
    required_columns = [
        "duration_steps",
        "motion_energy",
        "chord_size",
        "key_center",
        "start_state_norm",
        "end_state_norm",
    ]
    if assignments_df.empty or any(column not in assignments_df.columns for column in required_columns):
        return float("nan")
    usable = assignments_df.dropna(subset=required_columns + ["primitive_id"]).copy()
    if usable["primitive_id"].nunique() <= 1 or len(usable) < 32:
        return float("nan")
    train = usable[usable["split"] == "train"] if "split" in usable.columns else usable
    test = usable[usable["split"] != "train"] if "split" in usable.columns else usable
    if train.empty or test.empty:
        split_index = int(len(usable) * 0.8)
        train = usable.iloc[:split_index]
        test = usable.iloc[split_index:]
    if train.empty or test.empty:
        return float("nan")
    try:
        model = LogisticRegression(max_iter=256, multi_class="auto")
        model.fit(train[required_columns].to_numpy(dtype=np.float32), train["primitive_id"].astype(str).to_numpy())
        score = model.score(test[required_columns].to_numpy(dtype=np.float32), test["primitive_id"].astype(str).to_numpy())
        return float(score)
    except Exception:
        return float("nan")


def _entropy_from_series(series: pd.Series) -> float:
    counts = series.astype(str).value_counts(normalize=True)
    values = counts.to_numpy(dtype=np.float32)
    return float(-(values * np.log(values + 1e-8)).sum()) if values.size else 0.0


def _dominant_label(values: pd.Series) -> str:
    if values.empty:
        return "unknown"
    counts = values.astype(str).value_counts()
    return str(counts.index[0]) if not counts.empty else "unknown"
