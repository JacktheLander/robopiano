from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.mixture import GaussianMixture


@dataclass
class PhaseGMR:
    n_components: int = 4
    reg_covar: float = 1e-5
    random_state: int = 0
    covariance_type: str = "full"
    n_init: int = 1

    def fit(self, trajectories: np.ndarray) -> "PhaseGMR":
        if trajectories.ndim != 3:
            raise ValueError("trajectories must have shape [num_examples, horizon, dim]")
        num_examples, horizon, output_dim = trajectories.shape
        phases = np.linspace(0.0, 1.0, horizon, dtype=np.float32)
        x = np.repeat(phases[None, :, None], num_examples, axis=0)
        design = np.concatenate([x, trajectories], axis=-1).reshape(num_examples * horizon, 1 + output_dim)
        effective_components = min(
            int(self.n_components),
            max(1, num_examples),
            max(1, design.shape[0] // 8),
        )
        self.model = GaussianMixture(
            n_components=max(1, effective_components),
            covariance_type=str(self.covariance_type),
            reg_covar=float(self.reg_covar),
            random_state=int(self.random_state),
            n_init=max(int(self.n_init), 1),
        )
        self.model.fit(design)
        self.output_dim = int(output_dim)
        return self

    def predict(self, phases: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        phases = np.asarray(phases, dtype=np.float32).reshape(-1, 1)
        means = []
        covariances = []
        for phase in phases[:, 0]:
            component_means = []
            component_covs = []
            weights = []
            for idx in range(self.model.n_components):
                mu = np.asarray(self.model.means_[idx], dtype=np.float64)
                sigma = self._component_covariance(idx)
                mu_x = float(mu[0])
                mu_y = mu[1:]
                sigma_xx = float(sigma[0, 0]) + float(self.reg_covar)
                sigma_xy = sigma[0, 1:]
                sigma_yx = sigma[1:, 0:1]
                sigma_yy = sigma[1:, 1:]
                conditional_mean = mu_y + (sigma_yx[:, 0] / sigma_xx) * (phase - mu_x)
                conditional_cov = sigma_yy - (sigma_yx @ sigma_xy[None, :]) / sigma_xx
                component_means.append(conditional_mean)
                component_covs.append(conditional_cov)
                log_weight = np.log(self.model.weights_[idx] + 1e-8) - 0.5 * (
                    np.log(2.0 * np.pi * sigma_xx) + ((phase - mu_x) ** 2) / sigma_xx
                )
                weights.append(log_weight)
            weights = np.asarray(weights, dtype=np.float64)
            weights = np.exp(weights - weights.max())
            weights = weights / max(weights.sum(), 1e-8)
            mixture_mean = np.sum(np.asarray(component_means) * weights[:, None], axis=0)
            second_moment = np.zeros((self.output_dim, self.output_dim), dtype=np.float32)
            for weight, comp_mean, comp_cov in zip(weights, component_means, component_covs):
                second_moment += weight * (comp_cov + np.outer(comp_mean, comp_mean))
            mixture_cov = second_moment - np.outer(mixture_mean, mixture_mean)
            means.append(mixture_mean.astype(np.float32))
            covariances.append(mixture_cov.astype(np.float32))
        return np.stack(means, axis=0), np.stack(covariances, axis=0)

    def to_payload(self) -> dict[str, Any]:
        return {
            "n_components": int(self.n_components),
            "reg_covar": float(self.reg_covar),
            "random_state": int(self.random_state),
            "covariance_type": str(self.covariance_type),
            "n_init": int(self.n_init),
            "output_dim": getattr(self, "output_dim", None),
            "weights": np.asarray(self.model.weights_, dtype=np.float64),
            "means": np.asarray(self.model.means_, dtype=np.float64),
            "covariances": np.asarray(self.model.covariances_, dtype=np.float64),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "PhaseGMR":
        instance = cls(
            n_components=int(payload["n_components"]),
            reg_covar=float(payload["reg_covar"]),
            random_state=int(payload["random_state"]),
            covariance_type=str(payload.get("covariance_type", "full")),
            n_init=int(payload.get("n_init", 1)),
        )
        instance.output_dim = int(payload["output_dim"])
        instance.model = GaussianMixture(
            n_components=instance.n_components,
            covariance_type=instance.covariance_type,
            reg_covar=instance.reg_covar,
            random_state=instance.random_state,
            n_init=instance.n_init,
        )
        instance.model.weights_ = np.asarray(payload["weights"], dtype=np.float64)
        instance.model.means_ = np.asarray(payload["means"], dtype=np.float64)
        instance.model.covariances_ = np.asarray(payload["covariances"], dtype=np.float64)
        precisions, precisions_cholesky = _precisions_from_covariances(instance.model.covariances_, instance.covariance_type)
        instance.model.precisions_ = precisions
        instance.model.precisions_cholesky_ = precisions_cholesky
        return instance

    def _component_covariance(self, index: int) -> np.ndarray:
        covariance_type = getattr(self.model, "covariance_type", "full")
        if covariance_type == "full":
            return np.asarray(self.model.covariances_[index], dtype=np.float64)
        if covariance_type == "diag":
            return np.diag(np.asarray(self.model.covariances_[index], dtype=np.float64))
        if covariance_type == "tied":
            return np.asarray(self.model.covariances_, dtype=np.float64)
        if covariance_type == "spherical":
            scale = float(self.model.covariances_[index])
            return np.eye(self.model.means_.shape[1], dtype=np.float64) * scale
        raise ValueError(f"Unsupported covariance_type={covariance_type!r} for PhaseGMR.")


def fit_phase_gmr_with_selection(
    *,
    trajectories: np.ndarray,
    component_candidates: list[int],
    reg_covar: float,
    random_state: int,
    min_samples_per_component: int,
    strike_weight: float,
    covariance_type: str = "full",
    n_init: int = 1,
) -> tuple[PhaseGMR, dict[str, Any]]:
    if trajectories.ndim != 3 or trajectories.shape[0] == 0:
        raise ValueError("trajectories must have shape [num_examples, horizon, dim] with at least one example")
    num_examples = int(trajectories.shape[0])
    filtered_candidates = sorted(
        {
            max(int(candidate), 1)
            for candidate in component_candidates
            if int(candidate) <= max(num_examples // max(int(min_samples_per_component), 1), 1)
        }
    )
    if not filtered_candidates:
        filtered_candidates = [1]

    time_weights, dim_weights = build_strike_weighting(trajectories=trajectories, strike_weight=float(strike_weight))
    best_model: PhaseGMR | None = None
    best_metrics: dict[str, Any] | None = None
    best_score = float("inf")
    diagnostics: list[dict[str, Any]] = []

    for components in filtered_candidates:
        adaptive_reg = adaptive_reg_covar(trajectories=trajectories, base_reg=float(reg_covar), n_components=int(components))
        model = PhaseGMR(
            n_components=int(components),
            reg_covar=adaptive_reg,
            random_state=int(random_state),
            covariance_type=str(covariance_type),
            n_init=max(int(n_init), 1),
        ).fit(trajectories)
        phases = np.linspace(0.0, 1.0, trajectories.shape[1], dtype=np.float32)
        predicted_mean, _ = model.predict(phases)
        metrics = reconstruction_metrics(
            trajectories=trajectories,
            predicted_mean=predicted_mean,
            time_weights=time_weights,
            dim_weights=dim_weights,
        )
        metrics["component_count"] = int(components)
        metrics["reg_covar"] = float(adaptive_reg)
        diagnostics.append(metrics)
        score = float(metrics["weighted_strike_error"] + 0.25 * metrics["reconstruction_mse"] + 0.05 * metrics["reconstruction_l1"])
        if score < best_score:
            best_score = score
            best_model = model
            best_metrics = metrics

    assert best_model is not None and best_metrics is not None
    return best_model, {
        "selected": best_metrics,
        "candidates": diagnostics,
        "time_weights": time_weights.astype(np.float32),
        "dim_weights": dim_weights.astype(np.float32),
    }


def build_strike_weighting(trajectories: np.ndarray, strike_weight: float) -> tuple[np.ndarray, np.ndarray]:
    if trajectories.shape[1] <= 1:
        return np.ones((trajectories.shape[1],), dtype=np.float32), np.ones((trajectories.shape[2],), dtype=np.float32)
    delta = np.diff(trajectories, axis=1, prepend=trajectories[:, :1])
    time_importance = np.mean(np.linalg.norm(delta, axis=2), axis=0).astype(np.float32)
    dim_importance = np.mean(np.abs(delta), axis=(0, 1)).astype(np.float32)
    time_importance = _normalize_importance(time_importance)
    dim_importance = _normalize_importance(dim_importance)
    time_weights = 1.0 + float(strike_weight) * time_importance
    dim_weights = 1.0 + float(strike_weight) * dim_importance
    return time_weights.astype(np.float32), dim_weights.astype(np.float32)


def reconstruction_metrics(
    *,
    trajectories: np.ndarray,
    predicted_mean: np.ndarray,
    time_weights: np.ndarray,
    dim_weights: np.ndarray,
) -> dict[str, float]:
    residual = trajectories - predicted_mean[None, :, :]
    squared = residual ** 2
    absolute = np.abs(residual)
    weighted = squared * time_weights[None, :, None] * dim_weights[None, None, :]
    return {
        "reconstruction_mse": float(np.mean(squared)),
        "reconstruction_l1": float(np.mean(absolute)),
        "weighted_strike_error": float(np.mean(weighted)),
    }


def adaptive_reg_covar(*, trajectories: np.ndarray, base_reg: float, n_components: int) -> float:
    variance_scale = float(np.mean(np.var(trajectories.reshape(-1, trajectories.shape[-1]), axis=0)))
    return float(max(base_reg, (variance_scale * 1e-4) / max(int(n_components), 1)))


def _normalize_importance(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    values = values - float(values.min())
    maximum = float(values.max())
    return values / maximum if maximum > 0 else np.zeros_like(values)


def _precisions_from_covariances(covariances: np.ndarray, covariance_type: str) -> tuple[np.ndarray, np.ndarray]:
    if covariance_type == "full":
        precisions = np.linalg.inv(covariances)
        return precisions, np.linalg.cholesky(precisions)
    if covariance_type == "diag":
        precisions = 1.0 / np.clip(covariances, 1e-12, None)
        return precisions, np.sqrt(precisions)
    if covariance_type == "tied":
        precision = np.linalg.inv(covariances)
        return precision, np.linalg.cholesky(precision)
    if covariance_type == "spherical":
        precisions = 1.0 / np.clip(covariances, 1e-12, None)
        return precisions, np.sqrt(precisions)
    raise ValueError(f"Unsupported covariance_type={covariance_type!r}")
