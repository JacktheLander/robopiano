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

    def fit(self, trajectories: np.ndarray) -> "PhaseGMR":
        if trajectories.ndim != 3:
            raise ValueError("trajectories must have shape [num_examples, horizon, dim]")
        num_examples, horizon, output_dim = trajectories.shape
        phases = np.linspace(0.0, 1.0, horizon, dtype=np.float32)
        x = np.repeat(phases[None, :, None], num_examples, axis=0)
        design = np.concatenate([x, trajectories], axis=-1).reshape(num_examples * horizon, 1 + output_dim)
        self.model = GaussianMixture(
            n_components=min(self.n_components, max(1, design.shape[0] // 8)),
            covariance_type="full",
            reg_covar=self.reg_covar,
            random_state=self.random_state,
        )
        self.model.fit(design)
        self.output_dim = output_dim
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
                mu = self.model.means_[idx]
                sigma = self.model.covariances_[idx]
                mu_x = float(mu[0])
                mu_y = mu[1:]
                sigma_xx = float(sigma[0, 0]) + self.reg_covar
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
            "n_components": self.n_components,
            "reg_covar": self.reg_covar,
            "random_state": self.random_state,
            "output_dim": getattr(self, "output_dim", None),
            "weights": self.model.weights_,
            "means": self.model.means_,
            "covariances": self.model.covariances_,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "PhaseGMR":
        instance = cls(
            n_components=int(payload["n_components"]),
            reg_covar=float(payload["reg_covar"]),
            random_state=int(payload["random_state"]),
        )
        instance.output_dim = int(payload["output_dim"])
        instance.model = GaussianMixture(
            n_components=instance.n_components,
            covariance_type="full",
            reg_covar=instance.reg_covar,
            random_state=instance.random_state,
        )
        instance.model.weights_ = np.asarray(payload["weights"], dtype=np.float64)
        instance.model.means_ = np.asarray(payload["means"], dtype=np.float64)
        instance.model.covariances_ = np.asarray(payload["covariances"], dtype=np.float64)
        precisions = np.linalg.inv(instance.model.covariances_)
        instance.model.precisions_ = precisions
        instance.model.precisions_cholesky_ = np.linalg.cholesky(precisions)
        return instance
