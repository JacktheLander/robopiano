from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except Exception:  # pragma: no cover
    torch = None
    nn = None
    F = None


@dataclass
class ControlLatentResult:
    latent_matrix: np.ndarray
    scaler: StandardScaler
    feature_dim: int
    latent_dim: int
    family_names: list[str]
    phase_names: list[str]
    training_loss: float
    mode: str


class _Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, num_families: int, num_phases: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.latent = nn.Linear(hidden_dim, latent_dim)
        self.family_head = nn.Linear(latent_dim, num_families)
        self.phase_head = nn.Linear(latent_dim, num_phases)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.backbone(features)
        latent = self.latent(hidden)
        latent = F.normalize(latent, dim=-1)
        return latent, self.family_head(latent), self.phase_head(latent)


def learn_control_latent(
    feature_matrix: np.ndarray,
    segment_df: pd.DataFrame,
    config: dict[str, Any],
) -> ControlLatentResult:
    latent_cfg = dict(config.get("control_latent", {}))
    enabled = bool(latent_cfg.get("enabled", False))
    scaled, scaler = _scaled_features(feature_matrix)
    if not enabled or torch is None or len(segment_df) < 8:
        return ControlLatentResult(
            latent_matrix=scaled.astype(np.float32),
            scaler=scaler,
            feature_dim=int(feature_matrix.shape[1]) if feature_matrix.ndim == 2 else 0,
            latent_dim=int(scaled.shape[1]) if scaled.ndim == 2 else 0,
            family_names=[],
            phase_names=[],
            training_loss=float("nan"),
            mode="scaled_features",
        )

    coarse_family = _string_labels(segment_df.get("coarse_family"), fallback="other")
    control_phase = _string_labels(segment_df.get("control_phase"), fallback="whole_event")
    family_names = sorted(set(coarse_family.tolist()))
    phase_names = sorted(set(control_phase.tolist()))
    family_to_index = {name: idx for idx, name in enumerate(family_names)}
    phase_to_index = {name: idx for idx, name in enumerate(phase_names)}
    family_labels = np.asarray([family_to_index[item] for item in coarse_family.tolist()], dtype=np.int64)
    phase_labels = np.asarray([phase_to_index[item] for item in control_phase.tolist()], dtype=np.int64)
    combined_labels = np.asarray(
        [f"{family}:{phase}" for family, phase in zip(coarse_family.tolist(), control_phase.tolist(), strict=False)],
        dtype=object,
    )

    hidden_dim = int(latent_cfg.get("hidden_dim", 128))
    latent_dim = min(int(latent_cfg.get("latent_dim", 32)), max(4, scaled.shape[1]))
    epochs = int(latent_cfg.get("epochs", 12))
    batch_size = min(int(latent_cfg.get("batch_size", 256)), len(scaled))
    triplet_margin = float(latent_cfg.get("triplet_margin", 0.2))
    family_loss_weight = float(latent_cfg.get("family_loss_weight", 1.0))
    phase_loss_weight = float(latent_cfg.get("phase_loss_weight", 0.5))
    triplet_loss_weight = float(latent_cfg.get("triplet_loss_weight", 0.5))
    learning_rate = float(latent_cfg.get("learning_rate", 1e-3))
    rng = np.random.default_rng(int(config.get("seed", 0)))

    device = torch.device(str(latent_cfg.get("device", "cpu")))
    model = _Encoder(
        input_dim=int(scaled.shape[1]),
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_families=max(len(family_names), 1),
        num_phases=max(len(phase_names), 1),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    features_tensor = torch.from_numpy(scaled.astype(np.float32))
    family_tensor = torch.from_numpy(family_labels)
    phase_tensor = torch.from_numpy(phase_labels)
    model.train()
    final_loss = float("nan")
    for _ in range(max(epochs, 1)):
        batch_indices = rng.choice(len(scaled), size=batch_size, replace=len(scaled) < batch_size)
        batch_features = features_tensor[batch_indices].to(device)
        batch_family = family_tensor[batch_indices].to(device)
        batch_phase = phase_tensor[batch_indices].to(device)
        latent, family_logits, phase_logits = model(batch_features)
        loss = family_loss_weight * F.cross_entropy(family_logits, batch_family)
        loss = loss + phase_loss_weight * F.cross_entropy(phase_logits, batch_phase)
        triplets = _sample_triplets(combined_labels[batch_indices], rng=rng, max_triplets=max(batch_size // 2, 8))
        if triplets:
            anchor = latent[torch.as_tensor([item[0] for item in triplets], device=device)]
            positive = latent[torch.as_tensor([item[1] for item in triplets], device=device)]
            negative = latent[torch.as_tensor([item[2] for item in triplets], device=device)]
            loss = loss + triplet_loss_weight * F.triplet_margin_loss(
                anchor,
                positive,
                negative,
                margin=triplet_margin,
            )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu().item())

    model.eval()
    with torch.no_grad():
        latent_matrix = model(features_tensor.to(device))[0].cpu().numpy().astype(np.float32)
    return ControlLatentResult(
        latent_matrix=latent_matrix,
        scaler=scaler,
        feature_dim=int(feature_matrix.shape[1]) if feature_matrix.ndim == 2 else 0,
        latent_dim=int(latent_dim),
        family_names=family_names,
        phase_names=phase_names,
        training_loss=final_loss,
        mode="learned_metric",
    )


def _scaled_features(feature_matrix: np.ndarray) -> tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    if feature_matrix.size == 0:
        return feature_matrix.astype(np.float32), scaler
    scaled = scaler.fit_transform(feature_matrix).astype(np.float32)
    return scaled, scaler


def _string_labels(series: pd.Series | Any, fallback: str) -> np.ndarray:
    if series is None:
        return np.asarray([fallback], dtype=object)
    values = pd.Series(series).fillna(fallback).astype(str).replace({"nan": fallback})
    return values.to_numpy(dtype=object)


def _sample_triplets(labels: np.ndarray, *, rng: np.random.Generator, max_triplets: int) -> list[tuple[int, int, int]]:
    groups: dict[str, np.ndarray] = {}
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        if len(indices) > 0:
            groups[str(label)] = indices
    valid_anchors = [label for label, indices in groups.items() if len(indices) >= 2]
    negatives = [label for label, indices in groups.items() if len(indices) >= 1]
    triplets: list[tuple[int, int, int]] = []
    if len(valid_anchors) < 1 or len(negatives) < 2:
        return triplets
    for _ in range(max_triplets):
        anchor_label = str(rng.choice(valid_anchors))
        negative_candidates = [label for label in negatives if label != anchor_label]
        if not negative_candidates:
            continue
        negative_label = str(rng.choice(negative_candidates))
        anchor_index, positive_index = rng.choice(groups[anchor_label], size=2, replace=False).tolist()
        negative_index = int(rng.choice(groups[negative_label]))
        triplets.append((int(anchor_index), int(positive_index), negative_index))
    return triplets
