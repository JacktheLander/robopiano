from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from variations.models.latent_mdn import build_latent_mdn
from variations.models.pose_autoencoder import build_pose_autoencoder

FINGERPRED_OUTPUT_MODE = "active_fingertips"
FINGERPRED_TARGET_DIM = 30


@dataclass
class FingerPredNormalizer:
    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def from_npz(cls, path: str | Path, device: torch.device | str = "cpu") -> "FingerPredNormalizer":
        data = np.load(path, allow_pickle=False)
        mean = np.asarray(data["mean"], dtype=np.float32)
        std = np.asarray(data["std"], dtype=np.float32)
        if mean.shape[0] != FINGERPRED_TARGET_DIM:
            raise ValueError(f"FingerPred normalizer must be {FINGERPRED_TARGET_DIM}D, got {mean.shape[0]}")
        return cls(
            mean=torch.as_tensor(mean, dtype=torch.float32, device=device),
            std=torch.as_tensor(std, dtype=torch.float32, device=device),
        )

    @classmethod
    def from_state(cls, state: dict[str, Any], device: torch.device | str = "cpu") -> "FingerPredNormalizer":
        mean = torch.as_tensor(state["mean"], dtype=torch.float32, device=device)
        std = torch.as_tensor(state["std"], dtype=torch.float32, device=device)
        if mean.shape[0] != FINGERPRED_TARGET_DIM:
            raise ValueError(f"FingerPred normalizer must be {FINGERPRED_TARGET_DIM}D, got {mean.shape[0]}")
        return cls(mean=mean, std=std)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"mean": self.mean.detach().cpu(), "std": self.std.detach().cpu()}

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.mean.to(value.device, value.dtype)) / self.std.to(value.device, value.dtype)

    def denormalize(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.std.to(value.device, value.dtype) + self.mean.to(value.device, value.dtype)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, coord_mask: torch.Tensor) -> torch.Tensor:
    mask = coord_mask.to(device=pred.device, dtype=pred.dtype)
    denom = mask.sum().clamp_min(1.0)
    return (((pred - target) ** 2) * mask).sum() / denom


def masked_autoencoder_input(target_norm: torch.Tensor, coord_mask: torch.Tensor) -> torch.Tensor:
    return target_norm * coord_mask.to(device=target_norm.device, dtype=target_norm.dtype)


@torch.no_grad()
def predict_with_fingerpred(
    target_keys: torch.Tensor,
    mdn_model: torch.nn.Module,
    autoencoder: torch.nn.Module,
    latent_stats: dict[str, torch.Tensor],
    normalizer: FingerPredNormalizer,
    *,
    strategy: str = "best_component",
) -> torch.Tensor:
    if strategy != "best_component":
        raise NotImplementedError("Only strategy='best_component' is implemented for FingerPred.")
    device = next(mdn_model.parameters()).device
    target_keys = target_keys.to(device=device, dtype=torch.float32)
    mdn_model.eval()
    autoencoder.eval()
    mixture_logits, means_std, _log_stds = mdn_model(target_keys)
    best_idx = torch.argmax(mixture_logits, dim=-1)
    batch_idx = torch.arange(means_std.shape[0], device=means_std.device)
    z_std = means_std[batch_idx, best_idx]
    z_mean = latent_stats["z_mean"].to(device=device, dtype=z_std.dtype)
    z_scale = latent_stats["z_std"].to(device=device, dtype=z_std.dtype)
    z = z_std * z_scale + z_mean
    pred_norm = autoencoder.decode(z)
    return normalizer.denormalize(pred_norm)


def _load_torch(path: str | Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_fingerpred_checkpoint(checkpoint_path: str | Path, device: str | torch.device = "cpu"):
    device = torch.device(device)
    payload = _load_torch(checkpoint_path, device)
    config = payload["config"]
    if str(payload.get("model_type", config.get("model_type", ""))) != "fingerpred":
        raise ValueError(f"Checkpoint is not a fingerpred checkpoint: {checkpoint_path}")
    mdn_model = build_latent_mdn(config).to(device)
    mdn_model.load_state_dict(payload["model_state_dict"])
    mdn_model.eval()

    mdn_path = Path(checkpoint_path).resolve()
    autoencoder_checkpoint = Path(payload["autoencoder_checkpoint"])
    if not autoencoder_checkpoint.is_file():
        run_root = mdn_path.parent.parent.parent
        sibling = run_root / "autoencoder" / "checkpoints" / autoencoder_checkpoint.name
        if sibling.is_file():
            autoencoder_checkpoint = sibling
    if not autoencoder_checkpoint.is_file():
        raise FileNotFoundError(f"FingerPred autoencoder checkpoint not found: {payload['autoencoder_checkpoint']}")
    ae_payload = _load_torch(autoencoder_checkpoint, device)
    autoencoder = build_pose_autoencoder(config).to(device)
    autoencoder.load_state_dict(ae_payload["model_state_dict"])
    autoencoder.eval()

    normalizer = FingerPredNormalizer.from_state(payload["normalizer"], device=device)
    latent_stats = {
        "z_mean": torch.as_tensor(payload["latent_stats"]["z_mean"], dtype=torch.float32, device=device),
        "z_std": torch.as_tensor(payload["latent_stats"]["z_std"], dtype=torch.float32, device=device),
    }
    return mdn_model, autoencoder, latent_stats, normalizer, config, payload


def active_fingertip_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    tip_mask: torch.Tensor,
    *,
    pred_norm: torch.Tensor | None = None,
    target_norm: torch.Tensor | None = None,
    coord_mask: torch.Tensor | None = None,
    success_thresholds: tuple[float, ...] = (0.005, 0.01, 0.02, 0.05),
) -> dict[str, float]:
    if coord_mask is None:
        coord_mask = tip_mask.repeat_interleave(3, dim=1)
    metrics: dict[str, float] = {
        "active_fingertip_mse": float(masked_mse(pred, target, coord_mask).detach().cpu().item()),
    }
    metrics["active_fingertip_rmse"] = float(np.sqrt(metrics["active_fingertip_mse"]))
    if pred_norm is not None and target_norm is not None:
        metrics["active_fingertip_normalized_mse"] = float(masked_mse(pred_norm, target_norm, coord_mask).detach().cpu().item())

    pred_np = pred.detach().cpu().numpy().reshape(-1, 10, 3)
    target_np = target.detach().cpu().numpy().reshape(-1, 10, 3)
    mask_np = tip_mask.detach().cpu().numpy().astype(bool)
    dist = np.linalg.norm(pred_np - target_np, axis=2)
    width_dist = np.abs(pred_np[:, :, 0] - target_np[:, :, 0])
    active_dist = dist[mask_np]
    active_width_dist = width_dist[mask_np]
    metrics["active_tip_count"] = float(active_dist.size)
    metrics["active_tip_examples"] = float(mask_np.any(axis=1).sum())
    if active_dist.size:
        metrics["active_tip_distance_mean"] = float(np.mean(active_dist))
        metrics["active_tip_distance_median"] = float(np.median(active_dist))
        metrics["active_tip_distance_p95"] = float(np.percentile(active_dist, 95))
        metrics["active_tip_width_distance_mean"] = float(np.mean(active_width_dist))
        metrics["active_tip_width_distance_median"] = float(np.median(active_width_dist))
        metrics["active_tip_width_distance_p95"] = float(np.percentile(active_width_dist, 95))
    else:
        metrics["active_tip_distance_mean"] = float("nan")
        metrics["active_tip_distance_median"] = float("nan")
        metrics["active_tip_distance_p95"] = float("nan")
        metrics["active_tip_width_distance_mean"] = float("nan")
        metrics["active_tip_width_distance_median"] = float("nan")
        metrics["active_tip_width_distance_p95"] = float("nan")

    example_has_active = mask_np.any(axis=1)
    if example_has_active.any():
        per_example_max = np.where(mask_np, dist, -np.inf).max(axis=1)[example_has_active]
        per_example_width_max = np.where(mask_np, width_dist, -np.inf).max(axis=1)[example_has_active]
        for threshold in success_thresholds:
            key = str(threshold).replace(".", "p")
            metrics[f"active_tip_success_at_{key}"] = float(np.mean(per_example_max <= float(threshold)))
            metrics[f"active_tip_width_success_at_{key}"] = float(np.mean(per_example_width_max <= float(threshold)))
    else:
        for threshold in success_thresholds:
            key = str(threshold).replace(".", "p")
            metrics[f"active_tip_success_at_{key}"] = float("nan")
            metrics[f"active_tip_width_success_at_{key}"] = float("nan")
    return metrics
