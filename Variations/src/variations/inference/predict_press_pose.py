from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from variations.models.latent_mdn import build_latent_mdn
from variations.models.mlp_baseline import build_mlp_baseline
from variations.models.pose_autoencoder import build_pose_autoencoder

JOINT_STATE_DIM = 46


@dataclass
class HandStateNormalizer:
    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def from_npz(cls, path: str | Path, device: torch.device | str = "cpu") -> "HandStateNormalizer":
        data = np.load(path, allow_pickle=False)
        mean = np.asarray(data["mean"], dtype=np.float32)[:JOINT_STATE_DIM]
        std = np.asarray(data["std"], dtype=np.float32)[:JOINT_STATE_DIM]
        return cls(
            mean=torch.as_tensor(mean, dtype=torch.float32, device=device),
            std=torch.as_tensor(std, dtype=torch.float32, device=device),
        )

    @classmethod
    def from_state(cls, state: dict[str, Any], device: torch.device | str = "cpu") -> "HandStateNormalizer":
        mean = torch.as_tensor(state["mean"], dtype=torch.float32, device=device)[:JOINT_STATE_DIM]
        std = torch.as_tensor(state["std"], dtype=torch.float32, device=device)[:JOINT_STATE_DIM]
        return cls(
            mean=mean,
            std=std,
        )

    def state_dict(self) -> dict[str, np.ndarray]:
        return {
            "mean": self.mean.detach().cpu(),
            "std": self.std.detach().cpu(),
        }

    def denormalize_hand_state(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.std.to(value.device, value.dtype) + self.mean.to(value.device, value.dtype)

    def denormalize_joint_state(self, value: torch.Tensor) -> torch.Tensor:
        return self.denormalize_hand_state(value)


@torch.no_grad()
def predict_press_pose(
    target_keys: torch.Tensor,
    model,
    model_type: str,
    normalizer: HandStateNormalizer,
    device: str | torch.device = "cuda",
    num_diffusion_steps: int = 50,
    diffusion=None,
) -> torch.Tensor:
    device = torch.device(device)
    target_keys = target_keys.to(device=device, dtype=torch.float32)
    model = model.to(device)
    model.eval()
    if model_type == "mlp_baseline":
        y_norm = model(target_keys)
        y_norm = y_norm[:, :JOINT_STATE_DIM]
        return normalizer.denormalize_hand_state(y_norm)
    if model_type == "diffusion":
        if diffusion is None:
            raise ValueError("diffusion must be provided for model_type='diffusion'")
        y_norm = diffusion.p_sample_loop(model, (target_keys.shape[0], JOINT_STATE_DIM), target_keys, num_inference_steps=num_diffusion_steps)
        return normalizer.denormalize_hand_state(y_norm)
    if model_type == "latent_mdn":
        mdn_model, autoencoder, latent_stats = model
        return predict_with_latent_mdn(
            target_keys,
            mdn_model,
            autoencoder,
            latent_stats,
            normalizer,
            strategy="best_component",
        )
    raise ValueError(f"Unsupported model_type: {model_type}")


@torch.no_grad()
def predict_with_latent_mdn(
    target_keys: torch.Tensor,
    mdn_model,
    autoencoder,
    latent_stats: dict[str, torch.Tensor],
    normalizer: HandStateNormalizer,
    strategy: str = "best_component",
    num_samples: int = 16,
) -> torch.Tensor:
    """
    Predicts denormalized joint_state[46] from target_keys[88].
    """
    if strategy != "best_component":
        raise NotImplementedError("Only strategy='best_component' is implemented for the first latent MDN pass.")
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
    pose_norm = autoencoder.decode(z)
    return normalizer.denormalize_hand_state(pose_norm)


def load_mlp_baseline_checkpoint(checkpoint_path: str | Path, device: str | torch.device = "cpu"):
    device = torch.device(device)
    try:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location=device)
    config = payload["config"]
    model = build_mlp_baseline(config).to(device)
    state = payload.get("model_state_dict") or payload.get("model")
    model.load_state_dict(state)
    model.eval()
    normalizer = HandStateNormalizer.from_state(payload["normalizer"], device=device)
    return model, normalizer, config, payload


def load_latent_mdn_checkpoint(checkpoint_path: str | Path, device: str | torch.device = "cpu"):
    device = torch.device(device)
    try:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location=device)
    config = payload["config"]
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
        raise FileNotFoundError(
            f"Autoencoder checkpoint not found at {payload['autoencoder_checkpoint']} "
            f"or under the latent MDN run at {run_root / 'autoencoder' / 'checkpoints'}"
        )
    try:
        ae_payload = torch.load(autoencoder_checkpoint, map_location=device, weights_only=False)
    except TypeError:
        ae_payload = torch.load(autoencoder_checkpoint, map_location=device)
    autoencoder = build_pose_autoencoder(config).to(device)
    autoencoder.load_state_dict(ae_payload["model_state_dict"])
    autoencoder.eval()
    normalizer = HandStateNormalizer.from_state(payload["normalizer"], device=device)
    latent_stats = {
        "z_mean": torch.as_tensor(payload["latent_stats"]["z_mean"], dtype=torch.float32, device=device),
        "z_std": torch.as_tensor(payload["latent_stats"]["z_std"], dtype=torch.float32, device=device),
    }
    return mdn_model, autoencoder, latent_stats, normalizer, config, payload
