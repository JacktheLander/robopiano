from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch

from variations.diffusion.trainer import _resolve_norm_stats_npz, load_model_for_inference, resolve_device
from variations.inference.predict_press_pose import (
    HandStateNormalizer,
    load_latent_mdn_checkpoint,
    load_mlp_baseline_checkpoint,
    predict_press_pose,
    predict_with_latent_mdn,
)

ModelTypeLit = Literal["mlp_baseline", "diffusion", "latent_mdn"]


@dataclass
class LoadedSimulationModel:
    model_type: ModelTypeLit
    device: torch.device
    normalizer: HandStateNormalizer
    # Diffusion-only
    diffusion: Any | None = None
    diffusion_steps: int = 50
    # Raw model handle(s) for predict_press_pose
    model: Any = None

    def predict_hand_states(self, target_keys: np.ndarray, *, batch_size: int = 256) -> np.ndarray:
        """(N, 88) float32 -> (N, 46) float32 denormalized joint states."""
        keys = np.asarray(target_keys, dtype=np.float32)
        if keys.ndim != 2 or keys.shape[1] != 88:
            raise ValueError(f"target_keys must be (N, 88), got {keys.shape}")
        out_list: list[np.ndarray] = []
        n = keys.shape[0]
        dev = self.device
        for start in range(0, n, int(batch_size)):
            batch = keys[start : start + batch_size]
            t = torch.as_tensor(batch, dtype=torch.float32, device=dev)
            if self.model_type == "latent_mdn":
                mdn_model, autoencoder, latent_stats = self.model
                pred = predict_with_latent_mdn(t, mdn_model, autoencoder, latent_stats, self.normalizer)
            else:
                pred = predict_press_pose(
                    t,
                    self.model,
                    self.model_type,
                    self.normalizer,
                    dev,
                    num_diffusion_steps=int(self.diffusion_steps),
                    diffusion=self.diffusion,
                )
            out_list.append(pred.detach().cpu().numpy().astype(np.float32, copy=False))
        return np.concatenate(out_list, axis=0)


def load_simulation_model(
    checkpoint_path: str,
    model_type: str,
    *,
    device: str = "auto",
    diffusion_steps: int | None = None,
) -> LoadedSimulationModel:
    """Load a Variations checkpoint for MAESTRO rollout inference."""
    mt = str(model_type).strip().lower().replace("-", "_")
    # accept common aliases
    if mt in {"mlp", "mlpbaseline"}:
        mt = "mlp_baseline"
    if mt in {"mdn", "latentmdn"}:
        mt = "latent_mdn"
    if mt == "fingerpred":
        raise ValueError(
            "FingerPred checkpoints predict 30D fingertip positions and cannot be used for "
            "RoboPianist joint-state simulation or online rollout, which require 46D hand joints."
        )

    if mt not in ("mlp_baseline", "diffusion", "latent_mdn"):
        raise ValueError(f"Unsupported model_type: {model_type!r}; use mlp_baseline, diffusion, or latent_mdn.")

    dev = resolve_device(str(device))

    if mt == "mlp_baseline":
        model, normalizer, config, _payload = load_mlp_baseline_checkpoint(checkpoint_path, device=dev)
        return LoadedSimulationModel(model_type="mlp_baseline", device=dev, normalizer=normalizer, model=model)

    if mt == "diffusion":
        model, diffusion, config, _mean, _std = load_model_for_inference(checkpoint_path, device=dev)
        try:
            payload = torch.load(checkpoint_path, map_location=dev, weights_only=False)
        except TypeError:
            payload = torch.load(checkpoint_path, map_location=dev)
        norm_path = _resolve_norm_stats_npz(payload, payload["config"])
        normalizer = HandStateNormalizer.from_npz(norm_path, device=dev)
        steps = diffusion_steps if diffusion_steps is not None else int(config.get("diffusion", {}).get("ddim_steps", 50))
        return LoadedSimulationModel(
            model_type="diffusion",
            device=dev,
            normalizer=normalizer,
            model=model,
            diffusion=diffusion,
            diffusion_steps=steps,
        )

    # latent_mdn
    mdn_model, autoencoder, latent_stats, normalizer, config, _payload = load_latent_mdn_checkpoint(
        checkpoint_path, device=dev
    )
    triple = (mdn_model, autoencoder, latent_stats)
    return LoadedSimulationModel(model_type="latent_mdn", device=dev, normalizer=normalizer, model=triple)
