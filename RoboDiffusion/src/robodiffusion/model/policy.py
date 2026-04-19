from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from robodiffusion.model.diffusion import DiffusionConfig, DiffusionTransformer, GaussianDiffusionScheduler
from robodiffusion.utils.checkpointing import load_checkpoint


@dataclass
class PolicyMetadata:
    score_dim: int
    state_dim: int
    action_dim: int
    obs_horizon: int
    pred_horizon: int
    action_execute_horizon: int
    observation_spec: dict[str, Any] | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "score_dim": int(self.score_dim),
            "state_dim": int(self.state_dim),
            "action_dim": int(self.action_dim),
            "obs_horizon": int(self.obs_horizon),
            "pred_horizon": int(self.pred_horizon),
            "action_execute_horizon": int(self.action_execute_horizon),
            "observation_spec": dict(self.observation_spec or {}),
        }


class RoboDiffusionPolicy:
    def __init__(
        self,
        *,
        model: DiffusionTransformer,
        scheduler: GaussianDiffusionScheduler,
        metadata: PolicyMetadata,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.scheduler = scheduler.to(device)
        self.metadata = metadata
        self.device = device

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path, device: str | torch.device = "cpu") -> "RoboDiffusionPolicy":
        resolved_device = torch.device(device)
        payload = load_checkpoint(checkpoint_path, map_location=resolved_device)
        diffusion_config = DiffusionConfig(**payload["diffusion_config"])
        metadata = PolicyMetadata(**payload["policy_metadata"])
        model = DiffusionTransformer(
            score_dim=diffusion_config.score_dim,
            state_dim=diffusion_config.state_dim,
            action_dim=diffusion_config.action_dim,
            obs_horizon=diffusion_config.obs_horizon,
            pred_horizon=diffusion_config.pred_horizon,
            model_dim=diffusion_config.model_dim,
            num_layers=diffusion_config.num_layers,
            num_heads=diffusion_config.num_heads,
            dropout=diffusion_config.dropout,
        )
        model.load_state_dict(payload["model"])
        model.eval()
        scheduler = GaussianDiffusionScheduler(
            steps=diffusion_config.diffusion_steps,
            beta_start=diffusion_config.beta_start,
            beta_end=diffusion_config.beta_end,
        )
        return cls(model=model, scheduler=scheduler, metadata=metadata, device=resolved_device)

    @torch.no_grad()
    def sample_action_chunk(
        self,
        *,
        score_window: np.ndarray | torch.Tensor,
        state_window: np.ndarray | torch.Tensor,
        warm_start: np.ndarray | torch.Tensor | None = None,
    ) -> np.ndarray:
        score_tensor = self._to_tensor(score_window, expected_last_dim=self.metadata.score_dim)
        state_tensor = self._to_tensor(state_window, expected_last_dim=self.metadata.state_dim)
        if score_tensor.ndim == 2:
            score_tensor = score_tensor.unsqueeze(0)
        if state_tensor.ndim == 2:
            state_tensor = state_tensor.unsqueeze(0)
        initial_sample = None
        if warm_start is not None:
            initial_sample = self._to_tensor(warm_start, expected_last_dim=self.metadata.action_dim)
            if initial_sample.ndim == 2:
                initial_sample = initial_sample.unsqueeze(0)
        sample = self.scheduler.sample(
            self.model,
            score_window=score_tensor.to(self.device),
            state_window=state_tensor.to(self.device),
            action_shape=(score_tensor.shape[0], self.metadata.pred_horizon, self.metadata.action_dim),
            device=self.device,
            initial_sample=initial_sample.to(self.device) if initial_sample is not None else None,
        )
        return sample.detach().cpu().numpy()

    def build_warm_start(self, previous_chunk: np.ndarray, executed_steps: int) -> np.ndarray:
        chunk = np.asarray(previous_chunk, dtype=np.float32)
        if chunk.ndim != 2:
            raise ValueError("previous_chunk must have shape [pred_horizon, action_dim]")
        executed = max(int(executed_steps), 1)
        tail = chunk[min(executed, chunk.shape[0]) :]
        if tail.size == 0:
            return np.zeros_like(chunk)
        pad = np.repeat(tail[-1:, :], chunk.shape[0] - tail.shape[0], axis=0)
        return np.concatenate([tail, pad], axis=0).astype(np.float32)

    def _to_tensor(self, array: np.ndarray | torch.Tensor, expected_last_dim: int) -> torch.Tensor:
        tensor = array if isinstance(array, torch.Tensor) else torch.as_tensor(array, dtype=torch.float32)
        if tensor.shape[-1] != expected_last_dim:
            raise ValueError(f"Expected last dimension {expected_last_dim}, got {tensor.shape[-1]}")
        return tensor.float()
