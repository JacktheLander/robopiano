from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from sonata.diffusion.dataset import load_diffusion_inputs
from sonata.diffusion.diffusion import GaussianDiffusion1D
from sonata.diffusion.model import ConditionalTemporalDenoiser
from sonata.transformer.model import PrimitivePlannerTransformer
from sonata.utils.checkpointing import load_checkpoint


class Sonata3Pipeline:
    def __init__(self, primitive_root: Path, diffusion_checkpoint: Path, device: str = "cpu") -> None:
        self.primitive_root = primitive_root.resolve()
        self.diffusion_checkpoint = diffusion_checkpoint.resolve()
        self.device = torch.device(device)
        payload = load_checkpoint(self.diffusion_checkpoint, map_location=self.device)
        self.config = payload["config"]
        _, self.metadata, self.prior_lookup = load_diffusion_inputs(
            self.primitive_root,
            int(self.config["action_horizon"]),
            int(self.config["state_context_steps"]),
        )
        self.model = ConditionalTemporalDenoiser(
            action_dim=self.metadata.action_dim,
            prior_dim=self.metadata.action_dim,
            global_cond_dim=self._global_condition_dim(payload),
            model_dim=int(self.config["model_dim"]),
            num_blocks=int(self.config["num_blocks"]),
        ).to(self.device)
        self.model.load_state_dict(payload["model"])
        self.model.eval()
        self.primitive_embed = torch.nn.Embedding(self.metadata.num_primitives + 1, int(self.config["primitive_embedding_dim"])).to(self.device)
        self.primitive_embed.load_state_dict(payload["primitive_embed"])
        self.primitive_embed.eval()
        self.planner = self._load_planner(payload)
        self.diffusion = GaussianDiffusion1D(
            timesteps=int(self.config["diffusion_steps"]),
            beta_start=float(self.config["beta_start"]),
            beta_end=float(self.config["beta_end"]),
            device=self.device,
        )

    def _global_condition_dim(self, payload: dict[str, Any]) -> int:
        base = self.metadata.score_dim + self.metadata.state_dim + 3
        if payload.get("planner") is not None and str(self.config["variant"]) not in {"diffusion_only", "gmr_only"}:
            planner_config = self._planner_config_from_payload(payload)
            return base + int(planner_config["d_model"])
        return base + int(self.config["primitive_embedding_dim"])

    def _planner_config_from_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        planner_checkpoint = self.config.get("planner_checkpoint")
        if planner_checkpoint:
            planner_payload = load_checkpoint(Path(planner_checkpoint).resolve(), map_location=self.device)
            return planner_payload["config"]
        return {
            "d_model": int(self.config["planner_embedding_dim"]),
            "nhead": int(self.config["planner_nhead"]),
            "num_layers": int(self.config["planner_layers"]),
            "dim_feedforward": int(self.config["planner_ffn"]),
            "dropout": float(self.config["planner_dropout"]),
        }

    def _load_planner(self, payload: dict[str, Any]):
        if payload.get("planner") is None:
            return None
        planner_config = self._planner_config_from_payload(payload)
        planner = PrimitivePlannerTransformer(
            num_primitives=self.metadata.num_primitives,
            num_duration_buckets=self.metadata.num_duration_buckets,
            num_dynamics_buckets=self.metadata.num_dynamics_buckets,
            score_dim=self.metadata.score_dim,
            d_model=int(planner_config["d_model"]),
            nhead=int(planner_config["nhead"]),
            num_layers=int(planner_config["num_layers"]),
            dim_feedforward=int(planner_config["dim_feedforward"]),
            dropout=float(planner_config["dropout"]),
            max_length=int(self.config["context_length"]),
        ).to(self.device)
        planner.load_state_dict(payload["planner"], strict=False)
        planner.eval()
        return planner

    def _condition_vector(self, batch: dict[str, torch.Tensor], variant: str) -> torch.Tensor:
        score = batch["score_context"]
        state = batch["state_context"]
        scalar = torch.stack(
            [
                batch["duration_bucket"].float(),
                batch["dynamics_bucket"].float(),
                batch["primitive_index"].float(),
            ],
            dim=-1,
        )
        if self.planner is not None and variant not in {"diffusion_only", "gmr_only"}:
            with torch.no_grad():
                plan = self.planner(batch)["plan_embedding"]
            return torch.cat([score, state, scalar, plan], dim=-1)
        primitive = self.primitive_embed(batch["primitive_index"])
        return torch.cat([score, state, scalar, primitive], dim=-1)

    @torch.no_grad()
    def predict_batch(self, batch: dict[str, torch.Tensor], variant: str | None = None) -> torch.Tensor:
        local_variant = variant or str(self.config["variant"])
        batch = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in batch.items()}
        prior = batch["gmr_prior"]
        if local_variant == "gmr_only":
            return prior
        if local_variant in {"diffusion_only", "planner_no_prior"}:
            prior = torch.zeros_like(prior)
        condition = self._condition_vector(batch, variant=local_variant)
        return self.diffusion.sample(
            self.model,
            shape=tuple(batch["action_target"].shape),
            prior=prior,
            condition=condition,
        )
