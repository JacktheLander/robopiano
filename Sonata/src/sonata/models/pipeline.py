from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from sonata.diffusion.dataset import load_diffusion_inputs, metadata_to_planner
from sonata.diffusion.diffusion import GaussianDiffusion1D
from sonata.diffusion.model import ConditionalTemporalDenoiser
from sonata.transformer.model import FACTORED_PLANNER_ARCHITECTURE, build_planner_from_config, planner_output_dim_from_config
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
            family_mapping_mode=str(self.config.get("family_mapping_mode", "heuristic_stats")),
            continuous_param_names=self.config.get("continuous_param_names"),
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
            return base + planner_output_dim_from_config(planner_config)
        return base + int(self.config["primitive_embedding_dim"])

    def _planner_config_from_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("planner_config"):
            return dict(payload["planner_config"])
        planner_checkpoint = self.config.get("planner_checkpoint")
        if planner_checkpoint:
            planner_payload = load_checkpoint(Path(planner_checkpoint).resolve(), map_location=self.device)
            return dict(planner_payload.get("config", {}))
        return {
            "d_model": int(self.config["planner_embedding_dim"]),
            "nhead": int(self.config["planner_nhead"]),
            "num_layers": int(self.config["planner_layers"]),
            "dim_feedforward": int(self.config["planner_ffn"]),
            "dropout": float(self.config["planner_dropout"]),
            "context_length": int(self.config["context_length"]),
            "plan_embedding_dim": int(self.config["planner_embedding_dim"]),
        }

    def _load_planner(self, payload: dict[str, Any]):
        if payload.get("planner") is None:
            return None
        planner_config = self._planner_config_from_payload(payload)
        planner_metadata = payload.get("planner_metadata")
        self._validate_planner_metadata(planner_metadata)
        architecture = planner_config.get("planner_architecture", payload.get("planner_architecture", FACTORED_PLANNER_ARCHITECTURE))
        if architecture != FACTORED_PLANNER_ARCHITECTURE:
            raise ValueError(f"Unsupported planner architecture in diffusion checkpoint: {architecture!r}")
        planner = build_planner_from_config(metadata_to_planner(self.metadata), planner_config).to(self.device)
        planner.load_state_dict(payload["planner"])
        planner.eval()
        return planner

    def _validate_planner_metadata(self, payload: dict[str, Any] | None) -> None:
        if not payload:
            return
        if int(payload.get("num_primitives", self.metadata.num_primitives)) != int(self.metadata.num_primitives):
            raise ValueError("Diffusion checkpoint planner metadata does not match the current primitive vocabulary.")
        if [int(item) for item in payload.get("primitive_to_family", self.metadata.primitive_to_family)] != [int(item) for item in self.metadata.primitive_to_family]:
            raise ValueError(
                "Diffusion checkpoint planner metadata uses a different primitive-family mapping than the current primitive_root."
            )

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
