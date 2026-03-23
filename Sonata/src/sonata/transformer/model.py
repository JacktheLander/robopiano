from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from sonata.transformer.dataset import PlannerMetadata, family_mask_tensor

FACTORED_PLANNER_ARCHITECTURE = "sonata_factored_goal_conditioned_v1"


def planner_output_dim_from_config(config: dict[str, Any]) -> int:
    if "plan_embedding_dim" in config:
        return int(config["plan_embedding_dim"])
    if "planner_embedding_dim" in config:
        return int(config["planner_embedding_dim"])
    return int(config["d_model"])


class PrimitivePlannerTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_primitives: int,
        num_duration_buckets: int,
        num_dynamics_buckets: int,
        num_families: int,
        primitive_to_family: list[int],
        history_context_dim: int,
        goal_context_dim: int,
        continuous_param_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        max_length: int,
        plan_embedding_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.max_length = int(max_length)
        self.num_primitives = int(num_primitives)
        self.num_families = int(num_families)
        self.num_duration_buckets = int(num_duration_buckets)
        self.num_dynamics_buckets = int(num_dynamics_buckets)
        self.continuous_param_dim = int(continuous_param_dim)
        self.plan_embedding_dim = int(plan_embedding_dim or d_model)

        self.primitive_embed = nn.Embedding(num_primitives + 1, d_model)
        self.family_embed = nn.Embedding(num_families + 1, d_model)
        self.duration_embed = nn.Embedding(num_duration_buckets + 1, d_model)
        self.dynamics_embed = nn.Embedding(num_dynamics_buckets + 1, d_model)
        self.position_embed = nn.Embedding(max_length, d_model)
        self.history_context_proj = nn.Linear(history_context_dim, d_model)
        self.goal_context_proj = nn.Sequential(
            nn.Linear(goal_context_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.summary_fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )
        self.family_head = nn.Linear(d_model, num_families)
        self.family_intent_embed = nn.Embedding(num_families, d_model)
        self.primitive_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )
        self.primitive_head = nn.Linear(d_model, num_primitives)
        self.primitive_intent_embed = nn.Embedding(num_primitives, d_model)
        self.duration_head = nn.Linear(d_model, num_duration_buckets)
        self.duration_intent_embed = nn.Embedding(num_duration_buckets, d_model)
        self.dynamics_head = nn.Linear(d_model, num_dynamics_buckets)
        self.dynamics_intent_embed = nn.Embedding(num_dynamics_buckets, d_model)
        self.continuous_param_head = (
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, continuous_param_dim),
            )
            if continuous_param_dim > 0
            else None
        )
        self.plan_projection = nn.Sequential(
            nn.Linear(d_model * 6, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, self.plan_embedding_dim),
        )

        mask_metadata = PlannerMetadata(
            num_primitives=num_primitives,
            num_duration_buckets=num_duration_buckets,
            num_dynamics_buckets=num_dynamics_buckets,
            num_families=num_families,
            score_dim=goal_context_dim,
            history_dim=history_context_dim,
            pad_primitive=num_primitives,
            pad_duration=num_duration_buckets,
            pad_dynamics=num_dynamics_buckets,
            pad_family=num_families,
            primitive_ids=[str(index) for index in range(num_primitives)],
            primitive_family_names=[str(index) for index in range(num_families)],
            primitive_to_family=[int(index) for index in primitive_to_family],
            family_mapping_mode="runtime",
            continuous_param_names=[],
            continuous_param_mean=[],
            continuous_param_std=[],
            goal_context_features=[],
            history_context_features=[],
        )
        self.register_buffer("family_primitive_mask", family_mask_tensor(mask_metadata), persistent=False)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        primitive = batch["primitive_history"]
        family = batch["family_history"]
        duration = batch["duration_history"]
        dynamics = batch["dynamics_history"]
        history_context = batch.get("history_context", batch.get("score_history"))
        goal_context = batch.get("planner_context", batch.get("score_context"))
        attention_mask = batch["attention_mask"]
        if history_context is None or goal_context is None:
            raise ValueError("Planner batch must include history_context/score_history and planner_context/score_context.")
        batch_size, sequence_length = primitive.shape
        if sequence_length > self.max_length:
            raise ValueError(
                f"Planner history length {sequence_length} exceeds configured max_length={self.max_length}."
            )
        positions = torch.arange(sequence_length, device=primitive.device).unsqueeze(0).expand(batch_size, sequence_length)
        hidden = (
            self.primitive_embed(primitive)
            + self.family_embed(family)
            + self.duration_embed(duration)
            + self.dynamics_embed(dynamics)
            + self.position_embed(positions)
            + self.history_context_proj(history_context)
        )
        hidden = hidden * math.sqrt(self.d_model)
        causal_mask = torch.triu(torch.ones(sequence_length, sequence_length, device=primitive.device), diagonal=1).bool()
        key_padding_mask = attention_mask <= 0
        encoded = self.encoder(hidden, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        encoded = self.norm(encoded)
        last_index = attention_mask.sum(dim=1).long().clamp(min=1) - 1
        final_hidden = encoded[torch.arange(batch_size, device=primitive.device), last_index]
        pooled_hidden = masked_mean(encoded, attention_mask)
        goal_hidden = self.goal_context_proj(goal_context)
        planner_state = self.summary_fusion(torch.cat([final_hidden, pooled_hidden, goal_hidden], dim=-1))

        family_logits = self.family_head(planner_state)
        family_probs = torch.softmax(family_logits, dim=-1)
        family_intent = family_probs @ self.family_intent_embed.weight

        primitive_state = self.primitive_fusion(torch.cat([planner_state, family_intent], dim=-1))
        primitive_logits = self.primitive_head(primitive_state)
        soft_family_mask = torch.matmul(family_probs, self.family_primitive_mask.float())
        primitive_logits = primitive_logits + soft_family_mask.clamp(min=1e-6).log()

        duration_logits = self.duration_head(planner_state)
        dynamics_logits = self.dynamics_head(planner_state)
        continuous_params = (
            self.continuous_param_head(planner_state)
            if self.continuous_param_head is not None
            else torch.zeros((batch_size, 0), device=primitive.device, dtype=planner_state.dtype)
        )

        primitive_probs = torch.softmax(primitive_logits, dim=-1)
        primitive_intent = primitive_probs @ self.primitive_intent_embed.weight
        duration_intent = torch.softmax(duration_logits, dim=-1) @ self.duration_intent_embed.weight
        dynamics_intent = torch.softmax(dynamics_logits, dim=-1) @ self.dynamics_intent_embed.weight
        plan_embedding = self.plan_projection(
            torch.cat(
                [
                    planner_state,
                    goal_hidden,
                    family_intent,
                    primitive_intent,
                    duration_intent,
                    dynamics_intent,
                ],
                dim=-1,
            )
        )

        return {
            "family_logits": family_logits,
            "primitive_logits": primitive_logits,
            "duration_logits": duration_logits,
            "dynamics_logits": dynamics_logits,
            "continuous_param_pred": continuous_params,
            "planner_state": planner_state,
            "goal_context_hidden": goal_hidden,
            "plan_embedding": plan_embedding,
        }


class TransformerActionRegressor(nn.Module):
    def __init__(
        self,
        *,
        planner: PrimitivePlannerTransformer,
        action_horizon: int,
        action_dim: int,
    ) -> None:
        super().__init__()
        self.planner = planner
        hidden_dim = planner.plan_embedding_dim
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_horizon * action_dim),
        )
        self.action_horizon = action_horizon
        self.action_dim = action_dim

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        outputs = self.planner(batch)
        action = self.action_head(outputs["plan_embedding"]).reshape(-1, self.action_horizon, self.action_dim)
        outputs["action_pred"] = action
        return outputs


def build_planner_from_config(metadata: PlannerMetadata, config: dict[str, Any]) -> PrimitivePlannerTransformer:
    return PrimitivePlannerTransformer(
        num_primitives=metadata.num_primitives,
        num_duration_buckets=metadata.num_duration_buckets,
        num_dynamics_buckets=metadata.num_dynamics_buckets,
        num_families=metadata.num_families,
        primitive_to_family=metadata.primitive_to_family,
        history_context_dim=metadata.history_dim,
        goal_context_dim=metadata.score_dim,
        continuous_param_dim=metadata.continuous_param_dim,
        d_model=int(config["d_model"]),
        nhead=int(config["nhead"]),
        num_layers=int(config["num_layers"]),
        dim_feedforward=int(config["dim_feedforward"]),
        dropout=float(config["dropout"]),
        max_length=int(config["context_length"]),
        plan_embedding_dim=planner_output_dim_from_config(config),
    )


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(values.dtype)
    total = (values * weights).sum(dim=1)
    denom = weights.sum(dim=1).clamp(min=1.0)
    return total / denom
