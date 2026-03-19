from __future__ import annotations

import math

import torch
import torch.nn as nn


class PrimitivePlannerTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_primitives: int,
        num_duration_buckets: int,
        num_dynamics_buckets: int,
        score_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        max_length: int,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_length = max_length
        self.primitive_embed = nn.Embedding(num_primitives + 1, d_model)
        self.duration_embed = nn.Embedding(num_duration_buckets + 1, d_model)
        self.dynamics_embed = nn.Embedding(num_dynamics_buckets + 1, d_model)
        self.position_embed = nn.Embedding(max_length, d_model)
        self.score_proj = nn.Linear(score_dim, d_model)
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
        self.primitive_head = nn.Linear(d_model, num_primitives)
        self.duration_head = nn.Linear(d_model, num_duration_buckets)
        self.dynamics_head = nn.Linear(d_model, num_dynamics_buckets)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        primitive = batch["primitive_history"]
        duration = batch["duration_history"]
        dynamics = batch["dynamics_history"]
        score = batch["score_history"]
        attention_mask = batch["attention_mask"]
        batch_size, sequence_length = primitive.shape
        positions = torch.arange(sequence_length, device=primitive.device).unsqueeze(0).expand(batch_size, sequence_length)
        hidden = (
            self.primitive_embed(primitive)
            + self.duration_embed(duration)
            + self.dynamics_embed(dynamics)
            + self.position_embed(positions)
            + self.score_proj(score)
        )
        hidden = hidden * math.sqrt(self.d_model)
        causal_mask = torch.triu(torch.ones(sequence_length, sequence_length, device=primitive.device), diagonal=1).bool()
        key_padding_mask = attention_mask <= 0
        encoded = self.encoder(hidden, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        encoded = self.norm(encoded)
        last_index = attention_mask.sum(dim=1).long().clamp(min=1) - 1
        final_hidden = encoded[torch.arange(batch_size, device=primitive.device), last_index]
        return {
            "primitive_logits": self.primitive_head(final_hidden),
            "duration_logits": self.duration_head(final_hidden),
            "dynamics_logits": self.dynamics_head(final_hidden),
            "plan_embedding": final_hidden,
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
        hidden_dim = planner.d_model
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
