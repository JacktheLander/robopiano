from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn

NUM_PIANO_KEYS = 88


def resample_binary_sequence(array: np.ndarray, steps: int, *, threshold: float = 0.5) -> np.ndarray:
    """Resample a binary roll with recall-friendly window max pooling."""
    roll = np.asarray(array, dtype=np.float32)
    if roll.ndim == 1:
        roll = roll[:, None]
    if roll.size == 0:
        return np.zeros((steps, roll.shape[1] if roll.ndim == 2 else 0), dtype=np.float32)
    active = roll > threshold
    if active.shape[0] == steps:
        return active.astype(np.float32)
    edges = np.linspace(0.0, float(active.shape[0]), num=steps + 1, dtype=np.float32)
    output = np.zeros((steps, active.shape[1]), dtype=np.float32)
    for index in range(steps):
        start = int(np.floor(edges[index]))
        end = int(np.ceil(edges[index + 1]))
        start = min(max(start, 0), active.shape[0] - 1)
        end = min(max(end, start + 1), active.shape[0])
        output[index] = active[start:end].any(axis=0).astype(np.float32)
    return output


def build_note_state_targets(
    contact_roll: np.ndarray | None,
    *,
    action_horizon: int,
    previous_frame: np.ndarray | None = None,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive note, hold, and sustain targets from an episode contact roll slice."""
    if contact_roll is None:
        return (
            np.zeros((action_horizon, NUM_PIANO_KEYS), dtype=np.float32),
            np.zeros((action_horizon, NUM_PIANO_KEYS), dtype=np.float32),
            np.zeros((action_horizon, 1), dtype=np.float32),
        )

    roll = np.asarray(contact_roll, dtype=np.float32)
    if roll.ndim == 1:
        roll = roll[None, :]
    note_roll = roll[:, :NUM_PIANO_KEYS]
    note_active = note_roll > threshold
    if previous_frame is None:
        previous_active = np.zeros((1, NUM_PIANO_KEYS), dtype=bool)
    else:
        previous_array = np.asarray(previous_frame, dtype=np.float32).reshape(1, -1)
        previous_active = previous_array[:, :NUM_PIANO_KEYS] > threshold
    preceding = np.concatenate([previous_active, note_active[:-1]], axis=0)
    hold_active = note_active & preceding
    sustain_roll = roll[:, NUM_PIANO_KEYS : NUM_PIANO_KEYS + 1] if roll.shape[1] > NUM_PIANO_KEYS else np.zeros((roll.shape[0], 1), dtype=np.float32)

    return (
        resample_binary_sequence(note_active.astype(np.float32), action_horizon, threshold=threshold),
        resample_binary_sequence(hold_active.astype(np.float32), action_horizon, threshold=threshold),
        resample_binary_sequence(sustain_roll, action_horizon, threshold=threshold),
    )


def action_surrogate_features_np(actions: np.ndarray) -> np.ndarray:
    roll = np.asarray(actions, dtype=np.float32)
    previous = np.concatenate([roll[:1], roll[:-1]], axis=0)
    delta = roll - previous
    return np.concatenate([roll, delta], axis=-1)


def action_surrogate_features(actions: torch.Tensor) -> torch.Tensor:
    previous = torch.cat([actions[:, :1], actions[:, :-1]], dim=1)
    delta = actions - previous
    return torch.cat([actions, delta], dim=-1)


@dataclass(slots=True)
class LinearAccumulator:
    feature_dim: int
    output_dim: int
    xtx: np.ndarray = field(init=False)
    xty: np.ndarray = field(init=False)
    num_rows: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.xtx = np.zeros((self.feature_dim, self.feature_dim), dtype=np.float64)
        self.xty = np.zeros((self.feature_dim, self.output_dim), dtype=np.float64)
        self.num_rows = 0

    def add(self, features: np.ndarray, targets: np.ndarray) -> None:
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        self.xtx += x.T @ x
        self.xty += x.T @ y
        self.num_rows += int(x.shape[0])

    def solve(self, ridge_lambda: float) -> tuple[np.ndarray, np.ndarray]:
        regularized = self.xtx + float(ridge_lambda) * np.eye(self.feature_dim, dtype=np.float64)
        solution = np.linalg.solve(regularized, self.xty).astype(np.float32)
        weight = solution[:-1].T.copy()
        bias = solution[-1].copy()
        return weight, bias


class LinearActionNoteSurrogate(nn.Module):
    """Fixed lightweight surrogate from action trajectories to note-state logits."""

    def __init__(self, *, action_dim: int, has_sustain_head: bool = True) -> None:
        super().__init__()
        feature_dim = int(action_dim) * 2
        self.action_dim = int(action_dim)
        self.feature_dim = feature_dim
        self.note_head = nn.Linear(feature_dim, NUM_PIANO_KEYS)
        self.hold_head = nn.Linear(feature_dim, NUM_PIANO_KEYS)
        self.sustain_head = nn.Linear(feature_dim, 1) if has_sustain_head else None
        self._freeze_parameters()

    def _freeze_parameters(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False

    @classmethod
    def from_state(
        cls,
        *,
        action_dim: int,
        note_weight: np.ndarray,
        note_bias: np.ndarray,
        hold_weight: np.ndarray,
        hold_bias: np.ndarray,
        sustain_weight: np.ndarray | None,
        sustain_bias: np.ndarray | None,
    ) -> "LinearActionNoteSurrogate":
        module = cls(action_dim=action_dim, has_sustain_head=sustain_weight is not None and sustain_bias is not None)
        _load_linear(module.note_head, note_weight, note_bias)
        _load_linear(module.hold_head, hold_weight, hold_bias)
        if module.sustain_head is not None and sustain_weight is not None and sustain_bias is not None:
            _load_linear(module.sustain_head, sustain_weight, sustain_bias)
        module._freeze_parameters()
        return module

    def forward(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        features = action_surrogate_features(actions)
        outputs = {
            "note_logits": self.note_head(features),
            "hold_logits": self.hold_head(features),
        }
        if self.sustain_head is not None:
            outputs["sustain_logits"] = self.sustain_head(features)
        else:
            outputs["sustain_logits"] = actions.new_zeros(actions.shape[0], actions.shape[1], 1)
        return outputs


def fit_linear_note_surrogate(
    dataset: Any,
    *,
    action_dim: int,
    ridge_lambda: float = 1e-2,
) -> tuple[LinearActionNoteSurrogate, dict[str, float]]:
    feature_dim = int(action_dim) * 2 + 1
    note_accumulator = LinearAccumulator(feature_dim=feature_dim, output_dim=NUM_PIANO_KEYS)
    hold_accumulator = LinearAccumulator(feature_dim=feature_dim, output_dim=NUM_PIANO_KEYS)
    sustain_accumulator = LinearAccumulator(feature_dim=feature_dim, output_dim=1)

    for sample in dataset:
        features = action_surrogate_features_np(np.asarray(sample["action_target"], dtype=np.float32))
        features = np.concatenate([features, np.ones((features.shape[0], 1), dtype=np.float32)], axis=-1)
        note_target = np.where(np.asarray(sample["note_target"], dtype=np.float32) > 0.5, 1.0, -1.0)
        hold_target = np.where(np.asarray(sample["hold_target"], dtype=np.float32) > 0.5, 1.0, -1.0)
        sustain_target = np.where(np.asarray(sample["sustain_target"], dtype=np.float32) > 0.5, 1.0, -1.0)
        note_accumulator.add(features, note_target)
        hold_accumulator.add(features, hold_target)
        sustain_accumulator.add(features, sustain_target)

    if note_accumulator.num_rows == 0:
        raise ValueError("Diffusion train split did not contain any note-supervised samples to fit the action surrogate.")

    note_weight, note_bias = note_accumulator.solve(ridge_lambda=ridge_lambda)
    hold_weight, hold_bias = hold_accumulator.solve(ridge_lambda=ridge_lambda)
    sustain_weight, sustain_bias = sustain_accumulator.solve(ridge_lambda=ridge_lambda)
    module = LinearActionNoteSurrogate.from_state(
        action_dim=action_dim,
        note_weight=note_weight,
        note_bias=note_bias,
        hold_weight=hold_weight,
        hold_bias=hold_bias,
        sustain_weight=sustain_weight,
        sustain_bias=sustain_bias,
    )
    fit_stats = {
        "surrogate_rows": float(note_accumulator.num_rows),
        "surrogate_ridge_lambda": float(ridge_lambda),
    }
    return module, fit_stats


def _load_linear(layer: nn.Linear, weight: np.ndarray, bias: np.ndarray) -> None:
    layer.weight.data.copy_(torch.from_numpy(np.asarray(weight, dtype=np.float32)))
    layer.bias.data.copy_(torch.from_numpy(np.asarray(bias, dtype=np.float32)))
