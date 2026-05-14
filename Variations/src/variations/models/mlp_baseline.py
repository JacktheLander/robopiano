from __future__ import annotations

import torch
import torch.nn as nn


def _activation(name: str) -> nn.Module:
    value = str(name).lower()
    if value == "gelu":
        return nn.GELU()
    if value in {"silu", "swish"}:
        return nn.SiLU()
    if value == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation: {name}")


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, hidden_mult: int = 2, dropout: float = 0.0, activation: str = "gelu") -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * int(hidden_mult)),
            _activation(activation),
            nn.Dropout(float(dropout)),
            nn.Linear(dim * int(hidden_mult), dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class PressPoseMLP(nn.Module):
    """
    Direct supervised baseline:
    target_keys[88] -> joint_state[46]
    """

    def __init__(
        self,
        *,
        input_dim: int = 88,
        output_dim: int = 46,
        hidden_dims: list[int] | tuple[int, ...] = (256, 512, 512, 256),
        dropout: float = 0.0,
        activation: str = "gelu",
        hidden_mult: int = 2,
    ) -> None:
        super().__init__()
        dims = [int(input_dim), *[int(dim) for dim in hidden_dims]]
        layers: list[nn.Module] = []
        for idx in range(len(dims) - 1):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            layers.append(nn.LayerNorm(dims[idx + 1]))
            layers.append(_activation(activation))
            if float(dropout) > 0:
                layers.append(nn.Dropout(float(dropout)))
            if idx < len(dims) - 2 and dims[idx + 1] == dims[idx + 2]:
                layers.append(ResidualMLPBlock(dims[idx + 1], hidden_mult=hidden_mult, dropout=dropout, activation=activation))
        self.backbone = nn.Sequential(*layers)
        self.output = nn.Linear(dims[-1], int(output_dim))

    def forward(self, target_keys: torch.Tensor) -> torch.Tensor:
        return self.output(self.backbone(target_keys.float()))


def build_mlp_baseline(config: dict) -> PressPoseMLP:
    model_cfg = dict(config.get("model", {}))
    model_cfg["output_dim"] = int(model_cfg.get("output_dim", 46))
    return PressPoseMLP(**model_cfg)
