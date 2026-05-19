from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from etude.training.losses import behavior_cloning_loss


@dataclass
class TrainResult:
    train_loss: float


def train_bc_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str | torch.device = "cpu",
) -> TrainResult:
    model.train()
    device = torch.device(device)
    total = 0.0
    count = 0
    for batch in loader:
        features = batch["features"].to(device).float()
        actions = batch["actions"].to(device).float()
        pred = model(features.reshape(-1, features.shape[-1])).reshape_as(actions)
        loss = behavior_cloning_loss(pred, actions)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += float(loss.detach().cpu()) * features.shape[0]
        count += features.shape[0]
    return TrainResult(train_loss=total / max(count, 1))
