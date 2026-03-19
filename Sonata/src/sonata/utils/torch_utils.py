from __future__ import annotations

from contextlib import nullcontext
from typing import Any


def count_parameters(model: Any) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def move_to_device(batch: Any, device: str):
    try:
        import torch
    except ImportError:
        return batch

    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: move_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, (list, tuple)):
        values = [move_to_device(value, device) for value in batch]
        return type(batch)(values)
    return batch


def autocast_context(device: str, enabled: bool):
    try:
        import torch
    except ImportError:
        return nullcontext()
    if not enabled or not device.startswith("cuda"):
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16)
