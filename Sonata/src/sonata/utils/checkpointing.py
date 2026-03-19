from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> Path:
    resolved = Path(path).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, resolved)
    return resolved


def find_latest_checkpoint(checkpoint_dir: str | Path, pattern: str = "*.pt") -> Path | None:
    candidates = sorted(Path(checkpoint_dir).resolve().glob(pattern))
    return candidates[-1] if candidates else None


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(Path(path).resolve(), map_location=map_location, weights_only=False)
