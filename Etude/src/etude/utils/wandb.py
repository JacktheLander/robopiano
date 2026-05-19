from __future__ import annotations

from typing import Any


def maybe_init_wandb(enabled: bool, **kwargs: Any) -> Any:
    if not enabled:
        return None
    import wandb

    return wandb.init(**kwargs)
