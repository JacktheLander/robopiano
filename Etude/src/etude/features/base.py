from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np


class FeatureBlock(Protocol):
    def __call__(self, **kwargs: Any) -> np.ndarray:
        ...


@dataclass(slots=True)
class FeatureContext:
    values: dict[str, Any] = field(default_factory=dict)

    def with_updates(self, **kwargs: Any) -> "FeatureContext":
        merged = dict(self.values)
        merged.update(kwargs)
        return FeatureContext(values=merged)


FeatureBlockFactory = Callable[..., np.ndarray]
