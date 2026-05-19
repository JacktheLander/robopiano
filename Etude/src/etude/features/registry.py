from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from etude.utils.import_utils import load_symbol


FeatureBuilder = Callable[..., Any]


def resolve_feature_block(block: str | FeatureBuilder) -> FeatureBuilder:
    if callable(block):
        return block
    loaded = load_symbol(block)
    if not callable(loaded):
        raise TypeError(f"Resolved object is not callable: {block}")
    return loaded


@dataclass(slots=True)
class FeatureRegistry:
    _registered: dict[str, FeatureBuilder] = field(default_factory=dict)

    def register(self, name: str, builder: FeatureBuilder) -> None:
        if not callable(builder):
            raise TypeError(f"builder must be callable: {name}")
        self._registered[name] = builder

    def get(self, name_or_path: str) -> FeatureBuilder:
        if name_or_path in self._registered:
            return self._registered[name_or_path]
        resolved = resolve_feature_block(name_or_path)
        self._registered[name_or_path] = resolved
        return resolved

    def build_many(self, blocks: list[str | FeatureBuilder], **kwargs: Any) -> list[Any]:
        return [resolve_feature_block(block)(**kwargs) for block in blocks]


registry = FeatureRegistry()
