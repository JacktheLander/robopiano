from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from etude.training.reward_terms import REWARD_TERMS


RewardFn = Callable[..., float]


@dataclass(slots=True)
class RolloutObjectiveComposer:
    weights: dict[str, float]
    reward_fns: dict[str, RewardFn] | None = None

    def __post_init__(self) -> None:
        available = dict(REWARD_TERMS)
        if self.reward_fns:
            available.update(self.reward_fns)
        unknown = sorted(set(self.weights) - set(available))
        if unknown:
            raise KeyError(f"Unknown reward terms: {', '.join(unknown)}")
        self.reward_fns = available

    def compute(
        self,
        state: dict[str, Any],
        *,
        extras: dict[str, Any] | None = None,
        return_breakdown: bool = False,
    ) -> float | tuple[float, dict[str, float]]:
        extras = extras or {}
        breakdown: dict[str, float] = {}
        total = 0.0
        assert self.reward_fns is not None
        for name, weight in self.weights.items():
            raw_value = float(self.reward_fns[name](state, extras))
            weighted_value = float(weight) * raw_value
            breakdown[name] = weighted_value
            total += weighted_value
        if return_breakdown:
            return total, breakdown
        return total

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        reward_key: str = "reward",
        reward_fns: dict[str, RewardFn] | None = None,
    ) -> "RolloutObjectiveComposer":
        reward_section = config.get(reward_key, {})
        weights = {
            str(name): float(weight)
            for name, weight in reward_section.items()
            if isinstance(weight, (int, float))
        }
        return cls(weights=weights, reward_fns=reward_fns)
