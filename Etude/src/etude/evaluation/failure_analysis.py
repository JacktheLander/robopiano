from __future__ import annotations

from collections import Counter


def summarize_failure_reasons(reasons: list[str]) -> dict[str, int]:
    return dict(Counter(reasons))
