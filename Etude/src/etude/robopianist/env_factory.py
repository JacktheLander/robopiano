from __future__ import annotations

from typing import Any


def make_robopianist_env(task: str = "RoboPianist-repertoire-150-v0", **kwargs: Any) -> Any:
    """Create a RoboPianist environment when optional simulator deps are installed."""
    try:
        import robopianist  # noqa: F401
        from robopianist import suite
    except ImportError as exc:
        raise ImportError(
            "RoboPianist is not installed. Install Etude with simulator dependencies and "
            "the RoboPianist package before using closed-loop rollout."
        ) from exc

    return suite.load(task, **kwargs)
