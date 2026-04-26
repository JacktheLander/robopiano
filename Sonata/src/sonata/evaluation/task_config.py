from __future__ import annotations

from typing import Any


def build_rollout_task_kwargs(*, control_timestep: float, expected_action_dim: int) -> dict[str, Any]:
    task_kwargs: dict[str, Any] = {
        "control_timestep": float(control_timestep),
        "n_steps_lookahead": 1,
    }
    inferred_reduced = infer_reduced_action_space(expected_action_dim)
    if inferred_reduced is not None:
        task_kwargs["reduced_action_space"] = inferred_reduced
    return task_kwargs


def infer_reduced_action_space(expected_action_dim: int) -> bool | None:
    if int(expected_action_dim) == 39:
        return True
    if int(expected_action_dim) == 45:
        return False
    return None


def validate_rollout_action_dim(*, actual_action_dim: int, expected_action_dim: int, environment_name: str) -> None:
    if int(actual_action_dim) != int(expected_action_dim):
        raise ValueError(
            f"Rollout environment `{environment_name}` expects action_dim={actual_action_dim}, "
            f"but the Sonata artifacts expect action_dim={expected_action_dim}. "
            "This usually means the RoboPianist task was created with the wrong action-space variant."
        )
