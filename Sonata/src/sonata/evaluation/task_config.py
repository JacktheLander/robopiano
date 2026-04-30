from __future__ import annotations


def build_rollout_task_kwargs(*, control_timestep: float, expected_action_dim: int) -> dict[str, float | int]:
    del expected_action_dim
    return {
        "control_timestep": float(control_timestep),
        "n_steps_lookahead": 1,
    }


def validate_rollout_action_dim(*, actual_action_dim: int, expected_action_dim: int, environment_name: str) -> None:
    actual = int(actual_action_dim)
    expected = int(expected_action_dim)
    if actual < expected:
        raise ValueError(
            "RoboPianist environment "
            f"`{environment_name}` exposes action_dim={actual}, which is smaller than the "
            f"primitive prior action_dim={expected}."
        )
