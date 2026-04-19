from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robodiffusion.data.loading import load_manifest
from robodiffusion.data.score import context_to_feature_vector
from robodiffusion.model.policy import RoboDiffusionPolicy
from robodiffusion.utils.io import write_json, write_table
from robodiffusion.utils.robopianist import ensure_local_robopianist_on_path, format_robopianist_import_error

LOGGER = logging.getLogger(__name__)


def evaluate_policy_rollout(
    *,
    checkpoint_path: str | Path,
    output_root: str | Path,
    sources: list[dict[str, Any]],
    max_steps: int = 512,
    robopianist_root: str | Path | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    logger = logger or LOGGER
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    ensure_local_robopianist_on_path(robopianist_root)
    try:
        from robopianist import suite
    except Exception as exc:  # pragma: no cover
        payload = {"available": False, "error": format_robopianist_import_error(exc, robopianist_root)}
        write_json(payload, output_root / "rollout.json")
        return payload

    policy = RoboDiffusionPolicy.from_checkpoint(checkpoint_path, device="cpu")
    observation_spec = dict(policy.metadata.observation_spec or {})
    results: list[dict[str, Any]] = []

    for source_index, source in enumerate(sources):
        env = None
        try:
            env = suite.load(
                environment_name=str(source.get("environment_name", "RoboPianist-debug-TwinkleTwinkleLittleStar-v0")),
                midi_file=Path(source["midi_file"]).resolve() if source.get("midi_file") else None,
                seed=0,
                task_kwargs={
                    "control_timestep": float(source.get("control_timestep", 0.05)),
                    "n_steps_lookahead": int(source.get("n_steps_lookahead", 1)),
                },
            )
            controller = RecedingHorizonController(policy, observation_spec=observation_spec)
            timestep = env.reset()
            controller.reset()
            total_reward = 0.0
            actions_executed = 0
            action_dim = int(env.action_spec().shape[0])
            while not timestep.last() and actions_executed < max_steps:
                action = controller.action(timestep.observation)
                control = np.zeros((action_dim,), dtype=np.float32)
                control[: min(action_dim, action.shape[0])] = action[: min(action_dim, control.shape[0])]
                timestep = env.step(control)
                total_reward += float(timestep.reward or 0.0)
                actions_executed += 1
            results.append(
                {
                    "source_index": source_index,
                    "environment_name": str(source.get("environment_name", "")),
                    "midi_file": str(source.get("midi_file") or ""),
                    "reward": float(total_reward),
                    "actions_executed": int(actions_executed),
                    "terminated": bool(timestep.last()),
                }
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Rollout failed for source %s: %s", source, exc)
            results.append(
                {
                    "source_index": source_index,
                    "environment_name": str(source.get("environment_name", "")),
                    "midi_file": str(source.get("midi_file") or ""),
                    "error": str(exc),
                }
            )
        finally:
            try:
                if env is not None:
                    env.close()
            except Exception:
                pass

    summary = summarize_results(results)
    payload = {"available": True, "summary": summary, "episodes": results}
    write_json(payload, output_root / "rollout.json")
    write_table(pd.DataFrame(results), output_root / "rollout_results")
    return payload


def load_rollout_sources(config: dict[str, Any]) -> list[dict[str, Any]]:
    if config.get("midi_files"):
        return [
            {
                "environment_name": str(config.get("environment_name", "RoboPianist-debug-TwinkleTwinkleLittleStar-v0")),
                "midi_file": str(path),
                "control_timestep": float(config.get("control_timestep", 0.05)),
                "n_steps_lookahead": int(config.get("n_steps_lookahead", 1)),
            }
            for path in config["midi_files"]
        ]
    if config.get("environment_names"):
        return [
            {
                "environment_name": str(name),
                "midi_file": None,
                "control_timestep": float(config.get("control_timestep", 0.05)),
                "n_steps_lookahead": int(config.get("n_steps_lookahead", 1)),
            }
            for name in config["environment_names"]
        ]
    manifest_path = config.get("data_manifest_path")
    if manifest_path:
        manifest_df = load_manifest(manifest_path)
        limit = int(config.get("limit_episodes", 2))
        rows = manifest_df[manifest_df["note_path"].astype(str).str.len() > 0].head(limit)
        return [
            {
                "environment_name": str(config.get("environment_name", "RoboPianist-debug-TwinkleTwinkleLittleStar-v0")),
                "midi_file": str(row.note_path),
                "control_timestep": float(getattr(row, "control_timestep", config.get("control_timestep", 0.05))),
                "n_steps_lookahead": int(config.get("n_steps_lookahead", 1)),
            }
            for row in rows.itertuples(index=False)
        ]
    return [
        {
            "environment_name": str(config.get("environment_name", "RoboPianist-debug-TwinkleTwinkleLittleStar-v0")),
            "midi_file": None,
            "control_timestep": float(config.get("control_timestep", 0.05)),
            "n_steps_lookahead": int(config.get("n_steps_lookahead", 1)),
        }
    ]


class RecedingHorizonController:
    def __init__(self, policy: RoboDiffusionPolicy, observation_spec: dict[str, Any] | None = None) -> None:
        self.policy = policy
        self.obs_horizon = int(policy.metadata.obs_horizon)
        self.execute_horizon = int(policy.metadata.action_execute_horizon)
        self.observation_spec = dict(observation_spec or {})
        self.score_history: deque[np.ndarray] = deque(maxlen=self.obs_horizon)
        self.state_history: deque[np.ndarray] = deque(maxlen=self.obs_horizon)
        self.pending_actions: list[np.ndarray] = []
        self.previous_chunk: np.ndarray | None = None
        self.previous_joint_frame: np.ndarray | None = None

    def reset(self) -> None:
        self.score_history.clear()
        self.state_history.clear()
        self.pending_actions.clear()
        self.previous_chunk = None
        self.previous_joint_frame = None

    def action(self, observation: dict[str, Any]) -> np.ndarray:
        score_frame = extract_score_frame(observation)
        state_frame, self.previous_joint_frame = extract_state_frame(
            observation,
            previous_joint_frame=self.previous_joint_frame,
            include_goal=bool(self.observation_spec.get("use_goal", True)),
            include_piano_state=bool(self.observation_spec.get("use_piano_state", True)),
            include_sustain_state=bool(self.observation_spec.get("use_sustain_state", True)),
            include_joint_velocities=bool(self.observation_spec.get("use_joint_velocities", True)),
        )
        self.score_history.append(score_frame)
        self.state_history.append(state_frame)
        while len(self.score_history) < self.obs_horizon:
            self.score_history.appendleft(score_frame.copy())
        while len(self.state_history) < self.obs_horizon:
            self.state_history.appendleft(state_frame.copy())
        if not self.pending_actions:
            score_window = np.stack(list(self.score_history), axis=0)
            state_window = np.stack(list(self.state_history), axis=0)
            warm_start = self.policy.build_warm_start(self.previous_chunk, self.execute_horizon) if self.previous_chunk is not None else None
            chunk = self.policy.sample_action_chunk(score_window=score_window, state_window=state_window, warm_start=warm_start)[0]
            self.previous_chunk = chunk
            self.pending_actions = [chunk[index] for index in range(min(self.execute_horizon, chunk.shape[0]))]
        return np.asarray(self.pending_actions.pop(0), dtype=np.float32)


def extract_score_frame(observation: dict[str, Any]) -> np.ndarray:
    goal = np.asarray(observation.get("goal", np.zeros((89,), dtype=np.float32)), dtype=np.float32).reshape(-1)
    if goal.size == 0:
        return np.zeros((14,), dtype=np.float32)
    step_dim = 89 if goal.size % 89 == 0 else 88
    trimmed = goal[: goal.size - (goal.size % step_dim)] if goal.size >= step_dim else np.pad(goal, (0, step_dim - goal.size))
    reshaped = trimmed.reshape(-1, step_dim)
    piano_roll = reshaped[:, :88]
    active = piano_roll[0] > 0.5
    histogram = np.zeros((12,), dtype=np.float32)
    active_keys = np.flatnonzero(active)
    if active_keys.size:
        histogram += np.bincount(active_keys % 12, minlength=12).astype(np.float32)
        histogram /= max(float(histogram.sum()), 1.0)
    payload = {
        "goal_histogram": histogram.tolist(),
        "active_ratio": float(active.mean()),
        "future_density": float((piano_roll > 0.5).mean()),
    }
    return context_to_feature_vector(payload)


def extract_state_frame(
    observation: dict[str, Any],
    *,
    previous_joint_frame: np.ndarray | None,
    include_goal: bool,
    include_piano_state: bool,
    include_sustain_state: bool,
    include_joint_velocities: bool,
) -> tuple[np.ndarray, np.ndarray]:
    components: list[np.ndarray] = []
    goal = np.asarray(observation.get("goal", []), dtype=np.float32).reshape(-1)
    piano_state = np.asarray(observation.get("piano/state", []), dtype=np.float32).reshape(-1)
    sustain_state = np.asarray(observation.get("piano/sustain_state", []), dtype=np.float32).reshape(-1)
    joint_keys = [key for key in observation if str(key).endswith("/joints_pos")]
    ordered_keys = [key for key in ("rh_shadow_hand/joints_pos", "lh_shadow_hand/joints_pos") if key in joint_keys] + sorted(key for key in joint_keys if key not in {"rh_shadow_hand/joints_pos", "lh_shadow_hand/joints_pos"})
    joint_frame = np.concatenate([np.asarray(observation[key], dtype=np.float32).reshape(-1) for key in ordered_keys], axis=0) if ordered_keys else np.zeros((0,), dtype=np.float32)
    if include_goal and goal.size:
        components.append(goal)
    if include_piano_state and piano_state.size:
        components.append(piano_state)
    if include_sustain_state and sustain_state.size:
        components.append(sustain_state)
    components.append(joint_frame)
    if include_joint_velocities and joint_frame.size:
        joint_velocity = joint_frame - previous_joint_frame if previous_joint_frame is not None and previous_joint_frame.shape == joint_frame.shape else np.zeros_like(joint_frame)
        components.append(joint_velocity.astype(np.float32))
    return np.concatenate([component.astype(np.float32) for component in components], axis=0), joint_frame.astype(np.float32)


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in results if "error" not in row]
    if not completed:
        return {"episodes": len(results), "successful_rollouts": 0, "mean_reward": 0.0}
    return {
        "episodes": int(len(results)),
        "successful_rollouts": int(len(completed)),
        "mean_reward": float(np.mean([row.get("reward", 0.0) for row in completed])),
        "mean_actions_executed": float(np.mean([row.get("actions_executed", 0) for row in completed])),
    }
