from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sonata.evaluation.offline import stitch_segment_predictions
from sonata.training.mjx_rollout import MJXRolloutBackend, mjx_availability
from sonata.utils.io import write_json, write_table


def evaluate_dm_control_rollout(
    *,
    primitive_root: Path,
    predictions_by_episode: dict[str, list[dict[str, Any]]],
    output_root: Path,
    limit_episodes: int = 2,
) -> dict[str, Any]:
    try:
        from robopianist import suite
        from robopianist.wrappers.evaluation import MidiEvaluationWrapper
    except Exception as exc:  # pragma: no cover
        result = {"available": False, "error": str(exc)}
        write_json(result, output_root / "dm_control_rollout.json")
        return result

    token_df = pd.read_parquet(primitive_root / "tokens" / "primitive_tokens.parquet") if (primitive_root / "tokens" / "primitive_tokens.parquet").exists() else pd.read_csv(primitive_root / "tokens" / "primitive_tokens.csv")
    results: list[dict[str, Any]] = []
    for episode_id in sorted(predictions_by_episode)[:limit_episodes]:
        episode_rows = token_df[token_df["episode_id"] == episode_id].sort_values("onset_step")
        stitched = stitch_segment_predictions(token_df=episode_rows, episode_predictions=predictions_by_episode[episode_id], action_horizon=int(predictions_by_episode[episode_id][0]["predicted"].shape[0]))
        env_name = str(episode_rows.iloc[0]["song_id"])
        try:
            env = MidiEvaluationWrapper(
                suite.load(environment_name=env_name, seed=0, task_kwargs={"control_timestep": 0.05, "n_steps_lookahead": 1}),
                deque_size=1,
            )
            timestep = env.reset()
            total_reward = 0.0
            action_dim = int(env.action_spec().shape[0])
            for action in stitched:
                control = np.zeros((action_dim,), dtype=np.float32)
                control[: min(action_dim, action.shape[0])] = action[: min(action_dim, action.shape[0])]
                timestep = env.step(control)
                total_reward += float(timestep.reward or 0.0)
                if timestep.last():
                    break
            metrics = env.get_musical_metrics()
            results.append({"episode_id": episode_id, "song_id": env_name, "reward": total_reward, **metrics})
        except Exception as exc:  # pragma: no cover
            results.append({"episode_id": episode_id, "song_id": env_name, "error": str(exc)})
    payload = {"available": True, "episodes": results}
    write_json(payload, output_root / "dm_control_rollout.json")
    write_table(pd.DataFrame(results), output_root / "dm_control_rollout")
    return payload


def evaluate_mjx_physics(
    *,
    xml_path: Path,
    action_sequences: np.ndarray,
    output_root: Path,
) -> dict[str, Any]:
    availability = mjx_availability()
    if not availability.available:
        payload = {"available": False, "error": availability.message}
        write_json(payload, output_root / "mjx_rollout.json")
        return payload
    backend = MJXRolloutBackend(xml_path=xml_path, batch_size=int(action_sequences.shape[0]))
    backend.step(action_sequences[:, 0])
    payload = {
        "available": True,
        "batch_size": int(action_sequences.shape[0]),
        "ctrl_dim": int(action_sequences.shape[-1]),
        "qpos_shape": list(backend.qpos().shape),
        "qvel_shape": list(backend.qvel().shape),
    }
    write_json(payload, output_root / "mjx_rollout.json")
    return payload
