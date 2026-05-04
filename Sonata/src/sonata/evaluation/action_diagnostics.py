"""Action magnitude and chunk diagnostics for DM Control rollouts."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import numpy as np

from sonata.utils.io import save_npz, write_json

LOGGER = logging.getLogger(__name__)

ACTION_THRESH = 1e-5


def chunk_timing_from_predictions(episode_predictions: list[dict[str, Any]]) -> dict[str, Any]:
    if not episode_predictions:
        return {
            "number_of_chunks": 0,
            "mean_chunk_duration": 0.0,
            "max_chunk_duration": 0,
            "min_chunk_duration": 0,
            "onset_step_first": None,
            "end_step_last": None,
        }
    ordered = sorted(episode_predictions, key=lambda item: (int(item["onset_step"]), int(item["end_step"])))
    durations = [max(int(item["end_step"]) - int(item["onset_step"]), 1) for item in ordered]
    return {
        "number_of_chunks": len(ordered),
        "mean_chunk_duration": float(np.mean(durations)),
        "max_chunk_duration": int(np.max(durations)),
        "min_chunk_duration": int(np.min(durations)),
        "onset_step_first": int(ordered[0]["onset_step"]),
        "end_step_last": int(ordered[-1]["end_step"]),
    }


def compute_action_magnitude_diagnostics(
    *,
    stitched: np.ndarray,
    action_source: str,
    episode_id: str,
    song_id: str,
    expected_action_dim: int,
    env_action_dim: int | None,
    control_timestep: float,
    episode_predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    arr = np.asarray(stitched, dtype=np.float64)
    action_shape = list(arr.shape)
    action_dim = int(arr.shape[-1]) if arr.ndim >= 1 else 0
    num_steps = int(arr.shape[0]) if arr.ndim >= 1 else 0
    flat = np.abs(arr.reshape(num_steps, action_dim)) if num_steps and action_dim else np.zeros((0, 0))

    mean_abs = float(np.mean(flat)) if flat.size else 0.0
    max_abs = float(np.max(flat)) if flat.size else 0.0
    std_all = float(np.std(arr)) if arr.size else 0.0
    l2_per_row = np.linalg.norm(arr, axis=-1) if arr.ndim == 2 else np.array([])
    l2_mean = float(np.mean(l2_per_row)) if l2_per_row.size else 0.0
    nonzero_fraction = float(np.mean(flat > ACTION_THRESH)) if flat.size else 0.0

    per_dim_mean_abs = np.mean(flat, axis=0).tolist() if flat.size else []
    per_dim_max_abs = np.max(flat, axis=0).tolist() if flat.size else []

    timing = chunk_timing_from_predictions(episode_predictions)

    warnings: list[str] = []
    if max_abs < 1e-4:
        warnings.append("WARN: action_max_abs < 1e-4 (near-zero actions)")
    if nonzero_fraction < 0.01:
        warnings.append("WARN: nonzero_fraction < 0.01 (almost all entries ~0)")
    if env_action_dim is not None and expected_action_dim != int(env_action_dim):
        warnings.append(
            f"WARN: expected_action_dim={expected_action_dim} != env_action_dim={env_action_dim}"
        )
    if flat.size and action_dim:
        max_per_dim = np.max(flat, axis=0)
        nonzero_mask = max_per_dim > ACTION_THRESH
        if np.any(nonzero_mask):
            last_active = int(np.where(nonzero_mask)[0][-1])
            if last_active < action_dim - 1 and np.all(max_per_dim[last_active + 1 :] <= ACTION_THRESH):
                warnings.append(
                    f"WARN: only dims 0..{last_active} exceed |action|>{ACTION_THRESH}; "
                    f"higher dims appear inactive (possible ordering/shape mismatch)"
                )
        # Clipped / normalized heuristic: many values near ±1
        vals = np.abs(arr.reshape(-1))
        near_clip = float(np.mean(np.isclose(vals, 1.0, atol=1e-4))) if vals.size else 0.0
        if near_clip > 0.5 and max_abs <= 1.0001:
            warnings.append(
                "WARN: many |action| ~= 1.0 — values may be clipped or normalized to a bounded range"
            )

    row = {
        "action_source": action_source,
        "episode_id": episode_id,
        "song_id": song_id,
        "action_shape": action_shape,
        "action_dim": action_dim,
        "expected_action_dim": int(expected_action_dim),
        "env_action_dim": int(env_action_dim) if env_action_dim is not None else None,
        "action_mean_abs": mean_abs,
        "action_max_abs": max_abs,
        "action_std": std_all,
        "action_l2_mean": l2_mean,
        "nonzero_fraction": nonzero_fraction,
        "per_dim_mean_abs": per_dim_mean_abs,
        "per_dim_max_abs": per_dim_max_abs,
        "num_steps": num_steps,
        "control_timestep": float(control_timestep),
        "warnings": warnings,
        **timing,
    }
    return row


def save_action_diagnostic_artifacts(
    *,
    output_rollout_dir: Path,
    diagnostics_row: dict[str, Any],
    stitched: np.ndarray,
    logger: logging.Logger | None = None,
) -> None:
    log = logger or LOGGER
    output_rollout_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_rollout_dir / "action_magnitude_diagnostics.csv"
    json_path = output_rollout_dir / "action_magnitude_summary.json"
    npz_path = output_rollout_dir / "action_samples.npz"

    row_out = {k: v for k, v in diagnostics_row.items() if k not in {"per_dim_mean_abs", "per_dim_max_abs", "warnings"}}
    row_out["per_dim_mean_abs"] = diagnostics_row.get("per_dim_mean_abs")
    row_out["per_dim_max_abs"] = diagnostics_row.get("per_dim_max_abs")
    row_out["warnings"] = "; ".join(diagnostics_row.get("warnings", []))

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row_out.keys()))
        writer.writeheader()
        writer.writerow({k: row_out[k] for k in row_out})

    summary_payload = dict(diagnostics_row)
    first10 = np.asarray(stitched[:10], dtype=np.float32) if stitched.size else np.zeros((0, 0), dtype=np.float32)
    summary_payload["first_10_action_rows"] = first10.tolist()
    write_json(summary_payload, json_path)

    save_npz(
        npz_path,
        stitched_actions=np.asarray(stitched, dtype=np.float32),
        first_10_actions=first10,
    )
    for w in diagnostics_row.get("warnings", []):
        log.warning("%s", w)


def init_keypress_trace() -> dict[str, Any]:
    return {
        "keys_pressed_total": 0,
        "max_keys_pressed_simultaneously": 0,
        "timesteps_with_any_key_press": 0,
        "fraction_steps_with_key_press": 0.0,
        "first_key_press_step": None,
        "steps_recorded": 0,
    }


def update_keypress_trace(trace: dict[str, Any], *, step_index: int, activation: np.ndarray | None) -> None:
    if activation is None:
        return
    act = np.asarray(activation, dtype=np.float64).reshape(-1)
    trace["steps_recorded"] = int(trace["steps_recorded"]) + 1
    pressed = act > 0.1
    count = int(np.sum(pressed))
    trace["keys_pressed_total"] = int(trace["keys_pressed_total"]) + count
    trace["max_keys_pressed_simultaneously"] = max(int(trace["max_keys_pressed_simultaneously"]), count)
    if count > 0:
        trace["timesteps_with_any_key_press"] = int(trace["timesteps_with_any_key_press"]) + 1
        if trace["first_key_press_step"] is None:
            trace["first_key_press_step"] = int(step_index)


def finalize_keypress_trace(trace: dict[str, Any]) -> dict[str, Any]:
    steps = int(trace["steps_recorded"])
    trace["fraction_steps_with_key_press"] = (
        float(trace["timesteps_with_any_key_press"]) / steps if steps > 0 else 0.0
    )
    return trace


def try_read_piano_activation(env: Any) -> np.ndarray | None:
    inner = env
    for _ in range(6):
        if inner is None:
            break
        task = getattr(inner, "task", None)
        piano = getattr(task, "piano", None) if task is not None else None
        act = getattr(piano, "activation", None) if piano is not None else None
        if act is not None:
            return np.asarray(act)
        inner = getattr(inner, "_environment", None)
    return None


def try_read_target_notes_total(env: Any) -> int | None:
    inner = env
    for _ in range(6):
        if inner is None:
            break
        task = getattr(inner, "task", None)
        notes = getattr(task, "_notes", None) if task is not None else None
        if notes is not None:
            try:
                return int(len(notes))
            except Exception:
                return None
        inner = getattr(inner, "_environment", None)
    return None
