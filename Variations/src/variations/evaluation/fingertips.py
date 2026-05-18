from __future__ import annotations

from pathlib import Path
import ctypes
import os
import sys
import tempfile
from typing import Any, Callable

import numpy as np


FINGERTIP_DIM = 30


def fingertip_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    *,
    huber_delta: float = 0.01,
    success_thresholds: tuple[float, ...] = (0.005, 0.01, 0.02, 0.05),
    prefix: str = "fingertip",
) -> dict[str, float]:
    pred = np.asarray(predicted, dtype=np.float32)
    true = np.asarray(target, dtype=np.float32)
    steps = min(int(pred.shape[0]), int(true.shape[0]))
    if steps <= 0:
        return {
            f"{prefix}_examples": 0.0,
            f"{prefix}_mse": float("nan"),
            f"{prefix}_rmse": float("nan"),
            f"{prefix}_huber": float("nan"),
        }
    pred = pred[:steps].reshape(steps, -1)
    true = true[:steps].reshape(steps, -1)
    width = min(int(pred.shape[1]), int(true.shape[1]), FINGERTIP_DIM)
    pred = pred[:, :width]
    true = true[:, :width]
    diff = pred - true
    sq = diff * diff
    abs_diff = np.abs(diff)
    delta = float(huber_delta)
    quadratic = np.minimum(abs_diff, delta)
    linear = abs_diff - quadratic
    huber = 0.5 * quadratic * quadratic + delta * linear
    per_tip_dist = np.linalg.norm(diff.reshape(steps, -1, 3), axis=2)
    per_tip_width_dist = np.abs(diff.reshape(steps, -1, 3)[:, :, 0])

    out = {
        f"{prefix}_examples": float(steps),
        f"{prefix}_mse": float(np.mean(sq)),
        f"{prefix}_rmse": float(np.sqrt(np.mean(sq))),
        f"{prefix}_huber": float(np.mean(huber)),
        f"{prefix}_per_tip_distance_mean": float(np.mean(per_tip_dist)),
        f"{prefix}_per_tip_distance_median": float(np.median(per_tip_dist)),
        f"{prefix}_per_tip_distance_p95": float(np.percentile(per_tip_dist, 95)),
        f"{prefix}_per_tip_width_distance_mean": float(np.mean(per_tip_width_dist)),
        f"{prefix}_per_tip_width_distance_median": float(np.median(per_tip_width_dist)),
        f"{prefix}_per_tip_width_distance_p95": float(np.percentile(per_tip_width_dist, 95)),
    }
    per_example_max = np.max(per_tip_dist, axis=1)
    per_example_width_max = np.max(per_tip_width_dist, axis=1)
    for threshold in success_thresholds:
        key = str(threshold).replace(".", "p")
        out[f"{prefix}_success_at_{key}"] = float(np.mean(per_example_max <= float(threshold)))
        out[f"{prefix}_width_success_at_{key}"] = float(np.mean(per_example_width_max <= float(threshold)))
    return out


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def ensure_repo_paths() -> Path:
    repo = _repo_root()
    for path in (
        repo / "Intermezzo" / "src",
        repo / "Variations" / "src",
        repo / "Variations",
        repo / "partita" / "src",
        repo,
    ):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    return repo


def _preload_conda_libstdcxx() -> None:
    lib = Path(sys.prefix) / "lib" / "libstdc++.so.6"
    if not lib.is_file():
        return
    try:
        ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
    except Exception:
        return


def capture_fingertips(task: Any, physics: Any) -> np.ndarray:
    right = np.asarray(physics.bind(task.right_hand.fingertip_sites).xpos, dtype=np.float32).reshape(-1)
    left = np.asarray(physics.bind(task.left_hand.fingertip_sites).xpos, dtype=np.float32).reshape(-1)
    values = np.concatenate([right, left], axis=0).astype(np.float32, copy=False)
    if values.shape[0] != FINGERTIP_DIM:
        raise ValueError(f"Expected {FINGERTIP_DIM} fingertip coordinates, got {values.shape[0]}.")
    return values


def measure_fingertips_with_mujoco(
    hand_joints: np.ndarray,
    *,
    target_keys: np.ndarray | None = None,
    control_timestep: float = 0.05,
    environment_name: str = "RoboPianist-debug-TwinkleTwinkleLittleStar-v0",
    seed: int = 0,
    reduced_action_space: bool = True,
    output_dir: str | Path | None = None,
    label: str = "fingertip_eval",
    settle_steps: int = 0,
    load_env_fn: Callable[..., tuple[str, Any, dict[str, Any]]] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    ensure_repo_paths()
    os.environ.setdefault("MUJOCO_GL", "egl")
    _preload_conda_libstdcxx()
    from partita.evaluation.rollout import (
        _load_env,
        _locate_task_physics_piano,
        _set_reduced_hand_qpos,
        candidate_environment_names,
        write_goals_proto,
    )

    poses = np.asarray(hand_joints, dtype=np.float32)
    if poses.ndim != 2:
        raise ValueError(f"hand_joints must be [T, 46], got {poses.shape}")
    steps = int(poses.shape[0])
    if target_keys is None:
        keys = np.zeros((steps, 88), dtype=np.float32)
    else:
        keys = np.asarray(target_keys, dtype=np.float32)
        if keys.ndim != 2 or keys.shape[1] < 88:
            raise ValueError(f"target_keys must be [T, 88+], got {keys.shape}")
        keys = keys[:steps, :88]
    if output_dir is None:
        temp_context = tempfile.TemporaryDirectory(prefix="variations_fingertips_")
        proto_dir = Path(temp_context.name)
    else:
        temp_context = None
        proto_dir = Path(output_dir)
        proto_dir.mkdir(parents=True, exist_ok=True)

    env = None
    try:
        midi_proto = write_goals_proto(
            keys,
            proto_dir / f"{label}_target_goals.proto",
            dt=float(control_timestep),
            title=f"Variations fingertip eval {label}",
        )
        loader = load_env_fn or _load_env
        env_name, env, load_info = loader(
            environment_names=candidate_environment_names(environment_name),
            midi_proto_path=midi_proto,
            control_timestep=float(control_timestep),
            seed=int(seed),
            reduced_action_space=bool(reduced_action_space),
            extra_task_kwargs={
                "disable_forearm_reward": True,
                "disable_fingering_reward": True,
                "disable_colorization": True,
                "disable_hand_collisions": False,
                "wrong_press_termination": False,
            },
            suite_load_kwargs=None,
            prefer_canonical_midi=False,
        )
        env.reset()
        task, physics, _piano = _locate_task_physics_piano(env)
        values: list[np.ndarray] = []
        restored_hand_joint_count = 0
        for pose in poses:
            restored_hand_joint_count = _set_reduced_hand_qpos(task, physics, pose)
            if hasattr(physics, "forward"):
                physics.forward()
            for _ in range(max(int(settle_steps), 0)):
                _set_reduced_hand_qpos(task, physics, pose)
                if hasattr(physics, "step"):
                    physics.step()
                elif hasattr(physics, "forward"):
                    physics.forward()
            if hasattr(physics, "forward"):
                physics.forward()
            values.append(capture_fingertips(task, physics))
        meta = {
            "environment_name": env_name,
            "load_info": load_info,
            "midi_proto_path": str(midi_proto),
            "control_timestep": float(control_timestep),
            "settle_steps": int(settle_steps),
            "restored_hand_joint_count": int(restored_hand_joint_count),
            "fingertip_order": "right_hand_sites_then_left_hand_sites",
        }
        return np.stack(values, axis=0) if values else np.zeros((0, FINGERTIP_DIM), dtype=np.float32), meta
    finally:
        if env is not None:
            close = getattr(env, "close", None)
            if callable(close):
                close()
        if temp_context is not None:
            temp_context.cleanup()
