from __future__ import annotations

from dataclasses import asdict, dataclass
import ctypes
import os
from pathlib import Path
import sys
import tempfile
import types
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ContactRefinementConfig:
    max_iter: int = 40
    pose_reg_weight: float = 0.03
    smooth_weight: float = 0.01
    inactive_margin_m: float = 0.018
    inactive_weight: float = 0.02
    z_weight: float = 0.25
    key_z_offset_m: float = 0.0
    max_active_keys: int = 10
    ftol: float = 1e-7


@dataclass(frozen=True)
class ContactRefinementResult:
    refined_pose: np.ndarray
    initial_loss: float
    final_loss: float
    active_keys: int
    assigned_pairs: int
    success: bool
    message: str


def ensure_repo_paths() -> Path:
    repo = Path(__file__).resolve().parents[3]
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


def _stub_fluidsynth_for_headless_imports() -> None:
    # RoboPianist imports pretty_midi on suite import. On WAVE login/batch nodes,
    # pretty_midi's optional fluidsynth binding can fail before we ever use audio.
    # A dummy module is enough for these physics-only contact-refinement utilities.
    sys.modules.setdefault("fluidsynth", types.ModuleType("fluidsynth"))


def _target_key_roll(active_key_template: np.ndarray | None = None) -> np.ndarray:
    if active_key_template is None:
        return np.zeros((2, 88), dtype=np.float32)
    keys = np.asarray(active_key_template, dtype=np.float32).reshape(-1, 88)
    return keys[: max(int(keys.shape[0]), 2)]


class ContactRefiner:
    def __init__(
        self,
        *,
        control_timestep: float = 0.05,
        environment_name: str = "RoboPianist-debug-TwinkleTwinkleLittleStar-v0",
        seed: int = 0,
        reduced_action_space: bool = True,
        output_dir: str | Path | None = None,
    ) -> None:
        ensure_repo_paths()
        _preload_conda_libstdcxx()
        _stub_fluidsynth_for_headless_imports()
        os.environ.setdefault("MUJOCO_GL", "egl")
        from partita.evaluation.rollout import (
            _load_env,
            _locate_task_physics_piano,
            candidate_environment_names,
            write_goals_proto,
        )

        self._temp_context = None
        if output_dir is None:
            self._temp_context = tempfile.TemporaryDirectory(prefix="variations_contact_refine_")
            proto_dir = Path(self._temp_context.name)
        else:
            proto_dir = Path(output_dir)
            proto_dir.mkdir(parents=True, exist_ok=True)
        midi_proto = write_goals_proto(
            _target_key_roll(),
            proto_dir / "contact_refinement_probe.proto",
            dt=float(control_timestep),
            title="Variations contact refinement",
        )
        env_name, env, load_info = _load_env(
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
        task, physics, piano = _locate_task_physics_piano(env)
        self.env_name = env_name
        self.load_info = load_info
        self.env = env
        self.task = task
        self.physics = physics
        self.piano = piano
        self.hand_joints = [*list(task.right_hand.joints), *list(task.left_hand.joints)]
        self.fingertip_sites = [*list(task.right_hand.fingertip_sites), *list(task.left_hand.fingertip_sites)]
        self.key_sites = list(getattr(piano, "sites", getattr(piano, "_sites")))
        self.key_positions = np.asarray(physics.bind(self.key_sites).xpos, dtype=np.float64).reshape(88, 3)
        ranges = np.asarray(physics.bind(self.hand_joints).range, dtype=np.float64)
        if ranges.shape != (len(self.hand_joints), 2):
            ranges = np.tile(np.asarray([-np.inf, np.inf], dtype=np.float64), (len(self.hand_joints), 1))
        self.joint_lower = ranges[:, 0]
        self.joint_upper = ranges[:, 1]

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if callable(close):
            close()
        if self._temp_context is not None:
            self._temp_context.cleanup()

    def __enter__(self) -> "ContactRefiner":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def set_pose(self, pose: np.ndarray) -> None:
        values = np.asarray(pose, dtype=np.float64).reshape(-1)
        if values.size < len(self.hand_joints):
            raise ValueError(f"Pose has {values.size} values, expected at least {len(self.hand_joints)}.")
        self.physics.bind(self.hand_joints).qpos = values[: len(self.hand_joints)]
        if hasattr(self.physics, "forward"):
            self.physics.forward()

    def fingertip_positions(self, pose: np.ndarray) -> np.ndarray:
        self.set_pose(pose)
        return np.asarray(self.physics.bind(self.fingertip_sites).xpos, dtype=np.float64).reshape(-1, 3)

    def _assignment(self, tips: np.ndarray, active_keys: np.ndarray, cfg: ContactRefinementConfig) -> tuple[np.ndarray, np.ndarray]:
        key_positions = self.key_positions[active_keys].copy()
        key_positions[:, 2] += float(cfg.key_z_offset_m)
        if key_positions.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)
        if key_positions.shape[0] > int(cfg.max_active_keys):
            key_positions = key_positions[: int(cfg.max_active_keys)]
        cost = np.linalg.norm(tips[:, None, :2] - key_positions[None, :, :2], axis=2)
        try:
            from scipy.optimize import linear_sum_assignment

            row, col = linear_sum_assignment(cost)
            width = min(len(row), key_positions.shape[0])
            return tips[row[:width]], key_positions[col[:width]]
        except Exception:
            tip_idx = np.argmin(cost, axis=0)
            return tips[tip_idx], key_positions

    def _loss(
        self,
        pose: np.ndarray,
        *,
        initial_pose: np.ndarray,
        target_keys: np.ndarray,
        previous_pose: np.ndarray | None,
        cfg: ContactRefinementConfig,
    ) -> float:
        tips = self.fingertip_positions(pose)
        active_keys = np.flatnonzero(np.asarray(target_keys, dtype=np.float32).reshape(-1)[:88] > 0.5)
        assigned_tips, assigned_keys = self._assignment(tips, active_keys, cfg)
        if assigned_tips.size:
            diff = assigned_tips - assigned_keys
            press_loss = float(np.mean(diff[:, :2] ** 2) + float(cfg.z_weight) * np.mean(diff[:, 2] ** 2))
        else:
            press_loss = 0.0

        reg_loss = float(np.mean((pose - initial_pose) ** 2))
        smooth_loss = 0.0 if previous_pose is None else float(np.mean((pose - previous_pose) ** 2))
        inactive_loss = 0.0
        inactive_keys = np.setdiff1d(np.arange(88), active_keys, assume_unique=False)
        if inactive_keys.size and float(cfg.inactive_weight) > 0:
            inactive_xy = self.key_positions[inactive_keys, :2]
            d = np.linalg.norm(tips[:, None, :2] - inactive_xy[None, :, :], axis=2)
            nearest = np.min(d, axis=1)
            inactive_loss = float(np.mean(np.maximum(float(cfg.inactive_margin_m) - nearest, 0.0) ** 2))
        return (
            press_loss
            + float(cfg.pose_reg_weight) * reg_loss
            + float(cfg.smooth_weight) * smooth_loss
            + float(cfg.inactive_weight) * inactive_loss
        )

    def refine_pose(
        self,
        initial_pose: np.ndarray,
        target_keys: np.ndarray,
        *,
        previous_pose: np.ndarray | None = None,
        config: ContactRefinementConfig | None = None,
    ) -> ContactRefinementResult:
        cfg = config or ContactRefinementConfig()
        q0 = np.asarray(initial_pose, dtype=np.float64).reshape(-1)[: len(self.hand_joints)]
        q0 = np.clip(q0, self.joint_lower, self.joint_upper)
        active_keys = np.flatnonzero(np.asarray(target_keys, dtype=np.float32).reshape(-1)[:88] > 0.5)
        if active_keys.size == 0:
            return ContactRefinementResult(
                refined_pose=q0.astype(np.float32),
                initial_loss=0.0,
                final_loss=0.0,
                active_keys=0,
                assigned_pairs=0,
                success=True,
                message="no_active_keys",
            )
        initial_loss = self._loss(q0, initial_pose=q0, target_keys=target_keys, previous_pose=previous_pose, cfg=cfg)
        try:
            from scipy.optimize import minimize

            result = minimize(
                lambda q: self._loss(q, initial_pose=q0, target_keys=target_keys, previous_pose=previous_pose, cfg=cfg),
                q0,
                method="L-BFGS-B",
                bounds=list(zip(self.joint_lower, self.joint_upper)),
                options={"maxiter": int(cfg.max_iter), "ftol": float(cfg.ftol), "maxls": 20},
            )
            refined = np.asarray(result.x, dtype=np.float64)
            final_loss = float(result.fun)
            success = bool(result.success or final_loss < initial_loss)
            message = str(result.message)
        except Exception as exc:
            refined = q0
            final_loss = initial_loss
            success = False
            message = f"optimizer_failed: {exc}"
        tips = self.fingertip_positions(refined)
        assigned_tips, _assigned_keys = self._assignment(tips, active_keys, cfg)
        return ContactRefinementResult(
            refined_pose=refined.astype(np.float32),
            initial_loss=float(initial_loss),
            final_loss=float(final_loss),
            active_keys=int(active_keys.size),
            assigned_pairs=int(assigned_tips.shape[0]),
            success=success,
            message=message,
        )


def config_from_dict(values: dict[str, Any] | None) -> ContactRefinementConfig:
    if not values:
        return ContactRefinementConfig()
    allowed = set(ContactRefinementConfig.__dataclass_fields__.keys())
    return ContactRefinementConfig(**{key: value for key, value in dict(values).items() if key in allowed})


def result_dict(result: ContactRefinementResult) -> dict[str, Any]:
    out = asdict(result)
    out["refined_pose"] = np.asarray(result.refined_pose, dtype=np.float32)
    return out
