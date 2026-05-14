from __future__ import annotations

from dataclasses import dataclass
import ctypes
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from bagatelle.assignment import FingerAssignmentResult, NUM_FINGERS
from bagatelle.config import BagatelleConfig
from bagatelle.paths import ensure_repo_paths


HAND_STATE_DIM = 46


@dataclass(frozen=True)
class IKResult:
    pose: np.ndarray
    fingertip_positions: np.ndarray
    assigned_distances: np.ndarray
    residual_norm: float
    max_residual: float
    success: bool
    optimizer_success: bool
    optimizer_status: int
    optimizer_message: str
    optimizer_cost: float
    nfev: int
    active_keys: np.ndarray
    assigned_keys: np.ndarray
    assigned_finger_indices: np.ndarray
    unassigned_keys: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_keys": self.active_keys.astype(int).tolist(),
            "assigned_keys": self.assigned_keys.astype(int).tolist(),
            "assigned_finger_indices": self.assigned_finger_indices.astype(int).tolist(),
            "unassigned_keys": self.unassigned_keys.astype(int).tolist(),
            "assigned_distances": self.assigned_distances.astype(float).tolist(),
            "residual_norm": float(self.residual_norm),
            "max_residual": float(self.max_residual),
            "success": bool(self.success),
            "optimizer_success": bool(self.optimizer_success),
            "optimizer_status": int(self.optimizer_status),
            "optimizer_message": str(self.optimizer_message),
            "optimizer_cost": float(self.optimizer_cost),
            "nfev": int(self.nfev),
        }


def _preload_conda_libstdcxx() -> None:
    import sys

    lib = Path(sys.prefix) / "lib" / "libstdc++.so.6"
    if not lib.is_file():
        return
    try:
        ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
    except Exception:
        return


class BagatelleKinematics:
    """RoboPianist-backed forward kinematics and bounded reduced-hand IK."""

    def __init__(
        self,
        config: BagatelleConfig | None = None,
        *,
        target_keys: np.ndarray | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        self.config = config or BagatelleConfig()
        ensure_repo_paths()
        os.environ.setdefault("MUJOCO_GL", "egl")
        _preload_conda_libstdcxx()

        from partita.evaluation.rollout import (
            _load_env,
            _locate_task_physics_piano,
            candidate_environment_names,
            write_goals_proto,
        )

        if output_dir is None:
            self._temp_context = tempfile.TemporaryDirectory(prefix="bagatelle_kinematics_")
            proto_dir = Path(self._temp_context.name)
        else:
            self._temp_context = None
            proto_dir = Path(output_dir)
            proto_dir.mkdir(parents=True, exist_ok=True)

        goals = np.asarray(target_keys, dtype=np.float32) if target_keys is not None else np.zeros((1, 88), dtype=np.float32)
        if goals.ndim != 2 or goals.shape[1] < 88:
            raise ValueError(f"target_keys must have shape [T, 88+], got {goals.shape}")
        self.midi_proto_path = write_goals_proto(
            goals[:, :88],
            proto_dir / "bagatelle_target_goals.proto",
            dt=float(self.config.control_timestep),
            title="Bagatelle target goals",
        )
        self.environment_name, self.env, self.load_info = _load_env(
            environment_names=candidate_environment_names(self.config.environment_name),
            midi_proto_path=self.midi_proto_path,
            control_timestep=float(self.config.control_timestep),
            seed=int(self.config.seed),
            reduced_action_space=bool(self.config.reduced_action_space),
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
        self.env.reset()
        self.task, self.physics, self.piano = _locate_task_physics_piano(self.env)
        self.joint_handles = self._hand_joint_handles()
        if len(self.joint_handles) != HAND_STATE_DIM:
            raise RuntimeError(f"Expected {HAND_STATE_DIM} reduced hand joints, got {len(self.joint_handles)}")
        self.fingertip_sites = tuple(self.task.left_hand.fingertip_sites) + tuple(self.task.right_hand.fingertip_sites)
        if len(self.fingertip_sites) != NUM_FINGERS:
            raise RuntimeError(f"Expected {NUM_FINGERS} fingertip sites, got {len(self.fingertip_sites)}")

        bounds = np.asarray(self.physics.bind(self.joint_handles).range, dtype=np.float64)
        if bounds.shape != (HAND_STATE_DIM, 2):
            raise RuntimeError(f"Expected joint bounds [{HAND_STATE_DIM}, 2], got {bounds.shape}")
        self.joint_lower = bounds[:, 0].astype(np.float32)
        self.joint_upper = bounds[:, 1].astype(np.float32)
        self._repair_bounds()
        self.neutral_qpos = self.current_qpos()
        self.neutral_fingertips = self.current_fingertips()

    def close(self) -> None:
        close = getattr(getattr(self, "env", None), "close", None)
        if callable(close):
            close()
        if getattr(self, "_temp_context", None) is not None:
            self._temp_context.cleanup()
            self._temp_context = None

    def __enter__(self) -> "BagatelleKinematics":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @property
    def joint_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.joint_lower.copy(), self.joint_upper.copy()

    def _repair_bounds(self) -> None:
        for index in range(HAND_STATE_DIM):
            lo = float(self.joint_lower[index])
            hi = float(self.joint_upper[index])
            if not np.isfinite(lo):
                lo = -1.0
            if not np.isfinite(hi):
                hi = 1.0
            if hi <= lo:
                center = 0.5 * (lo + hi)
                lo = center - 1e-3
                hi = center + 1e-3
            self.joint_lower[index] = lo
            self.joint_upper[index] = hi

    def _hand_joint_handles(self) -> list[Any]:
        joints: list[Any] = []
        for hand_name in ("right_hand", "left_hand"):
            hand = getattr(self.task, hand_name)
            joints.extend(list(getattr(hand, "joints")))
        return joints

    def current_qpos(self) -> np.ndarray:
        values = []
        for joint in self.joint_handles:
            qpos = np.asarray(self.physics.bind(joint).qpos, dtype=np.float64).reshape(-1)
            values.append(float(qpos[0]) if qpos.size else 0.0)
        return np.asarray(values, dtype=np.float32)

    def _set_qpos(self, qpos: np.ndarray) -> None:
        values = self.clip_qpos(qpos)
        for joint, value in zip(self.joint_handles, values):
            self.physics.bind(joint).qpos = float(value)
        if hasattr(self.physics, "forward"):
            self.physics.forward()

    def clip_qpos(self, qpos: np.ndarray) -> np.ndarray:
        values = np.asarray(qpos, dtype=np.float32).reshape(-1)
        if values.shape != (HAND_STATE_DIM,):
            raise ValueError(f"qpos must have shape [{HAND_STATE_DIM}], got {values.shape}")
        return np.clip(values, self.joint_lower, self.joint_upper).astype(np.float32)

    def current_fingertips(self) -> np.ndarray:
        positions = np.asarray(self.physics.bind(self.fingertip_sites).xpos, dtype=np.float32)
        if positions.shape != (NUM_FINGERS, 3):
            raise RuntimeError(f"Expected fingertip positions [10, 3], got {positions.shape}")
        return np.ascontiguousarray(positions, dtype=np.float32)

    def fingertip_positions_for_qpos(self, qpos: np.ndarray) -> np.ndarray:
        self._set_qpos(qpos)
        return self.current_fingertips()

    def _key_targets(self, keys: np.ndarray | None, *, press_depth: float) -> np.ndarray:
        if keys is None:
            key_indices = np.arange(88, dtype=np.int32)
        else:
            key_indices = np.asarray(keys, dtype=np.int32).reshape(-1)
        targets = []
        for key in key_indices:
            if int(key) < 0 or int(key) >= self.piano.n_keys:
                raise ValueError(f"Invalid piano key index: {int(key)}")
            key_geom = self.piano.keys[int(key)].geom[0]
            bind = self.physics.bind(key_geom)
            pos = np.asarray(bind.xpos, dtype=np.float32).copy()
            size = np.asarray(bind.size, dtype=np.float32)
            pos[-1] += float(self.config.key_target_top_offset) * float(size[2])
            pos[0] += float(self.config.key_target_front_offset) * float(size[0])
            pos[-1] -= float(press_depth)
            targets.append(pos)
        return np.stack(targets, axis=0).astype(np.float32) if targets else np.zeros((0, 3), dtype=np.float32)

    def key_contact_targets(self, keys: np.ndarray | None = None) -> np.ndarray:
        return self._key_targets(keys, press_depth=0.0)

    def key_press_targets(self, keys: np.ndarray | None = None, *, press_depth: float | None = None) -> np.ndarray:
        depth = self.config.key_press_depth if press_depth is None else float(press_depth)
        return self._key_targets(keys, press_depth=float(depth))

    def solve_press_pose(
        self,
        assignments: FingerAssignmentResult,
        previous_qpos: np.ndarray,
        neutral_qpos: np.ndarray | None = None,
        config: BagatelleConfig | None = None,
    ) -> IKResult:
        cfg = config or self.config
        previous = self.clip_qpos(previous_qpos)
        neutral = self.clip_qpos(neutral_qpos if neutral_qpos is not None else self.neutral_qpos)
        if assignments.count == 0:
            fingertips = self.fingertip_positions_for_qpos(previous)
            return self._result_from_pose(
                previous,
                fingertips,
                assignments,
                optimizer_success=True,
                optimizer_status=0,
                optimizer_message="no active assignments",
                optimizer_cost=0.0,
                nfev=0,
                threshold=float(cfg.residual_success_threshold),
            )

        finger_indices = assignments.assigned_finger_indices.astype(np.int64)
        target_positions = assignments.target_positions.astype(np.float32)
        x0 = self.clip_qpos(previous)

        def residual(values: np.ndarray) -> np.ndarray:
            q = self.clip_qpos(values)
            fingertips = self.fingertip_positions_for_qpos(q)
            parts = [
                (fingertips[finger_indices] - target_positions).reshape(-1) * float(cfg.ik_fingertip_weight),
                (q - previous).reshape(-1) * float(cfg.ik_smoothness_weight),
                (q - neutral).reshape(-1) * float(cfg.ik_neutral_weight),
            ]
            return np.concatenate(parts, axis=0).astype(np.float64)

        try:
            opt = least_squares(
                residual,
                x0.astype(np.float64),
                bounds=(self.joint_lower.astype(np.float64), self.joint_upper.astype(np.float64)),
                max_nfev=max(int(cfg.ik_max_nfev), 1),
                ftol=float(cfg.ik_ftol),
                xtol=float(cfg.ik_xtol),
                gtol=float(cfg.ik_gtol),
            )
            pose = self.clip_qpos(opt.x)
            fingertips = self.fingertip_positions_for_qpos(pose)
            return self._result_from_pose(
                pose,
                fingertips,
                assignments,
                optimizer_success=bool(opt.success),
                optimizer_status=int(opt.status),
                optimizer_message=str(opt.message),
                optimizer_cost=float(opt.cost),
                nfev=int(opt.nfev),
                threshold=float(cfg.residual_success_threshold),
            )
        except Exception as exc:
            fingertips = self.fingertip_positions_for_qpos(previous)
            return self._result_from_pose(
                previous,
                fingertips,
                assignments,
                optimizer_success=False,
                optimizer_status=-1,
                optimizer_message=f"IK exception: {type(exc).__name__}: {exc}",
                optimizer_cost=float("nan"),
                nfev=0,
                threshold=float(cfg.residual_success_threshold),
            )

    def _result_from_pose(
        self,
        pose: np.ndarray,
        fingertips: np.ndarray,
        assignments: FingerAssignmentResult,
        *,
        optimizer_success: bool,
        optimizer_status: int,
        optimizer_message: str,
        optimizer_cost: float,
        nfev: int,
        threshold: float,
    ) -> IKResult:
        if assignments.count:
            distances = np.linalg.norm(
                fingertips[assignments.assigned_finger_indices.astype(np.int64)] - assignments.target_positions,
                axis=1,
            ).astype(np.float32)
        else:
            distances = np.zeros((0,), dtype=np.float32)
        max_residual = float(np.max(distances)) if distances.size else 0.0
        residual_norm = float(np.linalg.norm(distances)) if distances.size else 0.0
        return IKResult(
            pose=np.asarray(pose, dtype=np.float32).copy(),
            fingertip_positions=np.asarray(fingertips, dtype=np.float32).copy(),
            assigned_distances=distances,
            residual_norm=residual_norm,
            max_residual=max_residual,
            success=bool(optimizer_success and max_residual <= float(threshold) and assignments.unassigned_keys.size == 0),
            optimizer_success=bool(optimizer_success),
            optimizer_status=int(optimizer_status),
            optimizer_message=str(optimizer_message),
            optimizer_cost=float(optimizer_cost),
            nfev=int(nfev),
            active_keys=assignments.active_keys.copy(),
            assigned_keys=assignments.assigned_keys.copy(),
            assigned_finger_indices=assignments.assigned_finger_indices.copy(),
            unassigned_keys=assignments.unassigned_keys.copy(),
        )
