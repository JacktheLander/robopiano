from __future__ import annotations

from dataclasses import asdict, dataclass, replace
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
    xy_tolerance_m: float = 0.003
    z_tolerance_m: float = 0.002
    worst_contact_weight: float = 0.35
    mean_contact_weight: float = 0.65
    two_stage_refinement: bool = True
    micro_max_iter: int = 40
    micro_pose_reg_weight: float = 0.003
    micro_z_weight: float = 1.0
    assignment_mode: str = "ordered"
    crossing_penalty: float = 0.05
    hand_side_penalty: float = 0.03
    target_mode: str = "key_center"
    white_key_half_width_m: float = 0.010
    black_key_half_width_m: float = 0.006
    key_front_back_tolerance_m: float = 0.018


@dataclass(frozen=True)
class ContactRefinementResult:
    refined_pose: np.ndarray
    initial_loss: float
    final_loss: float
    active_keys: int
    assigned_pairs: int
    assignment_mode: str
    crossing_penalty_applied: bool
    hand_side_penalty_applied: bool
    assigned_tip_indices: tuple[int, ...]
    assigned_key_indices: tuple[int, ...]
    initial_metrics: "ContactAlignmentMetrics"
    final_metrics: "ContactAlignmentMetrics"
    success: bool
    message: str


@dataclass(frozen=True)
class ContactAlignmentMetrics:
    mean_error_m: float
    median_error_m: float
    max_error_m: float
    p95_error_m: float
    mean_xy_error_m: float
    mean_z_error_m: float
    within_2mm_rate: float
    within_5mm_rate: float
    within_10mm_rate: float
    wrong_key_nearest_count: int
    mean_center_error_m: float
    mean_surface_error_m: float
    mean_width_center_error_m: float
    median_width_center_error_m: float
    p95_width_center_error_m: float
    max_width_center_error_m: float
    mean_width_surface_error_m: float
    median_width_surface_error_m: float
    p95_width_surface_error_m: float
    max_width_surface_error_m: float


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


_BLACK_KEY_PITCH_CLASSES = {1, 3, 6, 8, 10}


@dataclass(frozen=True)
class ContactAssignment:
    tip_positions: np.ndarray
    key_positions: np.ndarray
    tip_indices: np.ndarray
    key_indices: np.ndarray
    crossing_penalty_applied: bool = False
    hand_side_penalty_applied: bool = False


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
        self.fingertip_site_names = [getattr(site, "name", str(site)) for site in self.fingertip_sites]
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

    def _key_targets(self, active_keys: np.ndarray, cfg: ContactRefinementConfig) -> tuple[np.ndarray, np.ndarray]:
        key_positions = self.key_positions[active_keys].copy()
        key_positions[:, 2] += float(cfg.key_z_offset_m)
        key_indices = np.asarray(active_keys, dtype=np.int64).copy()
        if key_positions.shape[0] == 0:
            return key_positions.reshape(0, 3), key_indices
        if key_positions.shape[0] > int(cfg.max_active_keys):
            key_positions = key_positions[: int(cfg.max_active_keys)]
            key_indices = key_indices[: int(cfg.max_active_keys)]
        return key_positions, key_indices

    def _ordered_finger_indices(self, tips: np.ndarray) -> np.ndarray:
        names = [name.lower() for name in self.fingertip_site_names]
        left_order = ("pinky", "ring", "middle", "index", "thumb")
        right_order = ("thumb", "index", "middle", "ring", "pinky")

        def side_indices(side: str, order: tuple[str, ...]) -> list[int]:
            side_matches = [idx for idx, name in enumerate(names) if side in name]
            ordered: list[int] = []
            for finger in order:
                matches = [idx for idx in side_matches if finger in names[idx] and idx not in ordered]
                if matches:
                    ordered.append(matches[0])
            return ordered

        ordered = side_indices("left", left_order) + side_indices("right", right_order)
        if len(ordered) == tips.shape[0]:
            return np.asarray(ordered, dtype=np.int64)
        return np.argsort(tips[:, 0]).astype(np.int64)

    def _assignment_info(self, tips: np.ndarray, active_keys: np.ndarray, cfg: ContactRefinementConfig) -> ContactAssignment:
        key_positions, key_indices = self._key_targets(active_keys, cfg)
        empty = ContactAssignment(
            tip_positions=np.zeros((0, 3), dtype=np.float64),
            key_positions=np.zeros((0, 3), dtype=np.float64),
            tip_indices=np.zeros((0,), dtype=np.int64),
            key_indices=np.zeros((0,), dtype=np.int64),
        )
        if key_positions.shape[0] == 0:
            return empty
        mode = str(cfg.assignment_mode).strip().lower()
        if mode == "ordered":
            tip_order = self._ordered_finger_indices(tips)
            usable = min(tip_order.size, key_positions.shape[0])
            key_order = np.argsort(key_positions[:, 0]).astype(np.int64)
            tip_idx = tip_order[:usable]
            key_order = key_order[:usable]
            return ContactAssignment(tips[tip_idx], key_positions[key_order], tip_idx, key_indices[key_order])

        cost = np.linalg.norm(tips[:, None, :2] - key_positions[None, :, :2], axis=2)
        crossing_applied = False
        hand_side_applied = False
        if mode == "hand_constrained":
            finger_rank = np.empty(tips.shape[0], dtype=np.float64)
            finger_rank[self._ordered_finger_indices(tips)] = np.arange(tips.shape[0], dtype=np.float64)
            key_rank = np.empty(key_positions.shape[0], dtype=np.float64)
            key_rank[np.argsort(key_positions[:, 0])] = np.arange(key_positions.shape[0], dtype=np.float64)
            denom = max(tips.shape[0] - 1, key_positions.shape[0] - 1, 1)
            cost = cost + float(cfg.crossing_penalty) * np.abs(finger_rank[:, None] - key_rank[None, :]) / float(denom)
            crossing_applied = float(cfg.crossing_penalty) > 0.0

            keyboard_mid_x = float(np.median(self.key_positions[:, 0]))
            left_tip = tips[:, 0] < keyboard_mid_x
            left_key = key_positions[:, 0] < keyboard_mid_x
            mismatch = left_tip[:, None] != left_key[None, :]
            cost = cost + float(cfg.hand_side_penalty) * mismatch.astype(np.float64)
            hand_side_applied = float(cfg.hand_side_penalty) > 0.0

        try:
            from scipy.optimize import linear_sum_assignment

            row, col = linear_sum_assignment(cost)
            width = min(len(row), key_positions.shape[0])
            row = np.asarray(row[:width], dtype=np.int64)
            col = np.asarray(col[:width], dtype=np.int64)
            return ContactAssignment(tips[row], key_positions[col], row, key_indices[col], crossing_applied, hand_side_applied)
        except Exception:
            tip_idx = np.argmin(cost, axis=0)
            tip_idx = np.asarray(tip_idx, dtype=np.int64)
            return ContactAssignment(tips[tip_idx], key_positions, tip_idx, key_indices, crossing_applied, hand_side_applied)

    def _assignment(self, tips: np.ndarray, active_keys: np.ndarray, cfg: ContactRefinementConfig) -> tuple[np.ndarray, np.ndarray]:
        assignment = self._assignment_info(tips, active_keys, cfg)
        return assignment.tip_positions, assignment.key_positions

    def _surface_xy_errors(self, tip_positions: np.ndarray, key_positions: np.ndarray, key_indices: np.ndarray, cfg: ContactRefinementConfig) -> np.ndarray:
        if tip_positions.size == 0:
            return np.zeros((0,), dtype=np.float64)
        widths = np.asarray(
            [
                float(cfg.black_key_half_width_m) if int(key) % 12 in _BLACK_KEY_PITCH_CLASSES else float(cfg.white_key_half_width_m)
                for key in key_indices
            ],
            dtype=np.float64,
        )
        dx = np.maximum(np.abs(tip_positions[:, 0] - key_positions[:, 0]) - widths, 0.0)
        dy = np.maximum(np.abs(tip_positions[:, 1] - key_positions[:, 1]) - float(cfg.key_front_back_tolerance_m), 0.0)
        return np.sqrt(dx**2 + dy**2)

    def _surface_width_errors(
        self,
        tip_positions: np.ndarray,
        key_positions: np.ndarray,
        key_indices: np.ndarray,
        cfg: ContactRefinementConfig,
    ) -> np.ndarray:
        if tip_positions.size == 0:
            return np.zeros((0,), dtype=np.float64)
        widths = np.asarray(
            [
                float(cfg.black_key_half_width_m) if int(key) % 12 in _BLACK_KEY_PITCH_CLASSES else float(cfg.white_key_half_width_m)
                for key in key_indices
            ],
            dtype=np.float64,
        )
        return np.maximum(np.abs(tip_positions[:, 0] - key_positions[:, 0]) - widths, 0.0)

    def _contact_errors(
        self,
        tip_positions: np.ndarray,
        key_positions: np.ndarray,
        key_indices: np.ndarray,
        cfg: ContactRefinementConfig,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if tip_positions.size == 0:
            empty = np.zeros((0,), dtype=np.float64)
            return empty, empty, empty
        diff = tip_positions - key_positions
        center_xy = np.linalg.norm(diff[:, :2], axis=1)
        surface_xy = self._surface_xy_errors(tip_positions, key_positions, key_indices, cfg)
        xy_err = surface_xy if str(cfg.target_mode).strip().lower() == "key_surface_box" else center_xy
        z_err = np.abs(diff[:, 2])
        contact_err = xy_err + float(cfg.z_weight) * z_err
        return contact_err, xy_err, z_err

    def contact_metrics(
        self,
        pose: np.ndarray,
        target_keys: np.ndarray,
        config: ContactRefinementConfig | None = None,
    ) -> ContactAlignmentMetrics:
        cfg = config or ContactRefinementConfig()
        tips = self.fingertip_positions(pose)
        active_keys = np.flatnonzero(np.asarray(target_keys, dtype=np.float32).reshape(-1)[:88] > 0.5)
        assignment = self._assignment_info(tips, active_keys, cfg)
        if assignment.tip_positions.size == 0:
            return ContactAlignmentMetrics(
                mean_error_m=0.0,
                median_error_m=0.0,
                max_error_m=0.0,
                p95_error_m=0.0,
                mean_xy_error_m=0.0,
                mean_z_error_m=0.0,
                within_2mm_rate=1.0,
                within_5mm_rate=1.0,
                within_10mm_rate=1.0,
                wrong_key_nearest_count=0,
                mean_center_error_m=0.0,
                mean_surface_error_m=0.0,
                mean_width_center_error_m=0.0,
                median_width_center_error_m=0.0,
                p95_width_center_error_m=0.0,
                max_width_center_error_m=0.0,
                mean_width_surface_error_m=0.0,
                median_width_surface_error_m=0.0,
                p95_width_surface_error_m=0.0,
                max_width_surface_error_m=0.0,
            )
        diff = assignment.tip_positions - assignment.key_positions
        center_error = np.linalg.norm(diff, axis=1)
        width_center_error = np.abs(diff[:, 0])
        width_surface_error = self._surface_width_errors(assignment.tip_positions, assignment.key_positions, assignment.key_indices, cfg)
        surface_xy = self._surface_xy_errors(assignment.tip_positions, assignment.key_positions, assignment.key_indices, cfg)
        surface_error = np.sqrt(surface_xy**2 + diff[:, 2] ** 2)
        error = surface_error if str(cfg.target_mode).strip().lower() == "key_surface_box" else center_error
        xy_error = surface_xy if str(cfg.target_mode).strip().lower() == "key_surface_box" else np.linalg.norm(diff[:, :2], axis=1)
        z_error = np.abs(diff[:, 2])
        active_limited = assignment.key_indices
        nearest_active = active_limited[
            np.argmin(np.linalg.norm(assignment.tip_positions[:, None, :2] - self.key_positions[active_limited][None, :, :2], axis=2), axis=1)
        ]
        wrong_key_nearest_count = int(np.sum(nearest_active != assignment.key_indices))
        return ContactAlignmentMetrics(
            mean_error_m=float(np.mean(error)),
            median_error_m=float(np.median(error)),
            max_error_m=float(np.max(error)),
            p95_error_m=float(np.percentile(error, 95)),
            mean_xy_error_m=float(np.mean(xy_error)),
            mean_z_error_m=float(np.mean(z_error)),
            within_2mm_rate=float(np.mean(error <= 0.002)),
            within_5mm_rate=float(np.mean(error <= 0.005)),
            within_10mm_rate=float(np.mean(error <= 0.010)),
            wrong_key_nearest_count=wrong_key_nearest_count,
            mean_center_error_m=float(np.mean(center_error)),
            mean_surface_error_m=float(np.mean(surface_error)),
            mean_width_center_error_m=float(np.mean(width_center_error)),
            median_width_center_error_m=float(np.median(width_center_error)),
            p95_width_center_error_m=float(np.percentile(width_center_error, 95)),
            max_width_center_error_m=float(np.max(width_center_error)),
            mean_width_surface_error_m=float(np.mean(width_surface_error)),
            median_width_surface_error_m=float(np.median(width_surface_error)),
            p95_width_surface_error_m=float(np.percentile(width_surface_error, 95)),
            max_width_surface_error_m=float(np.max(width_surface_error)),
        )

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
        assignment = self._assignment_info(tips, active_keys, cfg)
        if assignment.tip_positions.size:
            contact_err, xy_err, z_err = self._contact_errors(assignment.tip_positions, assignment.key_positions, assignment.key_indices, cfg)
            xy_loss = float(np.mean(np.maximum(xy_err - float(cfg.xy_tolerance_m), 0.0) ** 2))
            z_loss = float(np.mean(np.maximum(z_err - float(cfg.z_tolerance_m), 0.0) ** 2))
            mean_loss = float(np.mean(contact_err**2))
            worst_loss = float(np.max(contact_err**2))
            press_loss = (
                float(cfg.mean_contact_weight) * mean_loss
                + float(cfg.worst_contact_weight) * worst_loss
                + xy_loss
                + float(cfg.z_weight) * z_loss
            )
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
            metrics = self.contact_metrics(q0, target_keys, cfg)
            return ContactRefinementResult(
                refined_pose=q0.astype(np.float32),
                initial_loss=0.0,
                final_loss=0.0,
                active_keys=0,
                assigned_pairs=0,
                assignment_mode=str(cfg.assignment_mode),
                crossing_penalty_applied=False,
                hand_side_penalty_applied=False,
                assigned_tip_indices=(),
                assigned_key_indices=(),
                initial_metrics=metrics,
                final_metrics=metrics,
                success=True,
                message="no_active_keys",
            )
        initial_loss = self._loss(q0, initial_pose=q0, target_keys=target_keys, previous_pose=previous_pose, cfg=cfg)
        initial_metrics = self.contact_metrics(q0, target_keys, cfg)
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
            if bool(cfg.two_stage_refinement):
                micro_cfg = replace(
                    cfg,
                    max_iter=int(cfg.micro_max_iter),
                    pose_reg_weight=float(cfg.micro_pose_reg_weight),
                    z_weight=float(cfg.micro_z_weight),
                    two_stage_refinement=False,
                )
                micro = minimize(
                    lambda q: self._loss(q, initial_pose=q0, target_keys=target_keys, previous_pose=previous_pose, cfg=micro_cfg),
                    refined,
                    method="L-BFGS-B",
                    bounds=list(zip(self.joint_lower, self.joint_upper)),
                    options={"maxiter": int(micro_cfg.max_iter), "ftol": float(micro_cfg.ftol), "maxls": 20},
                )
                micro_loss = float(micro.fun)
                if bool(micro.success or micro_loss <= final_loss):
                    refined = np.asarray(micro.x, dtype=np.float64)
                    final_loss = micro_loss
                    success = bool(success or micro.success or final_loss < initial_loss)
                    message = f"{message}; micro: {micro.message}"
        except Exception as exc:
            refined = q0
            final_loss = initial_loss
            success = False
            message = f"optimizer_failed: {exc}"
        tips = self.fingertip_positions(refined)
        assignment = self._assignment_info(tips, active_keys, cfg)
        final_metrics = self.contact_metrics(refined, target_keys, cfg)
        return ContactRefinementResult(
            refined_pose=refined.astype(np.float32),
            initial_loss=float(initial_loss),
            final_loss=float(final_loss),
            active_keys=int(active_keys.size),
            assigned_pairs=int(assignment.tip_positions.shape[0]),
            assignment_mode=str(cfg.assignment_mode),
            crossing_penalty_applied=bool(assignment.crossing_penalty_applied),
            hand_side_penalty_applied=bool(assignment.hand_side_penalty_applied),
            assigned_tip_indices=tuple(int(idx) for idx in assignment.tip_indices.tolist()),
            assigned_key_indices=tuple(int(idx) for idx in assignment.key_indices.tolist()),
            initial_metrics=initial_metrics,
            final_metrics=final_metrics,
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
