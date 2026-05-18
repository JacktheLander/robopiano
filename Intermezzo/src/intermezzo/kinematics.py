from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from intermezzo.constants import HAND_STATE_DIM


class HandKinematics(Protocol):
    """Kinematics interface used by Intermezzo magnetic correction."""

    def set_hand_state(self, hand_state: np.ndarray) -> None:
        ...

    def fingertip_xy(self) -> np.ndarray:
        ...

    def fingertip_xyz(self) -> np.ndarray:
        ...

    def solve_xy_correction(
        self,
        hand_state: np.ndarray,
        fingertip_indices: np.ndarray,
        target_xy: np.ndarray,
        weights: np.ndarray,
        *,
        damping: float,
        max_delta_q: float,
        iterations: int,
        active_joint_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        ...

    def solve_xyz_correction(
        self,
        hand_state: np.ndarray,
        fingertip_indices: np.ndarray,
        target_xyz: np.ndarray,
        weights: np.ndarray,
        *,
        damping: float,
        max_delta_q: float,
        iterations: int,
        active_joint_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        ...


@dataclass
class FakeHandKinematics:
    """Simple 10-fingertip kinematics for tests.

    Each fingertip is controlled by a private pair of hand-state coordinates, so
    active fingertip corrections do not move inactive fingertips.
    Fingertip order is right thumb..little, then left thumb..little.
    """

    hand_state: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.hand_state is None:
            self.hand_state = np.zeros((HAND_STATE_DIM,), dtype=np.float32)
        self._pairs = [(2 * i, 2 * i + 1) for i in range(5)] + [(23 + 2 * i, 23 + 2 * i + 1) for i in range(5)]

    def set_hand_state(self, hand_state: np.ndarray) -> None:
        values = np.asarray(hand_state, dtype=np.float32).reshape(-1)
        if values.size < HAND_STATE_DIM:
            raise ValueError(f"hand_state must contain at least {HAND_STATE_DIM} values, got {values.size}")
        self.hand_state = values[:HAND_STATE_DIM].astype(np.float32, copy=True)

    def fingertip_xy(self) -> np.ndarray:
        return self.fingertip_xyz()[:, :2]

    def fingertip_xyz(self) -> np.ndarray:
        q = np.asarray(self.hand_state, dtype=np.float32).reshape(-1)
        out = np.zeros((10, 3), dtype=np.float32)
        for i, (x_idx, y_idx) in enumerate(self._pairs):
            out[i] = [q[x_idx], q[y_idx], q[y_idx]]
        return out

    def solve_xy_correction(
        self,
        hand_state: np.ndarray,
        fingertip_indices: np.ndarray,
        target_xy: np.ndarray,
        weights: np.ndarray,
        *,
        damping: float,
        max_delta_q: float,
        iterations: int,
        active_joint_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        del damping, iterations, active_joint_indices
        q = np.asarray(hand_state, dtype=np.float32).reshape(-1)[:HAND_STATE_DIM].copy()
        fingers = np.asarray(fingertip_indices, dtype=np.int64).reshape(-1)
        targets = np.asarray(target_xy, dtype=np.float32).reshape(-1, 2)
        gains = np.asarray(weights, dtype=np.float32).reshape(-1)
        for finger, target, gain in zip(fingers, targets, gains):
            if finger < 0 or finger >= len(self._pairs) or gain <= 0.0:
                continue
            x_idx, y_idx = self._pairs[int(finger)]
            current = np.asarray([q[x_idx], q[y_idx]], dtype=np.float32)
            delta = (target - current) * float(np.clip(gain, 0.0, 1.0))
            norm = float(np.linalg.norm(delta))
            limit = max(float(max_delta_q), 0.0)
            if norm > limit > 0.0:
                delta *= limit / norm
            q[x_idx] += float(delta[0])
            q[y_idx] += float(delta[1])
        self.set_hand_state(q)
        return q.astype(np.float32)

    def solve_xyz_correction(
        self,
        hand_state: np.ndarray,
        fingertip_indices: np.ndarray,
        target_xyz: np.ndarray,
        weights: np.ndarray,
        *,
        damping: float,
        max_delta_q: float,
        iterations: int,
        active_joint_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        del damping, iterations, active_joint_indices
        q = np.asarray(hand_state, dtype=np.float32).reshape(-1)[:HAND_STATE_DIM].copy()
        fingers = np.asarray(fingertip_indices, dtype=np.int64).reshape(-1)
        targets = np.asarray(target_xyz, dtype=np.float32).reshape(-1, 3)
        gains = np.asarray(weights, dtype=np.float32).reshape(-1)
        for finger, target, gain in zip(fingers, targets, gains):
            if finger < 0 or finger >= len(self._pairs) or gain <= 0.0:
                continue
            x_idx, y_idx = self._pairs[int(finger)]
            current = np.asarray([q[x_idx], q[y_idx], q[y_idx]], dtype=np.float32)
            delta = (target - current) * float(np.clip(gain, 0.0, 1.0))
            norm = float(np.linalg.norm(delta))
            limit = max(float(max_delta_q), 0.0)
            if norm > limit > 0.0:
                delta *= limit / norm
            q[x_idx] += float(delta[0])
            q[y_idx] += float(delta[1] + delta[2])
        self.set_hand_state(q)
        return q.astype(np.float32)


class RoboPianistHandKinematics:
    """RoboPianist-backed kinematics for reduced 46-D hand states."""

    def __init__(self, task: Any, physics: Any) -> None:
        self.task = task
        self.physics = physics
        self.joints = self._hand_joint_handles()
        self.sites = self._fingertip_site_handles()
        if len(self.joints) < HAND_STATE_DIM:
            raise ValueError(f"RoboPianist task exposes {len(self.joints)} hand joints, expected at least {HAND_STATE_DIM}")
        if len(self.sites) < 10:
            raise ValueError(f"RoboPianist task exposes {len(self.sites)} fingertip sites, expected at least 10")

    def _hand_joint_handles(self) -> list[Any]:
        joints: list[Any] = []
        for hand_name in ("right_hand", "left_hand"):
            hand = getattr(self.task, hand_name, None)
            hand_joints = getattr(hand, "joints", None)
            if hand_joints is not None:
                joints.extend(list(hand_joints))
        return joints

    def _fingertip_site_handles(self) -> list[Any]:
        sites: list[Any] = []
        for hand_name in ("right_hand", "left_hand"):
            hand = getattr(self.task, hand_name, None)
            fingertip_sites = getattr(hand, "fingertip_sites", None)
            if fingertip_sites is not None:
                sites.extend(list(fingertip_sites))
        return sites

    def set_hand_state(self, hand_state: np.ndarray) -> None:
        values = np.asarray(hand_state, dtype=np.float32).reshape(-1)
        if values.size < len(self.joints):
            raise ValueError(f"hand_state has {values.size} values but environment expects {len(self.joints)} joints")
        for joint, value in zip(self.joints, values[: len(self.joints)]):
            self.physics.bind(joint).qpos = float(value)
        if hasattr(self.physics, "forward"):
            self.physics.forward()

    def fingertip_xy(self) -> np.ndarray:
        return self.fingertip_xyz()[:, :2]

    def fingertip_xyz(self) -> np.ndarray:
        positions = np.asarray(self.physics.bind(self.sites).xpos, dtype=np.float32)
        return np.ascontiguousarray(positions[:10, :3], dtype=np.float32)

    def solve_xy_correction(
        self,
        hand_state: np.ndarray,
        fingertip_indices: np.ndarray,
        target_xy: np.ndarray,
        weights: np.ndarray,
        *,
        damping: float,
        max_delta_q: float,
        iterations: int,
        active_joint_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        return self._solve_cartesian_correction(
            hand_state,
            fingertip_indices,
            target_xy,
            weights,
            dimensions=2,
            damping=damping,
            max_delta_q=max_delta_q,
            iterations=iterations,
            active_joint_indices=active_joint_indices,
        )

    def solve_xyz_correction(
        self,
        hand_state: np.ndarray,
        fingertip_indices: np.ndarray,
        target_xyz: np.ndarray,
        weights: np.ndarray,
        *,
        damping: float,
        max_delta_q: float,
        iterations: int,
        active_joint_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        return self._solve_cartesian_correction(
            hand_state,
            fingertip_indices,
            target_xyz,
            weights,
            dimensions=3,
            damping=damping,
            max_delta_q=max_delta_q,
            iterations=iterations,
            active_joint_indices=active_joint_indices,
        )

    def _solve_cartesian_correction(
        self,
        hand_state: np.ndarray,
        fingertip_indices: np.ndarray,
        target_positions: np.ndarray,
        weights: np.ndarray,
        *,
        dimensions: int,
        damping: float,
        max_delta_q: float,
        iterations: int,
        active_joint_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        dims = int(dimensions)
        if dims not in (2, 3):
            raise ValueError(f"dimensions must be 2 or 3, got {dimensions}")
        q = np.asarray(hand_state, dtype=np.float32).reshape(-1)[: len(self.joints)].copy()
        fingers = np.asarray(fingertip_indices, dtype=np.int64).reshape(-1)
        targets = np.asarray(target_positions, dtype=np.float32).reshape(-1, dims)
        gains = np.asarray(weights, dtype=np.float32).reshape(-1)
        active = _normalize_active_joint_indices(active_joint_indices, q.size)
        for _ in range(max(int(iterations), 1)):
            self.set_hand_state(q)
            current = (self.fingertip_xy() if dims == 2 else self.fingertip_xyz())[fingers]
            error = ((targets - current) * gains[:, None]).reshape(-1)
            if not np.any(np.isfinite(error)) or float(np.linalg.norm(error)) <= 1e-7:
                break
            jac = self._finite_difference_cartesian_jacobian(q, fingers, dimensions=dims, active_joint_indices=active)
            lhs = jac @ jac.T + (float(damping) ** 2) * np.eye(jac.shape[0], dtype=np.float32)
            try:
                delta_active = jac.T @ np.linalg.solve(lhs, error.astype(np.float32))
                delta = np.zeros_like(q)
                delta[active] = delta_active
            except np.linalg.LinAlgError:
                delta = np.zeros_like(q)
            norm = float(np.linalg.norm(delta))
            limit = max(float(max_delta_q), 0.0)
            if norm > limit > 0.0:
                delta *= limit / norm
            q = (q + delta.astype(np.float32)).astype(np.float32)
        self.set_hand_state(q)
        return q.astype(np.float32)

    def _finite_difference_cartesian_jacobian(
        self,
        q: np.ndarray,
        fingers: np.ndarray,
        *,
        dimensions: int,
        active_joint_indices: np.ndarray,
    ) -> np.ndarray:
        eps = 1e-4
        base = q.astype(np.float32, copy=True)
        self.set_hand_state(base)
        base_positions = (self.fingertip_xy() if int(dimensions) == 2 else self.fingertip_xyz())[fingers].reshape(-1)
        active = np.asarray(active_joint_indices, dtype=np.int64).reshape(-1)
        jac = np.zeros((base_positions.size, active.size), dtype=np.float32)
        for col, joint_index in enumerate(active):
            perturbed = base.copy()
            perturbed[int(joint_index)] += eps
            self.set_hand_state(perturbed)
            positions = (self.fingertip_xy() if int(dimensions) == 2 else self.fingertip_xyz())[fingers].reshape(-1)
            jac[:, col] = (positions - base_positions) / eps
        self.set_hand_state(base)
        return jac


def _normalize_active_joint_indices(active_joint_indices: np.ndarray | None, size: int) -> np.ndarray:
    if active_joint_indices is None:
        return np.arange(int(size), dtype=np.int64)
    active = np.asarray(active_joint_indices, dtype=np.int64).reshape(-1)
    active = active[(active >= 0) & (active < int(size))]
    if active.size == 0:
        return np.arange(int(size), dtype=np.int64)
    return np.unique(active).astype(np.int64)