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
        q = np.asarray(self.hand_state, dtype=np.float32).reshape(-1)
        out = np.zeros((10, 2), dtype=np.float32)
        for i, (x_idx, y_idx) in enumerate(self._pairs):
            out[i] = [q[x_idx], q[y_idx]]
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
    ) -> np.ndarray:
        del damping, iterations
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
        positions = np.asarray(self.physics.bind(self.sites).xpos, dtype=np.float32)
        return np.ascontiguousarray(positions[:10, :2], dtype=np.float32)

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
    ) -> np.ndarray:
        q = np.asarray(hand_state, dtype=np.float32).reshape(-1)[: len(self.joints)].copy()
        fingers = np.asarray(fingertip_indices, dtype=np.int64).reshape(-1)
        targets = np.asarray(target_xy, dtype=np.float32).reshape(-1, 2)
        gains = np.asarray(weights, dtype=np.float32).reshape(-1)
        for _ in range(max(int(iterations), 1)):
            self.set_hand_state(q)
            current = self.fingertip_xy()[fingers]
            error = ((targets - current) * gains[:, None]).reshape(-1)
            if not np.any(np.isfinite(error)) or float(np.linalg.norm(error)) <= 1e-7:
                break
            jac = self._finite_difference_xy_jacobian(q, fingers)
            lhs = jac @ jac.T + (float(damping) ** 2) * np.eye(jac.shape[0], dtype=np.float32)
            try:
                delta = jac.T @ np.linalg.solve(lhs, error.astype(np.float32))
            except np.linalg.LinAlgError:
                delta = np.zeros_like(q)
            norm = float(np.linalg.norm(delta))
            limit = max(float(max_delta_q), 0.0)
            if norm > limit > 0.0:
                delta *= limit / norm
            q = (q + delta.astype(np.float32)).astype(np.float32)
        self.set_hand_state(q)
        return q.astype(np.float32)

    def _finite_difference_xy_jacobian(self, q: np.ndarray, fingers: np.ndarray) -> np.ndarray:
        eps = 1e-4
        base = q.astype(np.float32, copy=True)
        self.set_hand_state(base)
        base_xy = self.fingertip_xy()[fingers].reshape(-1)
        jac = np.zeros((base_xy.size, base.size), dtype=np.float32)
        for j in range(base.size):
            perturbed = base.copy()
            perturbed[j] += eps
            self.set_hand_state(perturbed)
            xy = self.fingertip_xy()[fingers].reshape(-1)
            jac[:, j] = (xy - base_xy) / eps
        self.set_hand_state(base)
        return jac
