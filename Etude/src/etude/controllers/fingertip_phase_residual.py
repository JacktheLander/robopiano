from __future__ import annotations

from typing import Any

import numpy as np
import torch

from etude.controllers.base import TrajectoryFollower
from etude.controllers.pd import PDController
from etude.features.fingertip_phase_blocks import (
    FingertipFeatureSpec,
    PhaseFeatureSpec,
    build_fingertip_phase_features,
    resolve_phase_state,
)
from etude.robopianist.state_mapping import StateMapping


class FingertipPhaseResidualController(TrajectoryFollower):
    """PD controller plus fingertip/phase-conditioned residual actions."""

    def __init__(
        self,
        mapping: StateMapping,
        residual_model: torch.nn.Module,
        pd: PDController | None = None,
        fingertip_spec: FingertipFeatureSpec | None = None,
        phase_spec: PhaseFeatureSpec | None = None,
        phase_gain: float = 1.0,
        device: str | torch.device = "cpu",
        residual_limit: float | None = None,
    ) -> None:
        self.mapping = mapping
        self.pd = pd or PDController(mapping)
        self.residual_model = residual_model.to(device)
        self.residual_model.eval()
        self.fingertip_spec = fingertip_spec or FingertipFeatureSpec(allow_missing=True)
        self.phase_spec = phase_spec or PhaseFeatureSpec(allow_missing=True)
        self.phase_gain = float(phase_gain)
        self.device = torch.device(device)
        self.residual_limit = None if residual_limit is None else float(residual_limit)
        self.q_ref: np.ndarray | None = None
        self.qdot_ref: np.ndarray | None = None
        self.metadata: dict[str, Any] = {}
        self.previous_action = mapping.zero_action()
        self._last_diagnostics: dict[str, float] = {}

    def reset(
        self,
        q_ref: np.ndarray,
        qdot_ref: np.ndarray | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.pd.reset(q_ref, qdot_ref, metadata)
        self.q_ref = self.pd.q_ref
        self.qdot_ref = self.pd.qdot_ref
        self.metadata = metadata or {}
        self.previous_action = self.mapping.zero_action()
        self._last_diagnostics = {}

    def act(self, obs: dict[str, np.ndarray], t: int) -> np.ndarray:
        if self.q_ref is None or self.qdot_ref is None:
            raise RuntimeError("FingertipPhaseResidualController.reset must be called before act")
        pd_action = self.pd.act(obs, t)
        phase_state = resolve_phase_state(
            t=t,
            metadata=self.metadata,
            plan_bundle=self.metadata.get("plan_bundle"),
            target_keys=obs.get("target_keys"),
            phase_names=self.phase_spec.phase_names,
            allow_missing=self.phase_spec.allow_missing,
            fill_value=self.phase_spec.missing_fill_value,
        )
        features = build_fingertip_phase_features(
            t=t,
            metadata=self.metadata,
            plan_bundle=self.metadata.get("plan_bundle"),
            target_keys=obs.get("target_keys"),
            current_fingertips=obs.get("fingertips"),
            desired_fingertips=_reference_timestep(self.metadata.get("desired_fingertips"), t),
            fingertip_weights=_reference_timestep(self.metadata.get("fingertip_weights"), t),
            active_finger_mask=_reference_timestep(self.metadata.get("active_finger_mask"), t),
            inactive_finger_mask=_reference_timestep(self.metadata.get("inactive_finger_mask"), t),
            fingertip_spec=self.fingertip_spec,
            phase_spec=self.phase_spec,
        )
        residual = self._predict_residual(features)
        residual = self._apply_phase_gate(residual, phase_state)
        residual = self._apply_residual_limit(residual)
        action = self.mapping.clip_action(pd_action + residual.astype(np.float32))
        self.previous_action = action
        self._last_diagnostics = {
            "phase/id": float(phase_state["phase_id"]),
            "phase/mask": float(phase_state["mask"]),
            "phase/scalar": float(phase_state["scalar"]),
            "residual/gain": float(self._phase_gate_value(phase_state)),
            "tracking/fingertip_error_l2": _fingertip_error_l2(
                obs.get("fingertips"),
                _reference_timestep(self.metadata.get("desired_fingertips"), t),
            ),
        }
        return action

    def diagnostics(self) -> dict[str, float]:
        return {**self.pd.diagnostics(), **self._last_diagnostics}

    def _predict_residual(self, features: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            residual = (
                self.residual_model(torch.from_numpy(features).to(self.device).float().unsqueeze(0))
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
            )
        return np.asarray(residual, dtype=np.float32).reshape(-1)

    def _apply_phase_gate(self, residual: np.ndarray, phase_state: dict[str, Any]) -> np.ndarray:
        gain = self._phase_gate_value(phase_state)
        return (residual * gain).astype(np.float32)

    def _phase_gate_value(self, phase_state: dict[str, Any]) -> float:
        return float(np.clip(self.phase_gain * float(phase_state["mask"]), 0.0, max(self.phase_gain, 1.0)))

    def _apply_residual_limit(self, residual: np.ndarray) -> np.ndarray:
        if self.residual_limit is None:
            return residual.astype(np.float32)
        return np.clip(residual, -self.residual_limit, self.residual_limit).astype(np.float32)


def _reference_timestep(value: Any, t: int) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        return array.reshape(1)
    if array.ndim == 1:
        return array.astype(np.float32)
    index = int(np.clip(t, 0, array.shape[0] - 1))
    return array[index].astype(np.float32)


def _fingertip_error_l2(current: np.ndarray | None, desired: np.ndarray | None) -> float:
    if current is None or desired is None:
        return 0.0
    current_array = np.asarray(current, dtype=np.float32).reshape(-1)
    desired_array = np.asarray(desired, dtype=np.float32).reshape(-1)
    if current_array.shape != desired_array.shape:
        return 0.0
    return float(np.linalg.norm(desired_array - current_array))
