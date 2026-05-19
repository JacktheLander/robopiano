from __future__ import annotations

from typing import Any

import numpy as np
import torch

from etude.controllers.base import TrajectoryFollower
from etude.controllers.pd import PDController
from etude.controllers.residual_safety import (
    ResidualSafetyConfig,
    ResidualSafetyProcessor,
    residual_diagnostics,
    resolve_phase,
)
from etude.data.feature_builder import FeatureSpec, build_tracking_features
from etude.robopianist.state_mapping import StateMapping


class SafeHybridPDResidualController(TrajectoryFollower):
    """PD controller plus a safety-processed learned residual action."""

    def __init__(
        self,
        mapping: StateMapping,
        residual_model: torch.nn.Module,
        pd: PDController | None = None,
        feature_spec: FeatureSpec | None = None,
        safety: ResidualSafetyConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.mapping = mapping
        self.pd = pd or PDController(mapping)
        self.residual_model = residual_model.to(device)
        self.residual_model.eval()
        self.feature_spec = feature_spec or FeatureSpec()
        self.safety = ResidualSafetyProcessor(safety)
        self.device = torch.device(device)
        self.hidden_state: torch.Tensor | None = None
        self.q_ref: np.ndarray | None = None
        self.qdot_ref: np.ndarray | None = None
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
        self.previous_action = self.mapping.zero_action()
        self.hidden_state = None
        self.safety.reset()
        self._metadata = metadata or {}
        self._last_diagnostics = {}

    def act(self, obs: dict[str, np.ndarray], t: int) -> np.ndarray:
        if self.q_ref is None or self.qdot_ref is None:
            raise RuntimeError("SafeHybridPDResidualController.reset must be called before act")
        pd_action = self.pd.act(obs, t)
        features = build_tracking_features(
            q=obs["q"],
            qdot=obs["qdot"],
            q_ref=self.q_ref,
            qdot_ref=self.qdot_ref,
            t=t,
            previous_action=self.previous_action,
            target_keys=obs.get("target_keys"),
            fingertips=obs.get("fingertips"),
            spec=self.feature_spec,
        )
        raw_residual = self._predict_residual(features)
        phase = resolve_phase(obs, self._metadata, t)
        processed_residual, processing_diag = self.safety.process(raw_residual, phase=phase)
        unclipped_action = pd_action + processed_residual
        action = self.mapping.clip_action(unclipped_action)
        self._last_diagnostics = residual_diagnostics(
            raw_residual=raw_residual,
            pre_clip_residual=raw_residual * np.float32(self.safety.config.scale * processing_diag["control/phase_gate"]),
            processed_residual=processed_residual,
            final_action_unclipped=unclipped_action,
            final_action=action,
            phase_gate=processing_diag["control/phase_gate"],
        )
        self.previous_action = action
        return action

    def diagnostics(self) -> dict[str, float]:
        return dict(self._last_diagnostics)

    def _predict_residual(self, features: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(features).to(self.device).float().unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            if hasattr(self.residual_model, "gru"):
                output = self.residual_model(tensor, self.hidden_state)
            else:
                output = self.residual_model(tensor)
            if isinstance(output, tuple):
                residual_tensor, self.hidden_state = output
            else:
                residual_tensor = output
                self.hidden_state = None
        return residual_tensor.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
