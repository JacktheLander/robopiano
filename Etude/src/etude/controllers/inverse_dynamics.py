from __future__ import annotations

from typing import Any

import numpy as np
import torch

from etude.controllers.base import TrajectoryFollower
from etude.controllers.pd import PDController
from etude.core.controller_output import ControllerOutput
from etude.features.inverse_dynamics_blocks import (
    InverseDynamicsFeatureSpec,
    build_inverse_dynamics_features,
)
from etude.robopianist.state_mapping import StateMapping


class InverseDynamicsController(TrajectoryFollower):
    """Model-driven controller that predicts either full actions or PD residuals."""

    def __init__(
        self,
        mapping: StateMapping,
        model: torch.nn.Module,
        *,
        output_mode: str = "full_action",
        pd: PDController | None = None,
        feature_spec: InverseDynamicsFeatureSpec | dict[str, Any] | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        if output_mode not in {"full_action", "pd_residual"}:
            raise ValueError(f"Unsupported inverse dynamics output_mode: {output_mode}")
        self.mapping = mapping
        self.model = model.to(device)
        self.model.eval()
        self.output_mode = output_mode
        self.pd = pd or PDController(mapping)
        self.feature_spec = feature_spec if feature_spec is not None else InverseDynamicsFeatureSpec()
        self.device = torch.device(device)
        self.q_ref: np.ndarray | None = None
        self.qdot_ref: np.ndarray | None = None
        self.metadata: dict[str, Any] = {}
        self.previous_action = mapping.zero_action()
        self.last_output: ControllerOutput | None = None
        self._last_diagnostics: dict[str, Any] = {}

    def reset(
        self,
        q_ref: np.ndarray,
        qdot_ref: np.ndarray | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.pd.reset(q_ref, qdot_ref, metadata)
        self.q_ref = self.pd.q_ref
        self.qdot_ref = self.pd.qdot_ref
        self.metadata = dict(metadata or {})
        self.previous_action = self.mapping.zero_action()
        self.last_output = None
        self._last_diagnostics = {}

    def act(self, obs: dict[str, np.ndarray], t: int) -> np.ndarray:
        if self.q_ref is None or self.qdot_ref is None:
            raise RuntimeError("InverseDynamicsController.reset must be called before act")

        q = np.asarray(obs["q"], dtype=np.float32).reshape(-1)
        qdot = np.asarray(obs["qdot"], dtype=np.float32).reshape(-1)
        if q.shape != (46,) or qdot.shape != (46,):
            raise ValueError(f"obs q/qdot must have shape [46], got {q.shape}, {qdot.shape}")

        pd_action = self.pd.act(obs, t) if self.output_mode == "pd_residual" else self.mapping.zero_action()
        features = self._build_features(obs, t, q, qdot)
        model_output = self._run_model(features)
        predicted_action = self._coerce_action(model_output)

        if self.output_mode == "pd_residual":
            residual = predicted_action
            unclipped_action = pd_action + residual
            action = self.mapping.clip_action(unclipped_action)
        else:
            residual = None
            unclipped_action = predicted_action
            action = self.mapping.clip_action(predicted_action)

        diagnostics = {
            "control/output_mode": self.output_mode,
            "control/feature_dim": int(features.size),
            "control/pd_action_norm": float(np.linalg.norm(pd_action)),
            "control/model_action_norm": float(np.linalg.norm(predicted_action)),
            "control/final_action_norm": float(np.linalg.norm(action)),
            "control/action_was_clipped": bool(not np.allclose(unclipped_action, action)),
        }
        if residual is not None:
            diagnostics["control/residual_norm"] = float(np.linalg.norm(residual))

        self.last_output = ControllerOutput(action=action, residual=residual, diagnostics=diagnostics)
        self._last_diagnostics = diagnostics
        self.previous_action = action
        return action

    def diagnostics(self) -> dict[str, Any]:
        return dict(self._last_diagnostics)

    def _build_features(
        self,
        obs: dict[str, np.ndarray],
        t: int,
        q: np.ndarray,
        qdot: np.ndarray,
    ) -> np.ndarray:
        return build_inverse_dynamics_features(
            q=q,
            qdot=qdot,
            q_ref=self.q_ref,
            t=t,
            fingertips=obs.get("fingertips"),
            fingertip_ref=self.metadata.get("fingertip_ref"),
            target_keys=obs.get("target_keys", self.metadata.get("target_keys")),
            previous_action=obs.get("previous_action", self.previous_action),
            spec=self.feature_spec,
        )

    def _run_model(self, features: np.ndarray) -> Any:
        tensor = torch.from_numpy(features).to(self.device).float().unsqueeze(0)
        with torch.no_grad():
            return self.model(tensor)

    def _coerce_action(self, model_output: Any) -> np.ndarray:
        value = model_output
        if isinstance(model_output, dict):
            for key in ("action", "residual", "delta_u", "action_residual", "u"):
                if key in model_output:
                    value = model_output[key]
                    break
        elif isinstance(model_output, (tuple, list)):
            if not model_output:
                raise ValueError("Inverse dynamics model returned an empty sequence")
            value = model_output[0]

        if torch.is_tensor(value):
            action = value.squeeze(0).detach().cpu().numpy().astype(np.float32)
        else:
            action = np.asarray(value, dtype=np.float32)
        action = action.reshape(-1)
        if action.shape != (self.mapping.action_dim,):
            raise ValueError(f"Inverse dynamics output must have shape [{self.mapping.action_dim}], got {action.shape}")
        return action
