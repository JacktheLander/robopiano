from __future__ import annotations

from typing import Any

import numpy as np
import torch

from etude.controllers.base import TrajectoryFollower
from etude.controllers.pd import PDController
from etude.controllers.temporal_context import TemporalContextBuffer
from etude.data.feature_builder import FeatureSpec, build_tracking_features
from etude.robopianist.state_mapping import StateMapping


class TemporalResidualController(TrajectoryFollower):
    """PD controller plus a temporal residual model over recent feature history."""

    def __init__(
        self,
        mapping: StateMapping,
        temporal_model: torch.nn.Module,
        *,
        pd: PDController | None = None,
        feature_spec: FeatureSpec | None = None,
        history_steps: int = 16,
        include_previous_residual: bool = True,
        reset_hidden_on_episode: bool = True,
        residual_scale: float = 1.0,
        residual_clip: float = 1.0,
        device: str | torch.device = "cpu",
        use_hidden_state: bool = True,
    ) -> None:
        self.mapping = mapping
        self.pd = pd or PDController(mapping)
        self.temporal_model = temporal_model.to(device)
        self.temporal_model.eval()
        self.feature_spec = feature_spec or FeatureSpec()
        self.context = TemporalContextBuffer(
            history_steps,
            store_actions=True,
            store_residuals=include_previous_residual,
        )
        self.reset_hidden_on_episode = bool(reset_hidden_on_episode)
        self.residual_scale = float(residual_scale)
        self.residual_clip = float(residual_clip)
        self.device = torch.device(device)
        self.use_hidden_state = bool(use_hidden_state)
        self.hidden_state: Any = None
        self.q_ref: np.ndarray | None = None
        self.qdot_ref: np.ndarray | None = None
        self.metadata: dict[str, Any] = {}
        self.previous_action = mapping.zero_action()
        self.previous_residual = np.zeros(mapping.action_dim, dtype=np.float32)

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
        self.context.reset()
        self.previous_action = self.mapping.zero_action()
        self.previous_residual = np.zeros(self.mapping.action_dim, dtype=np.float32)
        if self.reset_hidden_on_episode:
            self.hidden_state = None
        self._reset_model_state_if_available()

    def act(self, obs: dict[str, np.ndarray], t: int) -> np.ndarray:
        if self.q_ref is None or self.qdot_ref is None:
            raise RuntimeError("TemporalResidualController.reset must be called before act")

        q = np.asarray(obs["q"], dtype=np.float32).reshape(-1)
        qdot = np.asarray(obs["qdot"], dtype=np.float32).reshape(-1)
        if q.shape != (46,) or qdot.shape != (46,):
            raise ValueError(f"obs q/qdot must have shape [46], got {q.shape}, {qdot.shape}")

        pd_action = self.pd.act(obs, t)
        features = build_tracking_features(
            q=q,
            qdot=qdot,
            q_ref=self.q_ref,
            qdot_ref=self.qdot_ref,
            t=t,
            previous_action=self.previous_action,
            target_keys=obs.get("target_keys", self.metadata.get("target_keys")),
            fingertips=obs.get("fingertips", self.metadata.get("fingertips")),
            spec=self.feature_spec,
        )

        self.context.append(
            features,
            action=self.previous_action,
            residual=self.previous_residual,
        )
        residual = self._run_temporal_model(self.context.feature_history())
        residual = np.clip(self.residual_scale * residual, -self.residual_clip, self.residual_clip)
        action = self.mapping.clip_action(pd_action + residual)

        self.previous_action = action
        self.previous_residual = residual
        return action

    def _run_temporal_model(self, feature_history: np.ndarray) -> np.ndarray:
        sequence = torch.from_numpy(feature_history).to(self.device).float().unsqueeze(0)
        with torch.no_grad():
            raw_output = self._call_model(sequence)
        residual = self._extract_residual(raw_output)
        if residual.shape != (self.mapping.action_dim,):
            raise ValueError(
                f"Residual action must have shape [{self.mapping.action_dim}], got {residual.shape}"
            )
        return residual

    def _call_model(self, sequence: torch.Tensor) -> Any:
        if self.use_hidden_state:
            try:
                output = self.temporal_model(sequence, self.hidden_state)
            except TypeError:
                output = self.temporal_model(sequence)
        else:
            output = self.temporal_model(sequence)
        if isinstance(output, tuple) and len(output) == 2:
            output, maybe_hidden = output
            self.hidden_state = maybe_hidden
        return output

    def _extract_residual(self, model_output: Any) -> np.ndarray:
        output = model_output
        if isinstance(output, dict):
            output = output.get("residual")
        if output is None:
            raise ValueError("Temporal model did not return a residual output")
        if torch.is_tensor(output):
            array = output.detach().cpu().numpy()
        else:
            array = np.asarray(output, dtype=np.float32)
        residual = self._coerce_output_shape(array.astype(np.float32))
        return residual

    def _coerce_output_shape(self, output: np.ndarray) -> np.ndarray:
        if output.ndim == 1:
            return output.reshape(-1)
        if output.ndim == 2:
            if output.shape[0] != 1:
                raise ValueError(f"Unsupported temporal residual shape {output.shape}")
            return output[0].reshape(-1)
        if output.ndim == 3:
            return output[-1, -1].reshape(-1)
        raise ValueError(f"Unsupported temporal residual shape {output.shape}")

    def _reset_model_state_if_available(self) -> None:
        for name in ("reset_state", "reset_hidden_state", "reset_hidden"):
            fn = getattr(self.temporal_model, name, None)
            if callable(fn):
                fn()
                break
