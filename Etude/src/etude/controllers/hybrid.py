from __future__ import annotations

from typing import Any

import numpy as np
import torch

from etude.controllers.base import TrajectoryFollower
from etude.controllers.pd import PDController
from etude.data.feature_builder import FeatureSpec, build_tracking_features
from etude.robopianist.state_mapping import StateMapping


class HybridPDResidualController(TrajectoryFollower):
    """PD controller plus a learned residual action model."""

    def __init__(
        self,
        mapping: StateMapping,
        residual_model: torch.nn.Module,
        pd: PDController | None = None,
        feature_spec: FeatureSpec | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.mapping = mapping
        self.pd = pd or PDController(mapping)
        self.residual_model = residual_model.to(device)
        self.residual_model.eval()
        self.feature_spec = feature_spec or FeatureSpec()
        self.device = torch.device(device)
        self.q_ref: np.ndarray | None = None
        self.qdot_ref: np.ndarray | None = None
        self.previous_action = mapping.zero_action()

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

    def act(self, obs: dict[str, np.ndarray], t: int) -> np.ndarray:
        if self.q_ref is None or self.qdot_ref is None:
            raise RuntimeError("HybridPDResidualController.reset must be called before act")
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
        with torch.no_grad():
            residual = (
                self.residual_model(torch.from_numpy(features).to(self.device).float().unsqueeze(0))
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
            )
        action = self.mapping.clip_action(pd_action + residual.astype(np.float32))
        self.previous_action = action
        return action
