from __future__ import annotations

import importlib
from typing import Any, Callable

import numpy as np
import torch

from etude.controllers.base import TrajectoryFollower
from etude.controllers.pd import PDController
from etude.robopianist.state_mapping import StateMapping


FeatureBlock = Callable[..., np.ndarray]


class KeyAwareResidualController(TrajectoryFollower):
    """PD controller plus feature-block-driven residual predictions."""

    def __init__(
        self,
        mapping: StateMapping,
        residual_model: torch.nn.Module,
        pd: PDController | None = None,
        feature_block_paths: list[str] | None = None,
        feature_block_kwargs: dict[str, dict[str, Any]] | None = None,
        device: str | torch.device = "cpu",
        residual_scale: float = 1.0,
        residual_clip: float = 1.0,
    ) -> None:
        self.mapping = mapping
        self.pd = pd or PDController(mapping)
        self.residual_model = residual_model.to(device)
        self.residual_model.eval()
        self.device = torch.device(device)
        self.residual_scale = float(residual_scale)
        self.residual_clip = float(residual_clip)
        self.q_ref: np.ndarray | None = None
        self.qdot_ref: np.ndarray | None = None
        self.metadata: dict[str, Any] = {}
        self.previous_action = mapping.zero_action()
        self.last_press_intent: np.ndarray | None = None
        self.feature_block_kwargs = feature_block_kwargs or {}
        paths = feature_block_paths or ["etude.features.key_blocks:build_key_features"]
        self.feature_blocks = [(path, _resolve_feature_block(path)) for path in paths]
        self._safety_fn = _resolve_residual_safety()

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
        self.last_press_intent = None

    def act(self, obs: dict[str, np.ndarray], t: int) -> np.ndarray:
        if self.q_ref is None or self.qdot_ref is None:
            raise RuntimeError("KeyAwareResidualController.reset must be called before act")

        q = np.asarray(obs["q"], dtype=np.float32).reshape(-1)
        qdot = np.asarray(obs["qdot"], dtype=np.float32).reshape(-1)
        if q.shape != (46,) or qdot.shape != (46,):
            raise ValueError(f"obs q/qdot must have shape [46], got {q.shape}, {qdot.shape}")

        pd_action = self.pd.act(obs, t)
        features = self._build_features(obs, t, q, qdot, pd_action)
        residual_raw, press_intent = self._run_model(features)
        residual = self._coerce_residual(residual_raw)
        action = self._postprocess_action(pd_action, residual, press_intent, obs, t)
        self.previous_action = action
        self.last_press_intent = press_intent
        return action

    def _build_features(
        self,
        obs: dict[str, np.ndarray],
        t: int,
        q: np.ndarray,
        qdot: np.ndarray,
        pd_action: np.ndarray,
    ) -> np.ndarray:
        idx = int(np.clip(t, 0, self.q_ref.shape[0] - 1))
        base = [
            q,
            qdot,
            self.q_ref[idx] - q,
            self.qdot_ref[idx] - qdot,
            pd_action.astype(np.float32),
            self.previous_action.astype(np.float32),
        ]
        block_values: list[np.ndarray] = []
        key_state = obs.get("key_state", obs.get("keys", obs.get("piano_keys")))
        for path, block in self.feature_blocks:
            kwargs = self.feature_block_kwargs.get(path, {})
            value = block(
                q=q,
                qdot=qdot,
                q_ref=self.q_ref,
                qdot_ref=self.qdot_ref,
                pd_action=pd_action,
                previous_action=self.previous_action,
                target_keys=obs.get("target_keys", self.metadata.get("target_keys")),
                key_state=key_state,
                metadata=self.metadata,
                obs=obs,
                t=t,
                **kwargs,
            )
            block_values.append(np.asarray(value, dtype=np.float32).reshape(-1))
        return np.concatenate(base + block_values).astype(np.float32)

    def _run_model(self, features: np.ndarray) -> tuple[Any, np.ndarray | None]:
        with torch.no_grad():
            output = self.residual_model(
                torch.from_numpy(features).to(self.device).float().unsqueeze(0)
            )
        residual_out: Any = output
        press_intent: np.ndarray | None = None
        if isinstance(output, dict):
            residual_out = output.get("residual", output.get("delta_u", output.get("action_residual")))
            press_intent = _to_numpy(output.get("press_intent", output.get("press_logits")))
        elif isinstance(output, (tuple, list)):
            residual_out = output[0]
            if len(output) > 1:
                press_intent = _to_numpy(output[1])
        return residual_out, press_intent

    def _coerce_residual(self, residual_raw: Any) -> np.ndarray:
        residual = _to_numpy(residual_raw)
        if residual is None:
            raise ValueError("Residual model did not return a residual output")
        residual = residual.reshape(-1).astype(np.float32)
        if residual.shape != (self.mapping.action_dim,):
            raise ValueError(
                f"Residual action must have shape [{self.mapping.action_dim}], got {residual.shape}"
            )
        return self.residual_scale * residual

    def _postprocess_action(
        self,
        pd_action: np.ndarray,
        residual: np.ndarray,
        press_intent: np.ndarray | None,
        obs: dict[str, np.ndarray],
        t: int,
    ) -> np.ndarray:
        if self._safety_fn is not None:
            safe = self._safety_fn(
                mapping=self.mapping,
                pd_action=pd_action,
                residual=residual,
                press_intent=press_intent,
                obs=obs,
                metadata=self.metadata,
                t=t,
            )
            return np.asarray(safe, dtype=np.float32).reshape(-1)
        clipped_residual = np.clip(residual, -self.residual_clip, self.residual_clip)
        return self.mapping.clip_action(pd_action + clipped_residual)


def _resolve_feature_block(path: str) -> FeatureBlock:
    if ":" not in path:
        raise ValueError(f"Feature block path must be module:function, got {path}")
    module_name, func_name = path.split(":", 1)
    module = importlib.import_module(module_name)
    block = getattr(module, func_name)
    if not callable(block):
        raise TypeError(f"Feature block {path} is not callable")
    return block


def _resolve_residual_safety() -> Callable[..., np.ndarray] | None:
    try:
        module = importlib.import_module("etude.controllers.residual_safety")
    except Exception:
        return None
    for name in ("postprocess_residual", "apply_residual_safety", "safe_residual_action"):
        fn = getattr(module, name, None)
        if callable(fn):
            return fn
    return None


def _to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.astype(np.float32)
    if torch.is_tensor(value):
        return value.squeeze(0).detach().cpu().numpy().astype(np.float32)
    return np.asarray(value, dtype=np.float32)
