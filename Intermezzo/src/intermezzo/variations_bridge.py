from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from intermezzo.constants import HAND_STATE_DIM
from intermezzo.keys import validate_target_keys
from intermezzo.midi import ensure_variations_paths


@dataclass
class VariationsDiffusionPredictor:
    checkpoint_path: Path
    loaded_model: Any

    @property
    def device(self) -> Any:
        return self.loaded_model.device

    @property
    def diffusion_steps(self) -> int:
        return int(self.loaded_model.diffusion_steps)

    def predict(self, target_keys: np.ndarray, *, batch_size: int = 256) -> np.ndarray:
        keys = validate_target_keys(target_keys)
        out = self.loaded_model.predict_hand_states(keys, batch_size=int(batch_size))
        values = np.asarray(out, dtype=np.float32)
        if values.shape != (keys.shape[0], HAND_STATE_DIM):
            raise ValueError(
                "Variations diffusion returned an unexpected shape: "
                f"expected {(keys.shape[0], HAND_STATE_DIM)}, got {values.shape}"
            )
        return values

    def __call__(self, target_keys: np.ndarray) -> np.ndarray:
        return self.predict(target_keys)


def load_variations_diffusion_predictor(
    checkpoint_path: str | Path,
    *,
    device: str = "auto",
    diffusion_steps: int | None = None,
) -> VariationsDiffusionPredictor:
    ensure_variations_paths()
    from simulate.model_loader import load_simulation_model

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Variations diffusion checkpoint not found: {checkpoint}")
    loaded = load_simulation_model(
        str(checkpoint),
        "diffusion",
        device=str(device),
        diffusion_steps=diffusion_steps,
    )
    if getattr(loaded, "model_type", None) != "diffusion":
        raise ValueError("Intermezzo requires a Variations diffusion model, not another Variations model type.")
    return VariationsDiffusionPredictor(checkpoint_path=checkpoint, loaded_model=loaded)
