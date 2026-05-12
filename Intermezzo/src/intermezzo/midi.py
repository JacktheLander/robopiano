from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np

from intermezzo.keys import validate_target_keys


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def ensure_variations_paths() -> Path:
    # Runtime imports should not create pycache files in sibling project folders.
    sys.dont_write_bytecode = True
    repo = _repo_root()
    for path in (repo / "Variations" / "src", repo / "Variations", repo / "partita" / "src", repo):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    return repo


def load_target_keys_from_midi(
    midi_path: str | Path,
    *,
    control_timestep: float = 0.05,
    max_steps: int | None = None,
    max_duration_s: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    ensure_variations_paths()
    from simulate.midi_keysets import midi_to_target_key_roll

    target_keys, meta = midi_to_target_key_roll(
        midi_path,
        control_timestep=float(control_timestep),
        max_steps=max_steps,
        max_duration_s=max_duration_s,
    )
    return validate_target_keys(target_keys), dict(meta)
