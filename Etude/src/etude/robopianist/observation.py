from __future__ import annotations

from typing import Any

import numpy as np

from etude.robopianist.state_mapping import StateMapping


def extract_tracking_observation(source: Any, mapping: StateMapping) -> dict[str, np.ndarray]:
    obs = {
        "q": mapping.extract_q(source),
        "qdot": mapping.extract_qdot(source),
    }
    fingertips = mapping.extract_fingertips(source)
    if fingertips is not None:
        obs["fingertips"] = fingertips
    key_state = mapping.extract_key_state(source)
    if key_state is not None:
        obs["key_state"] = key_state
    return obs
