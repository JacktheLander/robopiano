from __future__ import annotations

from pathlib import Path

import numpy as np


def save_video(frames: list[np.ndarray], output_path: str | Path, fps: int = 30) -> None:
    import imageio.v2 as imageio

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, fps=fps)
