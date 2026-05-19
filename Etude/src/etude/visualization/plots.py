from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_joint_tracking(q: np.ndarray, q_ref: np.ndarray, output_path: str | Path) -> None:
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.asarray(q_ref)[:, 0], label="q_ref[0]")
    ax.plot(np.asarray(q)[:, 0], label="q[0]")
    ax.set_xlabel("step")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
