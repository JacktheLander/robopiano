from __future__ import annotations

from pathlib import Path

from partita.utils.plotting import save_pianoroll_comparison


def plot_pianoroll_comparison(goal, original, reconstructed, path: str | Path, threshold: float = 0.5) -> None:
    save_pianoroll_comparison(goal, original, reconstructed, path, threshold=threshold)
