from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_segment_debug(segments: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    sample = segments.head(800)
    ax.scatter(sample["start_t"], sample["trajectory_index"], c=sample["duration"], s=8, cmap="viridis")
    ax.set_xlabel("start_t")
    ax.set_ylabel("trajectory_index")
    ax.set_title("Segment starts by trajectory")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_duration_histogram(segments: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(segments["duration"], bins=30, color="#3b7ddd", edgecolor="white")
    ax.set_xlabel("Segment duration")
    ax.set_ylabel("Count")
    ax.set_title("Segment duration histogram")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_primitive_usage(summary: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(summary["primitive_id"].astype(str), summary["count"], color="#547aa5")
    ax.set_xlabel("Primitive")
    ax.set_ylabel("Count")
    ax.set_title("Primitive usage")
    ax.tick_params(axis="x", labelrotation=90)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_primitive_timeline(assignments: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    sample = assignments.head(5000)
    scatter = ax.scatter(sample["start_t"], sample["trajectory_index"], c=sample["primitive_id"], s=8, cmap="tab20")
    ax.set_xlabel("start_t")
    ax.set_ylabel("trajectory_index")
    ax.set_title("Primitive timeline by trajectory")
    fig.colorbar(scatter, ax=ax, label="primitive_id")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_action_comparison(original, reconstructed, path: str | Path, dims: int = 8) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    orig = np.asarray(original)
    recon = np.asarray(reconstructed)
    n = min(dims, orig.shape[-1], recon.shape[-1])
    fig, axes = plt.subplots(n, 1, figsize=(12, max(3, n * 1.5)), sharex=True)
    if n == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(orig[:, i], label="original", linewidth=1.0)
        ax.plot(recon[:, i], label="reconstructed", linewidth=1.0, alpha=0.8)
        ax.set_ylabel(f"a{i}")
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("time")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_pianoroll_comparison(goal, original, reconstructed, path: str | Path, threshold: float = 0.5) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    panels = []
    labels = []
    if goal is not None:
        panels.append(np.asarray(goal).T > threshold)
        labels.append("goal")
    if original is not None:
        panels.append(np.asarray(original).T > threshold)
        labels.append("original piano_state")
    if reconstructed is not None:
        panels.append(np.asarray(reconstructed).T > threshold)
        labels.append("reconstructed piano_state")
    if not panels:
        panels = [np.zeros((1, 1))]
        labels = ["no piano-state data"]
    fig, axes = plt.subplots(len(panels), 1, figsize=(12, 3 * len(panels)), sharex=True)
    if len(panels) == 1:
        axes = [axes]
    for ax, data, label in zip(axes, panels, labels):
        ax.imshow(data, aspect="auto", origin="lower", interpolation="nearest", cmap="gray_r")
        ax.set_ylabel(label)
    axes[-1].set_xlabel("time")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
