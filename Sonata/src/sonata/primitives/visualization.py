from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_primitive_frequency(assignments_df: pd.DataFrame, output_path: Path) -> None:
    counts = assignments_df["primitive_id"].value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(np.arange(len(counts)), counts.to_numpy())
    ax.set_title("Primitive Usage Frequency")
    ax.set_xlabel("Primitive Rank")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_gmr_reconstruction(library_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ordered = library_df.sort_values("reconstruction_mse")
    ax.bar(np.arange(len(ordered)), ordered["reconstruction_mse"].to_numpy())
    ax.set_title("Per-Primitive GMR Reconstruction MSE")
    ax.set_xlabel("Primitive")
    ax.set_ylabel("MSE")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_usage_entropy(assignments_df: pd.DataFrame, output_path: Path) -> None:
    pivot = assignments_df.pivot_table(index="song_id", columns="primitive_id", values="segment_id", aggfunc="count", fill_value=0)
    probabilities = pivot.to_numpy(dtype=np.float32)
    probabilities = probabilities / np.clip(probabilities.sum(axis=1, keepdims=True), 1.0, None)
    entropy = -(probabilities * np.log(probabilities + 1e-8)).sum(axis=1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(entropy, bins=min(20, max(len(entropy), 5)))
    ax.set_title("Song-Level Primitive Usage Entropy")
    ax.set_xlabel("Entropy")
    ax.set_ylabel("Songs")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_primitive_traces(representatives: dict[str, np.ndarray], output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    for label, trace in representatives.items():
        ax.plot(trace, label=label)
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean")
    if len(representatives) <= 8:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
