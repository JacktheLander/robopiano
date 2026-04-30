from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_primitive_frequency(assignments_df: pd.DataFrame, output_path: Path) -> None:
    counts = assignments_df["primitive_id"].value_counts().sort_values(ascending=False) if not assignments_df.empty else pd.Series(dtype=int)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(np.arange(len(counts)), counts.to_numpy(dtype=np.float32))
    ax.set_title("Primitive Usage Frequency")
    ax.set_xlabel("Primitive Rank")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_gmr_reconstruction(library_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    if not library_df.empty and "reconstruction_mse" in library_df.columns:
        ordered = library_df.sort_values("reconstruction_mse")
        ax.bar(np.arange(len(ordered)), ordered["reconstruction_mse"].to_numpy(dtype=np.float32))
    ax.set_title("Per-Primitive GMR Reconstruction MSE")
    ax.set_xlabel("Primitive")
    ax.set_ylabel("MSE")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_usage_entropy(assignments_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    if not assignments_df.empty:
        pivot = assignments_df.pivot_table(index="song_id", columns="primitive_id", values="segment_id", aggfunc="count", fill_value=0)
        probabilities = pivot.to_numpy(dtype=np.float32)
        probabilities = probabilities / np.clip(probabilities.sum(axis=1, keepdims=True), 1.0, None)
        entropy = -(probabilities * np.log(probabilities + 1e-8)).sum(axis=1)
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


def plot_segment_length_hist(segment_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    if not segment_df.empty and "duration_steps" in segment_df.columns:
        ax.hist(segment_df["duration_steps"].astype(float).to_numpy(), bins=min(40, max(len(segment_df) // 32, 10)))
    ax.set_title("Segment Length Distribution")
    ax.set_xlabel("Duration Steps")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_family_balance(segment_df: pd.DataFrame, output_path: Path, family_column: str = "coarse_family") -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    column = family_column if family_column in segment_df.columns else "heuristic_family"
    counts = segment_df[column].astype(str).value_counts().sort_values(ascending=False) if not segment_df.empty and column in segment_df.columns else pd.Series(dtype=int)
    ax.bar(np.arange(len(counts)), counts.to_numpy(dtype=np.float32))
    ax.set_xticks(np.arange(len(counts)))
    ax.set_xticklabels(counts.index.tolist(), rotation=35, ha="right", fontsize=8)
    ax.set_title("Segment Family Balance")
    ax.set_xlabel("Family")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_primitive_quality_hist(library_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    if not library_df.empty:
        column = "weighted_strike_error" if "weighted_strike_error" in library_df.columns else "reconstruction_mse"
        ax.hist(library_df[column].astype(float).to_numpy(), bins=min(30, max(len(library_df) // 4, 6)))
        ax.set_xlabel(column)
    else:
        ax.set_xlabel("quality")
    ax.set_title("Primitive Quality Distribution")
    ax.set_ylabel("Primitives")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_cluster_size_hist(assignments_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    if not assignments_df.empty and "primitive_id" in assignments_df.columns:
        counts = assignments_df["primitive_id"].value_counts().astype(float).to_numpy()
        ax.hist(counts, bins=min(30, max(len(counts), 5)))
    ax.set_title("Cluster Size Distribution")
    ax.set_xlabel("Segments per Primitive")
    ax.set_ylabel("Clusters")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
