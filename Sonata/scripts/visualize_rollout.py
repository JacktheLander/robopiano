from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import matplotlib.pyplot as plt
import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize Sonata-3 rollout samples.")
    parser.add_argument("--sample-npz", required=True)
    parser.add_argument("--output-path", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = np.load(Path(args.sample_npz).resolve(), allow_pickle=True)
    predicted = np.asarray(payload["predicted"], dtype=np.float32)
    target = np.asarray(payload["target"], dtype=np.float32)
    prior = np.asarray(payload["prior"], dtype=np.float32)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(predicted[0, :, 0], label="predicted")
    axes[0].plot(target[0, :, 0], label="target")
    axes[0].set_title("Action channel 0")
    axes[0].legend(loc="best")
    axes[1].plot(prior[0, :, 0], label="prior")
    axes[1].plot(target[0, :, 0], label="target")
    axes[1].set_title("GMR prior vs target")
    axes[2].plot(np.linalg.norm(predicted[0], axis=-1), label="pred norm")
    axes[2].plot(np.linalg.norm(target[0], axis=-1), label="target norm")
    axes[2].set_title("Action norm over time")
    axes[2].legend(loc="best")
    fig.tight_layout()
    fig.savefig(Path(args.output_path).resolve(), dpi=180)
    plt.close(fig)
