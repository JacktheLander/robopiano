from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from etude.visualization.plots import plot_joint_tracking


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot q/q_ref from an Etude rollout or episode artifact.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as npz:
        q = npz["q"]
        q_ref = npz["q_ref"] if "q_ref" in npz else npz["q"]
    plot_joint_tracking(q, q_ref, Path(args.output))


if __name__ == "__main__":
    main()
