from __future__ import annotations

import sys
from pathlib import Path

PARTITA_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PARTITA_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse
import numpy as np

from partita.segmentation.segmenter import segment_dataset
from partita.utils.config import experiment_name, load_config, output_root
from partita.utils.io import ensure_dir, save_csv
from partita.utils.plotting import save_duration_histogram, save_segment_debug


def main() -> None:
    parser = argparse.ArgumentParser(description="Segment selected Partita trajectories.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    exp = experiment_name(config)
    data_dir = output_root(config) / "data" / exp
    prim_dir = ensure_dir(output_root(config) / "primitives" / exp)
    npz = np.load(data_dir / "selected_trajectories.npz")
    data = {k: npz[k] for k in npz.files}
    threshold = float(config.get("selection", {}).get("key_threshold", 0.5))
    segments = segment_dataset(data, config.get("segmentation", {}), key_threshold=threshold)
    save_csv(prim_dir / "segments.csv", segments)
    save_segment_debug(segments, prim_dir / "segment_debug.png")
    save_duration_histogram(segments, prim_dir / "segment_duration_histogram.png")
    print(f"Saved {len(segments)} segments to {prim_dir / 'segments.csv'}")


if __name__ == "__main__":
    main()
