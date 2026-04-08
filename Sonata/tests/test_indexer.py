from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sonata.data.indexer import scan_dataset


def test_scan_dataset_respects_max_episodes_per_song(tmp_path: Path) -> None:
    dataset_root = tmp_path / "rp1m_300"
    for song_id in ("song_a", "song_b"):
        song_dir = dataset_root / song_id
        song_dir.mkdir(parents=True, exist_ok=True)
        np.save(song_dir / "actions.npy", np.zeros((25, 4, 39), dtype=np.float32))
        np.save(song_dir / "hand_joints.npy", np.zeros((25, 4, 46), dtype=np.float32))

    output_root = tmp_path / "outputs"
    config = {
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "manifest_name": "dataset_manifest",
        "split_name": "dataset_splits",
        "summary_name": "dataset_summary.json",
        "note_search_roots": [],
        "subset_mode": "all",
        "max_songs": 0,
        "max_episodes": 20,
        "debug_num_songs": 0,
        "split_seed": 7,
        "split_ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
        "control_timestep": 0.05,
        "force": True,
        "scan_num_workers": 2,
    }

    class _Logger:
        def info(self, *args, **kwargs) -> None:
            del args, kwargs

    outputs = scan_dataset(config=config, logger=_Logger())
    manifest_df = pd.read_csv(outputs["manifest_base"].with_suffix(".csv"))

    assert manifest_df.groupby("song_id").size().to_dict() == {"song_a": 20, "song_b": 20}
