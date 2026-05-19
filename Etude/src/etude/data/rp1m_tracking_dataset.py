from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from etude.data.feature_builder import FeatureSpec, build_tracking_features


class RP1MTrackingDataset(Dataset):
    """Dataset over Etude episode `.npz` files listed in a manifest."""

    def __init__(
        self,
        dataset_root: str | Path,
        sequence_length: int = 1,
        feature_spec: FeatureSpec | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.sequence_length = int(sequence_length)
        if self.sequence_length < 1:
            raise ValueError("sequence_length must be >= 1")
        self.feature_spec = feature_spec or FeatureSpec()
        manifest_path = self.dataset_root / "manifest.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)
        self.manifest = pd.read_csv(manifest_path)
        self._episodes: list[dict[str, np.ndarray]] = []
        self._index: list[tuple[int, int]] = []
        for episode_idx, row in self.manifest.iterrows():
            path = self.dataset_root / str(row["path"])
            with np.load(path, allow_pickle=False) as npz:
                episode = {key: np.asarray(npz[key]) for key in npz.files}
            length = int(episode["q"].shape[0])
            for t in range(max(0, length - self.sequence_length + 1)):
                self._index.append((episode_idx, t))
            self._episodes.append(episode)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        episode_idx, start = self._index[idx]
        episode = self._episodes[episode_idx]
        features = []
        actions = []
        previous_action = np.zeros(episode["actions"].shape[1], dtype=np.float32)
        if start > 0:
            previous_action = episode["actions"][start - 1].astype(np.float32)
        for offset in range(self.sequence_length):
            t = start + offset
            feat = build_tracking_features(
                q=episode["q"][t],
                qdot=episode["qdot"][t],
                q_ref=episode["q_ref"],
                qdot_ref=episode["qdot_ref"],
                t=t,
                previous_action=previous_action,
                target_keys=episode.get("target_keys"),
                fingertips=episode.get("fingertips"),
                spec=self.feature_spec,
            )
            features.append(feat)
            action = episode["actions"][t].astype(np.float32)
            actions.append(action)
            previous_action = action
        return {
            "features": torch.from_numpy(np.stack(features)),
            "actions": torch.from_numpy(np.stack(actions)),
        }
