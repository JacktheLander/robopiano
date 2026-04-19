from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from robodiffusion.data.windows import CachedWindowDataset, build_window_cache


def _write_episode_arrays(song_root: Path) -> None:
    timesteps = 24
    np.save(song_root / "actions.npy", np.linspace(-1.0, 1.0, timesteps * 3, dtype=np.float32).reshape(timesteps, 3))
    goal = np.zeros((timesteps, 89), dtype=np.float32)
    goal[:, :4] = 1.0
    np.save(song_root / "goals.npy", goal)
    np.save(song_root / "piano_states.npy", goal.copy())
    joints = np.linspace(0.0, 1.0, timesteps * 6, dtype=np.float32).reshape(timesteps, 6)
    np.save(song_root / "hand_joints.npy", joints)
    np.save(song_root / "joint_velocities.npy", np.gradient(joints, axis=0).astype(np.float32))


def test_build_window_cache_from_npy_manifest(tmp_path: Path) -> None:
    dataset_root = tmp_path / "rp1m"
    dataset_root.mkdir()
    song_root = dataset_root / "song_a"
    song_root.mkdir()
    _write_episode_arrays(song_root)

    manifest = pd.DataFrame(
        [
            {
                "song_id": "song_a",
                "episode_id": "episode_0",
                "split": "train",
                "backend": "npy_dir",
                "dataset_root": str(dataset_root),
                "song_path": str(song_root),
                "song_key": "song_a",
                "note_path": "",
                "control_timestep": 0.05,
                "episode_index": 0,
            }
        ]
    )
    manifest_path = tmp_path / "dataset_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    result = build_window_cache(
        {
            "data_manifest_path": manifest_path.with_suffix(""),
            "output_root": tmp_path / "cache",
            "obs_horizon": 4,
            "pred_horizon": 6,
            "max_samples_per_shard": 8,
            "observation_spec": {
                "use_goal": True,
                "use_piano_state": True,
                "use_sustain_state": True,
                "use_hand_joints": True,
                "use_joint_velocities": True,
            },
        }
    )
    assert result["metadata_path"].exists()
    dataset = CachedWindowDataset(tmp_path / "cache", split="train")
    assert len(dataset) > 0
    sample = dataset[0]
    assert sample["score_window"].shape == (4, 14)
    assert sample["state_window"].shape[0] == 4
    assert sample["action_target"].shape == (6, 3)
