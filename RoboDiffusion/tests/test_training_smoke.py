from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from robodiffusion.data.windows import build_window_cache
from robodiffusion.training.trainer import run_training


def _write_song(song_root: Path, timesteps: int = 24) -> None:
    np.save(song_root / "actions.npy", np.random.randn(timesteps, 3).astype(np.float32))
    goal = np.zeros((timesteps, 89), dtype=np.float32)
    goal[:, :8] = 1.0
    np.save(song_root / "goals.npy", goal)
    np.save(song_root / "piano_states.npy", goal.copy())
    joints = np.random.randn(timesteps, 6).astype(np.float32)
    np.save(song_root / "hand_joints.npy", joints)
    np.save(song_root / "joint_velocities.npy", np.gradient(joints, axis=0).astype(np.float32))


def test_training_smoke(tmp_path: Path) -> None:
    dataset_root = tmp_path / "rp1m"
    dataset_root.mkdir()
    rows = []
    for split, song_name in (("train", "song_a"), ("val", "song_b")):
        song_root = dataset_root / song_name
        song_root.mkdir()
        _write_song(song_root)
        rows.append(
            {
                "song_id": song_name,
                "episode_id": f"{song_name}_ep0",
                "split": split,
                "backend": "npy_dir",
                "dataset_root": str(dataset_root),
                "song_path": str(song_root),
                "song_key": song_name,
                "note_path": "",
                "control_timestep": 0.05,
                "episode_index": 0,
            }
        )
    manifest_path = tmp_path / "dataset_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    build_window_cache(
        {
            "data_manifest_path": manifest_path.with_suffix(""),
            "output_root": tmp_path / "cache",
            "obs_horizon": 4,
            "pred_horizon": 4,
            "max_samples_per_shard": 16,
            "observation_spec": {
                "use_goal": True,
                "use_piano_state": True,
                "use_sustain_state": True,
                "use_hand_joints": True,
                "use_joint_velocities": True,
            },
        }
    )
    result = run_training(
        {
            "dataset_root": tmp_path / "cache",
            "output_root": tmp_path / "train",
            "experiment_name": "smoke",
            "seed": 7,
            "device": "cpu",
            "epochs": 1,
            "batch_size": 4,
            "num_workers": 0,
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "checkpoint_interval": 1,
            "resume": False,
            "progress_bar": False,
            "model_dim": 32,
            "num_layers": 2,
            "num_heads": 4,
            "dropout": 0.0,
            "diffusion_steps": 4,
            "beta_start": 1e-4,
            "beta_end": 2e-2,
            "action_execute_horizon": 2,
            "action_loss_weight": 1.0,
            "smoothness_weight": 0.05,
            "grad_clip_norm": 1.0,
        }
    )
    assert result["best_checkpoint"].exists()
