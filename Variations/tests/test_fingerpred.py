from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
VARIATIONS_DIR = ROOT
for path in (SRC, VARIATIONS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from variations.data.dataset import FINGERTIP_STATE_DIM, JOINT_STATE_DIM, PressPairsDataset, compute_fingerpred_norm_stats
from variations.data.fingerpred import coord_mask_from_tip_mask, infer_active_tip_mask
from variations.fingerpred import FingerPredNormalizer, load_fingerpred_checkpoint, masked_mse, predict_with_fingerpred
from variations.models.latent_mdn import build_latent_mdn
from variations.models.pose_autoencoder import build_pose_autoencoder


def _write_split(tmp_path: Path) -> None:
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    (split_dir / "split_index.csv").write_text(
        "song_id,safe_song_id,split,rows_accepted\nsong-a,song_a,train,2\n",
        encoding="utf-8",
    )


def _write_song(tmp_path: Path, target_keys: np.ndarray, hand_state: np.ndarray) -> None:
    np.savez_compressed(tmp_path / "song_song_a.npz", target_keys=target_keys, hand_state=hand_state)


def test_infer_active_tip_mask_assigns_nearest_observed_tip() -> None:
    key_positions = np.zeros((88, 3), dtype=np.float32)
    key_positions[2] = [1.0, 0.0, 0.0]
    key_positions[7] = [-1.0, 0.0, 0.0]
    target_keys = np.zeros((1, 88), dtype=np.float32)
    target_keys[0, [2, 7]] = 1.0
    fingertips = np.full((1, 30), 10.0, dtype=np.float32)
    tips = fingertips.reshape(1, 10, 3)
    tips[0, 3] = [1.01, 0.0, 0.0]
    tips[0, 8] = [-1.02, 0.0, 0.0]

    mask = infer_active_tip_mask(target_keys, fingertips, key_positions=key_positions)

    assert mask.shape == (1, 10)
    np.testing.assert_array_equal(np.flatnonzero(mask[0]), np.array([3, 8]))
    coord = coord_mask_from_tip_mask(mask)
    assert coord.shape == (1, 30)
    assert coord[0, 9:12].sum() == 3.0
    assert coord[0, 24:27].sum() == 3.0


def test_press_pairs_dataset_default_joint_mode_unchanged(tmp_path: Path) -> None:
    _write_split(tmp_path)
    np.savez_compressed(
        tmp_path / "splits" / "norm_stats.npz",
        mean=np.zeros((JOINT_STATE_DIM,), dtype=np.float32),
        std=np.ones((JOINT_STATE_DIM,), dtype=np.float32),
    )
    target_keys = np.zeros((2, 88), dtype=np.float32)
    hand_state = np.arange(2 * (JOINT_STATE_DIM + FINGERTIP_STATE_DIM), dtype=np.float32).reshape(2, -1)
    _write_song(tmp_path, target_keys, hand_state)

    dataset = PressPairsDataset(tmp_path, split="train")
    sample = dataset[1]

    assert dataset.output_mode == "joints_only"
    assert dataset.target_state.shape == (2, JOINT_STATE_DIM)
    assert sample["hand_state_normalized"].shape == (JOINT_STATE_DIM,)
    assert sample["target_state"].shape == (JOINT_STATE_DIM,)
    np.testing.assert_array_equal(sample["target_state"].numpy(), hand_state[1, :JOINT_STATE_DIM])


def test_press_pairs_dataset_fingerpred_mode_emits_masked_fingertip_targets(tmp_path: Path) -> None:
    _write_split(tmp_path)
    np.savez_compressed(
        tmp_path / "splits" / "norm_stats_fingerpred.npz",
        mean=np.zeros((FINGERTIP_STATE_DIM,), dtype=np.float32),
        std=np.ones((FINGERTIP_STATE_DIM,), dtype=np.float32),
    )
    key_positions = np.zeros((88, 3), dtype=np.float32)
    key_positions[12] = [0.25, 0.0, 0.0]
    target_keys = np.zeros((1, 88), dtype=np.float32)
    target_keys[0, 12] = 1.0
    hand_state = np.zeros((1, JOINT_STATE_DIM + FINGERTIP_STATE_DIM), dtype=np.float32)
    tips = hand_state[:, JOINT_STATE_DIM:].reshape(1, 10, 3)
    tips[:] = 4.0
    tips[0, 4] = [0.251, 0.0, 0.0]
    _write_song(tmp_path, target_keys, hand_state)

    dataset = PressPairsDataset(tmp_path, split="train", output_mode="fingerpred", key_positions=key_positions)
    sample = dataset[0]

    assert dataset.output_mode == "fingerpred"
    assert sample["target_state"].shape == (FINGERTIP_STATE_DIM,)
    assert sample["target_state_normalized"].shape == (FINGERTIP_STATE_DIM,)
    assert sample["active_tip_mask"].shape == (10,)
    assert sample["target_coord_mask"].shape == (FINGERTIP_STATE_DIM,)
    assert int(sample["active_tip_mask"].argmax().item()) == 4
    assert sample["target_coord_mask"][12:15].sum().item() == 3.0


def test_compute_fingerpred_norm_stats_uses_active_coordinates(tmp_path: Path) -> None:
    _write_split(tmp_path)
    key_positions = np.zeros((88, 3), dtype=np.float32)
    key_positions[3] = [1.0, 0.0, 0.0]
    target_keys = np.zeros((2, 88), dtype=np.float32)
    target_keys[:, 3] = 1.0
    hand_state = np.zeros((2, JOINT_STATE_DIM + FINGERTIP_STATE_DIM), dtype=np.float32)
    tips = hand_state[:, JOINT_STATE_DIM:].reshape(2, 10, 3)
    tips[:] = 5.0
    tips[0, 1] = [1.0, 0.0, 0.0]
    tips[1, 1] = [1.2, 0.0, 0.0]
    _write_song(tmp_path, target_keys, hand_state)

    path = compute_fingerpred_norm_stats(tmp_path, key_positions=key_positions)
    stats = np.load(path, allow_pickle=False)

    assert stats["mean"].shape == (FINGERTIP_STATE_DIM,)
    assert stats["std"].shape == (FINGERTIP_STATE_DIM,)
    np.testing.assert_allclose(stats["mean"][3], 1.1, atol=1e-6)


def test_masked_mse_only_uses_active_coordinates() -> None:
    pred = torch.tensor([[1.0, 100.0, 3.0]])
    target = torch.tensor([[0.0, 0.0, 1.0]])
    mask = torch.tensor([[1.0, 0.0, 1.0]])

    loss = masked_mse(pred, target, mask)

    assert torch.isclose(loss, torch.tensor(2.5))


def test_fingerpred_checkpoint_load_and_inference(tmp_path: Path) -> None:
    config = {
        "model_type": "fingerpred",
        "autoencoder": {"input_dim": 30, "latent_dim": 4, "dropout": 0.0},
        "mdn": {"input_dim": 88, "latent_dim": 4, "num_components": 2, "hidden_dim": 32, "dropout": 0.0},
    }
    autoencoder = build_pose_autoencoder(config)
    mdn = build_latent_mdn(config)
    latent_stats = {"z_mean": torch.zeros(4), "z_std": torch.ones(4)}
    normalizer = FingerPredNormalizer(mean=torch.zeros(30), std=torch.ones(30))
    run_root = tmp_path / "fingerpred_run"
    ae_path = run_root / "autoencoder" / "checkpoints" / "best.pt"
    mdn_path = run_root / "mdn" / "checkpoints" / "best.pt"
    ae_path.parent.mkdir(parents=True)
    mdn_path.parent.mkdir(parents=True)
    torch.save({"model_state_dict": autoencoder.state_dict(), "config": config, "model_type": "fingerpred"}, ae_path)
    torch.save(
        {
            "model_state_dict": mdn.state_dict(),
            "model_type": "fingerpred",
            "output_mode": "active_fingertips",
            "target_dim": 30,
            "latent_stats": latent_stats,
            "autoencoder_checkpoint": str(ae_path),
            "normalizer": normalizer.state_dict(),
            "config": config,
        },
        mdn_path,
    )

    loaded_mdn, loaded_ae, loaded_stats, loaded_norm, _cfg, _payload = load_fingerpred_checkpoint(mdn_path)
    out = predict_with_fingerpred(torch.randn(3, 88), loaded_mdn, loaded_ae, loaded_stats, loaded_norm)

    assert out.shape == (3, 30)
    assert torch.isfinite(out).all()
