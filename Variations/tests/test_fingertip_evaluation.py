from __future__ import annotations

from types import ModuleType, SimpleNamespace
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from variations.data.dataset import FINGERTIP_STATE_DIM, JOINT_STATE_DIM, PressPairsDataset  # noqa: E402
from variations.evaluation.fingertips import (  # noqa: E402
    capture_fingertips,
    fingertip_metrics,
    measure_fingertips_with_mujoco,
)


def test_press_pairs_dataset_exposes_fingertips(tmp_path: Path) -> None:
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    (split_dir / "split_index.csv").write_text(
        "song_id,safe_song_id,split,rows_accepted\nsong-a,song_a,train,2\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        split_dir / "norm_stats.npz",
        mean=np.zeros((JOINT_STATE_DIM,), dtype=np.float32),
        std=np.ones((JOINT_STATE_DIM,), dtype=np.float32),
    )
    hand_state = np.arange(2 * (JOINT_STATE_DIM + FINGERTIP_STATE_DIM), dtype=np.float32).reshape(2, -1)
    target_keys = np.zeros((2, 88), dtype=np.float32)
    np.savez_compressed(tmp_path / "song_song_a.npz", target_keys=target_keys, hand_state=hand_state)

    dataset = PressPairsDataset(tmp_path, split="train")
    sample = dataset[1]

    assert dataset.hand_state.shape == (2, JOINT_STATE_DIM)
    assert dataset.fingertip_state.shape == (2, FINGERTIP_STATE_DIM)
    np.testing.assert_array_equal(sample["joint_state"].numpy(), hand_state[1, :JOINT_STATE_DIM])
    np.testing.assert_array_equal(sample["fingertip_state"].numpy(), hand_state[1, JOINT_STATE_DIM:])
    np.testing.assert_array_equal(sample["hand_fingertips"].numpy(), sample["fingertip_state"].numpy())


def test_fingertip_metrics_success_thresholds() -> None:
    target = np.zeros((2, 30), dtype=np.float32)
    pred = target.copy()
    pred[1, 0] = 0.02

    metrics = fingertip_metrics(pred, target, success_thresholds=(0.01, 0.03))

    assert metrics["fingertip_examples"] == 2.0
    assert metrics["fingertip_rmse"] > 0.0
    assert np.isclose(metrics["fingertip_per_tip_width_distance_mean"], 0.001)
    assert metrics["fingertip_success_at_0p01"] == 0.5
    assert metrics["fingertip_width_success_at_0p01"] == 0.5
    assert metrics["fingertip_success_at_0p03"] == 1.0


def test_capture_fingertips_uses_right_then_left_site_order() -> None:
    right = np.arange(15, dtype=np.float32).reshape(5, 3)
    left = (np.arange(15, dtype=np.float32) + 100).reshape(5, 3)

    class FakePhysics:
        def bind(self, sites):
            return SimpleNamespace(xpos=right if sites == "right-sites" else left)

    task = SimpleNamespace(
        right_hand=SimpleNamespace(fingertip_sites="right-sites"),
        left_hand=SimpleNamespace(fingertip_sites="left-sites"),
    )

    values = capture_fingertips(task, FakePhysics())

    np.testing.assert_array_equal(values[:15], right.reshape(-1))
    np.testing.assert_array_equal(values[15:], left.reshape(-1))


def test_measure_fingertips_with_mujoco_restores_each_pose(monkeypatch, tmp_path: Path) -> None:
    applied: list[np.ndarray] = []
    right = np.ones((5, 3), dtype=np.float32)
    left = np.full((5, 3), 2.0, dtype=np.float32)

    class FakePhysics:
        def __init__(self) -> None:
            self.forward_count = 0

        def forward(self) -> None:
            self.forward_count += 1

        def bind(self, sites):
            return SimpleNamespace(xpos=right if sites == "right-sites" else left)

    class FakeEnv:
        def reset(self) -> None:
            pass

        def close(self) -> None:
            pass

    fake_physics = FakePhysics()
    fake_task = SimpleNamespace(
        right_hand=SimpleNamespace(fingertip_sites="right-sites"),
        left_hand=SimpleNamespace(fingertip_sites="left-sites"),
    )

    fake_rollout = ModuleType("partita.evaluation.rollout")
    fake_rollout.candidate_environment_names = lambda name: [name]
    fake_rollout.write_goals_proto = lambda _keys, path, **_kwargs: Path(path)
    fake_rollout._load_env = lambda **_kwargs: ("fake-env", FakeEnv(), {})
    fake_rollout._locate_task_physics_piano = lambda _env: (fake_task, fake_physics, object())

    def fake_set_qpos(_task, _physics, hand_qpos):
        applied.append(np.asarray(hand_qpos, dtype=np.float32).copy())
        return int(np.asarray(hand_qpos).size)

    fake_rollout._set_reduced_hand_qpos = fake_set_qpos

    monkeypatch.setitem(sys.modules, "partita", ModuleType("partita"))
    monkeypatch.setitem(sys.modules, "partita.evaluation", ModuleType("partita.evaluation"))
    monkeypatch.setitem(sys.modules, "partita.evaluation.rollout", fake_rollout)

    poses = np.arange(2 * JOINT_STATE_DIM, dtype=np.float32).reshape(2, JOINT_STATE_DIM)
    measured, meta = measure_fingertips_with_mujoco(poses, output_dir=tmp_path)

    assert meta["environment_name"] == "fake-env"
    assert meta["restored_hand_joint_count"] == JOINT_STATE_DIM
    assert len(applied) == 2
    assert fake_physics.forward_count == 4
    assert measured.shape == (2, 30)
    np.testing.assert_array_equal(measured[0, :15], right.reshape(-1))
    np.testing.assert_array_equal(measured[0, 15:], left.reshape(-1))
