from __future__ import annotations

import numpy as np

from etude.data.trajectory_io import finite_difference, load_qpos_trajectory, save_qpos_trajectory


def test_finite_difference_shape_and_endpoint() -> None:
    q_ref = np.stack([np.arange(46, dtype=np.float32), np.arange(46, dtype=np.float32) + 1.0])
    qdot = finite_difference(q_ref, dt=0.5)
    assert qdot.shape == q_ref.shape
    assert np.allclose(qdot, 2.0)


def test_save_and_load_qpos_trajectory(tmp_path) -> None:
    path = tmp_path / "trajectory.npz"
    q_ref = np.zeros((3, 46), dtype=np.float32)
    save_qpos_trajectory(path, {"q_ref": q_ref, "metadata": {"source": "unit"}})
    payload = load_qpos_trajectory(path)
    assert payload["q_ref"].shape == (3, 46)
    assert payload["qdot_ref"].shape == (3, 46)
    assert np.isclose(payload["dt"], 0.005)
    assert payload["metadata"]["source"] == "unit"
