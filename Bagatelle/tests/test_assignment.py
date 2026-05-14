from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bagatelle.assignment import assign_fingers_previous_pose  # noqa: E402


def test_assigns_single_key_to_nearest_finger() -> None:
    fingertips = np.zeros((10, 3), dtype=np.float32)
    fingertips[4] = [1.0, 0.0, 0.0]
    key_targets = np.asarray([[1.05, 0.0, 0.0]], dtype=np.float32)

    result = assign_fingers_previous_pose(np.asarray([20]), fingertips, key_targets)

    assert result.assigned_keys.tolist() == [20]
    assert result.assigned_finger_indices.tolist() == [4]
    assert result.unassigned_keys.tolist() == []


def test_chord_uses_unique_fingers_and_keys() -> None:
    fingertips = np.zeros((10, 3), dtype=np.float32)
    fingertips[1] = [0.0, 0.0, 0.0]
    fingertips[3] = [1.0, 0.0, 0.0]
    fingertips[7] = [2.0, 0.0, 0.0]
    key_targets = np.asarray(
        [
            [2.05, 0.0, 0.0],
            [0.05, 0.0, 0.0],
            [1.05, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    result = assign_fingers_previous_pose(np.asarray([60, 10, 40]), fingertips, key_targets)

    assert sorted(result.assigned_keys.tolist()) == [10, 40, 60]
    assert len(set(result.assigned_keys.tolist())) == 3
    assert len(set(result.assigned_finger_indices.tolist())) == 3


def test_unsorted_active_keys_keep_matching_target_rows() -> None:
    fingertips = np.zeros((10, 3), dtype=np.float32)
    fingertips[0] = [0.0, 0.0, 0.0]
    fingertips[1] = [10.0, 0.0, 0.0]
    key_targets = np.asarray(
        [
            [10.0, 0.0, 0.0],  # key 60
            [0.0, 0.0, 0.0],  # key 10
        ],
        dtype=np.float32,
    )

    result = assign_fingers_previous_pose(np.asarray([60, 10]), fingertips, key_targets)

    assert result.assigned_keys.tolist() == [10, 60]
    assert result.assigned_finger_indices.tolist() == [0, 1]


def test_equal_cost_ties_are_deterministic() -> None:
    fingertips = np.zeros((10, 3), dtype=np.float32)
    key_targets = np.zeros((2, 3), dtype=np.float32)

    first = assign_fingers_previous_pose(np.asarray([8, 4]), fingertips, key_targets)
    second = assign_fingers_previous_pose(np.asarray([4, 8]), fingertips, key_targets)

    np.testing.assert_array_equal(first.assigned_finger_indices, second.assigned_finger_indices)
    np.testing.assert_array_equal(first.assigned_keys, second.assigned_keys)
    assert first.assigned_finger_indices.tolist() == [0, 1]
    assert first.assigned_keys.tolist() == [4, 8]


def test_more_than_ten_keys_assigns_lowest_cost_subset() -> None:
    fingertips = np.zeros((10, 3), dtype=np.float32)
    for i in range(10):
        fingertips[i] = [float(i), 0.0, 0.0]
    active_keys = np.arange(12, dtype=np.int32)
    key_targets = np.asarray([[float(i), 0.0, 0.0] for i in range(12)], dtype=np.float32)

    result = assign_fingers_previous_pose(active_keys, fingertips, key_targets)

    assert result.count == 10
    assert result.assigned_keys.tolist() == list(range(10))
    assert result.unassigned_keys.tolist() == [10, 11]


def test_dense_outputs_are_padded_by_finger() -> None:
    fingertips = np.zeros((10, 3), dtype=np.float32)
    fingertips[2] = [1.0, 0.0, 0.0]
    target = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)

    result = assign_fingers_previous_pose(np.asarray([7]), fingertips, target)

    dense = result.dense_key_by_finger()
    assert dense.shape == (10,)
    assert dense[2] == 7
    assert np.all(dense[np.arange(10) != 2] == -1)
    assert result.dense_targets_by_finger().shape == (10, 3)
