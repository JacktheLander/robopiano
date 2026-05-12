from __future__ import annotations

from pathlib import Path

import numpy as np

from intermezzo.constants import KEY_SPLIT_LEFT_RIGHT, NUM_PIANO_KEYS


def validate_target_keys(value: np.ndarray, *, name: str = "target_keys") -> np.ndarray:
    keys = np.asarray(value, dtype=np.float32)
    if keys.ndim != 2:
        raise ValueError(f"{name} must be a 2D array with shape [T, 88], got {keys.shape}")
    if keys.shape[1] < NUM_PIANO_KEYS:
        raise ValueError(f"{name} must have at least 88 columns, got {keys.shape}")
    return np.ascontiguousarray(keys[:, :NUM_PIANO_KEYS], dtype=np.float32)


def load_target_keys_npz(path: str | Path) -> np.ndarray:
    npz_path = Path(path).expanduser().resolve()
    if not npz_path.is_file():
        raise FileNotFoundError(f"target_keys NPZ not found: {npz_path}")
    data = np.load(npz_path, allow_pickle=False)
    if "target_keys" in data:
        return validate_target_keys(data["target_keys"])
    if not data.files:
        raise ValueError(f"No arrays found in {npz_path}")
    return validate_target_keys(data[data.files[0]], name=data.files[0])


def binarize_target_keys(target_keys: np.ndarray, *, threshold: float = 0.5) -> np.ndarray:
    return validate_target_keys(target_keys) > float(threshold)


def extract_waypoint_frames(target_keys: np.ndarray, *, threshold: float = 0.5) -> np.ndarray:
    active = binarize_target_keys(target_keys, threshold=threshold)
    frames: list[int] = []
    previous = np.zeros((NUM_PIANO_KEYS,), dtype=bool)
    for index, row in enumerate(active):
        if bool(row.any()) and (index == 0 or not np.array_equal(row, previous)):
            frames.append(index)
        previous = row
    return np.asarray(frames, dtype=np.int64)


def keyset_hand_sides(target_key_row: np.ndarray, *, threshold: float = 0.5) -> dict[str, bool]:
    row = np.asarray(target_key_row, dtype=np.float32).reshape(-1)
    if row.shape[0] < NUM_PIANO_KEYS:
        raise ValueError(f"target key row must have at least 88 values, got {row.shape}")
    indices = np.flatnonzero(row[:NUM_PIANO_KEYS] > float(threshold))
    return {
        "left": bool(np.any(indices < KEY_SPLIT_LEFT_RIGHT)),
        "right": bool(np.any(indices >= KEY_SPLIT_LEFT_RIGHT)),
    }


def active_hands_for_transition(
    current_key_row: np.ndarray,
    next_key_row: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, bool]:
    current = keyset_hand_sides(current_key_row, threshold=threshold)
    nxt = keyset_hand_sides(next_key_row, threshold=threshold)
    return {"left": current["left"] or nxt["left"], "right": current["right"] or nxt["right"]}
