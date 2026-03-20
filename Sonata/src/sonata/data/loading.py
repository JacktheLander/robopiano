from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sonata.data.schema import EpisodeRecord
from sonata.utils.io import read_json, read_table

ARRAY_FIELDS = (
    "actions",
    "goals",
    "piano_states",
    "hand_joints",
    "joint_velocities",
    "hand_fingertips",
    "wrist_pose",
    "hand_pose",
)


def load_manifest(path_without_suffix: str | Path) -> pd.DataFrame:
    df = read_table(path_without_suffix).copy()
    defaults = {
        "split": "train",
        "backend": "zarr",
        "song_key": "",
        "song_path": "",
        "note_path": "",
    }
    for column, default in defaults.items():
        if column not in df.columns:
            df[column] = default
    return df


def load_stage1_source_manifest(primitive_root: str | Path) -> pd.DataFrame:
    primitive_root = Path(primitive_root).resolve()
    run_config = read_json(primitive_root / "run_config.json")
    data_output_root = Path(str(run_config["data_output_root"])).resolve()
    manifest_name = str(run_config.get("data_manifest_name", "dataset_manifest"))
    return load_manifest(data_output_root / manifest_name)


def build_manifest_lookup(manifest_df: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in manifest_df.itertuples(index=False):
        lookup[(str(row.song_id), str(row.episode_id))] = row._asdict()
    return lookup


def load_episode_record(row: dict[str, Any] | pd.Series) -> EpisodeRecord:
    payload = dict(row) if not isinstance(row, dict) else dict(row)
    backend = _optional_str(payload.get("backend")) or _infer_backend(payload)
    if backend == "zarr":
        arrays = _load_from_zarr(payload)
    elif backend == "npy_dir":
        arrays = _load_from_npy_dir(payload)
    else:
        raise ValueError(f"Unsupported Sonata dataset backend: {backend}")
    return EpisodeRecord(
        song_id=str(payload["song_id"]),
        episode_id=str(payload["episode_id"]),
        split=_optional_str(payload.get("split")) or "train",
        note_path=_optional_path(payload.get("note_path")),
        control_timestep=float(payload.get("control_timestep", 0.05)),
        actions=arrays.get("actions"),
        goals=arrays.get("goals"),
        piano_states=arrays.get("piano_states"),
        hand_joints=arrays.get("hand_joints"),
        joint_velocities=arrays.get("joint_velocities"),
        hand_fingertips=arrays.get("hand_fingertips"),
        wrist_pose=arrays.get("wrist_pose"),
        hand_pose=arrays.get("hand_pose"),
    )


def _infer_backend(payload: dict[str, Any]) -> str:
    dataset_root = _optional_path(payload.get("dataset_root"))
    if dataset_root is not None and dataset_root.name.endswith(".zarr"):
        return "zarr"
    return "npy_dir"


def _load_from_zarr(payload: dict[str, Any]) -> dict[str, np.ndarray | None]:
    try:
        import zarr
    except Exception as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "zarr is required to load Sonata dataset episodes from a .zarr archive."
        ) from exc

    dataset_root = _required_path(payload.get("dataset_root"), "dataset_root")
    song_key = _optional_str(payload.get("song_key")) or str(payload["song_id"])
    episode_index = int(payload.get("episode_index", 0))
    root = zarr.open(str(dataset_root), mode="r")
    group = root[song_key]
    arrays: dict[str, np.ndarray | None] = {}
    for name in ARRAY_FIELDS:
        if name not in group:
            arrays[name] = None
            continue
        arrays[name] = np.asarray(group[name][episode_index], dtype=np.float32)
    return arrays


def _load_from_npy_dir(payload: dict[str, Any]) -> dict[str, np.ndarray | None]:
    song_path = _optional_path(payload.get("song_path"))
    if song_path is None:
        dataset_root = _required_path(payload.get("dataset_root"), "dataset_root")
        song_key = _optional_str(payload.get("song_key")) or str(payload["song_id"])
        song_path = dataset_root / song_key
    episode_index = int(payload.get("episode_index", 0))
    arrays: dict[str, np.ndarray | None] = {}
    for name in ARRAY_FIELDS:
        arrays[name] = _load_npy_array(song_path=song_path, array_name=name, episode_index=episode_index)
    return arrays


def _load_npy_array(song_path: Path, array_name: str, episode_index: int) -> np.ndarray | None:
    direct_file = song_path / f"{array_name}.npy"
    if direct_file.exists():
        array = np.load(direct_file, mmap_mode="r")
        return _slice_episode(array, episode_index)

    array_dir = song_path / array_name
    if array_dir.is_dir():
        for candidate in (
            array_dir / f"{episode_index}.npy",
            array_dir / f"{episode_index:05d}.npy",
            array_dir / f"episode_{episode_index:05d}.npy",
        ):
            if candidate.exists():
                array = np.load(candidate, mmap_mode="r")
                return np.asarray(array, dtype=np.float32)
    return None


def _slice_episode(array: np.ndarray, episode_index: int) -> np.ndarray:
    if array.ndim >= 3:
        return np.asarray(array[episode_index], dtype=np.float32)
    if array.ndim == 2:
        return np.asarray(array, dtype=np.float32)
    if array.ndim == 1:
        return np.asarray(array[:, None], dtype=np.float32)
    raise ValueError(f"Unsupported episode array rank: {array.ndim}")


def _optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text).resolve()


def _required_path(value: Any, name: str) -> Path:
    path = _optional_path(value)
    if path is None:
        raise ValueError(f"Missing required Sonata manifest field: {name}")
    return path


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    text = str(value).strip()
    return text or None
