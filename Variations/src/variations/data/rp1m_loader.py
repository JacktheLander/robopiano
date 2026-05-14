from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    import zarr
except ImportError:  # pragma: no cover
    zarr = None

KNOWN_ARRAYS = ["actions", "goals", "piano_states", "hand_joints", "hand_fingertips"]
WAVE_ZARR_PYTHON = Path("/WAVE/users2/unix/jlanders/.conda/envs/sonata/bin/python")


def _maybe_reexec_with_wave_zarr_python() -> None:
    if os.environ.get("VARIATIONS_DISABLE_ZARR_REEXEC") == "1":
        return
    if os.environ.get("VARIATIONS_ZARR_REEXECED") == "1":
        return
    fallback = Path(os.environ.get("VARIATIONS_ZARR_PYTHON", str(WAVE_ZARR_PYTHON)))
    if not fallback.exists():
        return
    try:
        if Path(sys.executable).resolve() == fallback.resolve():
            return
    except Exception:
        pass
    env = os.environ.copy()
    env["VARIATIONS_ZARR_REEXECED"] = "1"
    print(f"[variations] zarr unavailable in {sys.executable}; re-executing with {fallback}", file=sys.stderr)
    os.execve(str(fallback), [str(fallback), *sys.argv], env)


def require_zarr() -> Any:
    if zarr is None:
        _maybe_reexec_with_wave_zarr_python()
        raise RuntimeError(
            "The Python package 'zarr' is required to read RP1M. Activate an environment with zarr "
            "or set VARIATIONS_ZARR_PYTHON to a Python executable that can import zarr."
        )
    return zarr


def open_rp1m_root(path: str | Path):
    zp = require_zarr()
    root_path = Path(path)
    if not root_path.exists():
        raise FileNotFoundError(f"RP1M Zarr root not found: {root_path}")
    return zp.open_group(str(root_path), mode="r")


def _keys(group) -> list[str]:
    if hasattr(group, "keys"):
        return sorted(list(group.keys()))
    return []


def group_keys(group) -> list[str]:
    if hasattr(group, "group_keys"):
        return sorted(list(group.group_keys()))
    names = []
    for key in _keys(group):
        try:
            item = group[key]
            if hasattr(item, "keys"):
                names.append(key)
        except Exception:
            pass
    return sorted(names)


def array_keys(group) -> list[str]:
    if hasattr(group, "array_keys"):
        return sorted(list(group.array_keys()))
    names = []
    for key in _keys(group):
        try:
            item = group[key]
            if hasattr(item, "shape") and hasattr(item, "dtype"):
                names.append(key)
        except Exception:
            pass
    return sorted(names)


def list_songs(root, max_songs: int | None = None) -> list[str]:
    songs = group_keys(root)
    return songs if max_songs is None else songs[:max_songs]


def inspect_song(root, song_name: str) -> dict[str, Any]:
    group = root[song_name]
    arrays = {}
    for name in array_keys(group):
        arr = group[name]
        arrays[name] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
    return {"song_name": song_name, "arrays": arrays}


def available_arrays(song_group) -> list[str]:
    names = set(array_keys(song_group))
    ordered = [name for name in KNOWN_ARRAYS if name in names]
    ordered.extend(sorted(names - set(ordered)))
    return ordered


def trajectory_count(song_group) -> int:
    if "actions" not in array_keys(song_group):
        raise RuntimeError("Selected song does not contain required 'actions' array.")
    shape = song_group["actions"].shape
    if len(shape) < 2:
        raise RuntimeError(f"Expected actions to be at least 2D, got shape {shape}")
    return int(shape[0])


def read_trajectory(song_group, trajectory_id: int, arrays: Iterable[str] | None = None) -> dict[str, np.ndarray]:
    names = list(arrays) if arrays is not None else available_arrays(song_group)
    out: dict[str, np.ndarray] = {}
    available = set(array_keys(song_group))
    for name in names:
        if name not in available:
            continue
        arr = song_group[name]
        if len(arr.shape) >= 2 and arr.shape[0] > trajectory_id:
            out[name] = np.asarray(arr[int(trajectory_id)])
    out["trajectory_id"] = np.asarray(int(trajectory_id))
    return out


def read_trajectories(song_group, trajectory_ids: list[int], arrays: Iterable[str] | None = None) -> dict[str, np.ndarray]:
    names = list(arrays) if arrays is not None else available_arrays(song_group)
    out: dict[str, np.ndarray] = {}
    available = set(array_keys(song_group))
    for name in names:
        if name not in available:
            continue
        arr = song_group[name]
        vals = [np.asarray(arr[int(i)]) for i in trajectory_ids]
        out[name] = np.stack(vals, axis=0)
    out["trajectory_ids"] = np.asarray(trajectory_ids, dtype=np.int64)
    return out

