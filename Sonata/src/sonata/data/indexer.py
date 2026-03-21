from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import logging
import math
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sonata.data.schema import ManifestRecord
from sonata.utils.io import write_json, write_table

ARRAY_NAMES = (
    "actions",
    "goals",
    "piano_states",
    "hand_joints",
    "hand_fingertips",
    "joint_velocities",
    "wrist_pose",
    "hand_pose",
)


def scan_dataset(config: dict[str, Any], logger: logging.Logger) -> dict[str, Path]:
    output_root = Path(config["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_base = output_root / str(config.get("manifest_name", "dataset_manifest"))
    split_base = output_root / str(config.get("split_name", "dataset_splits"))
    summary_path = output_root / str(config.get("summary_name", "dataset_summary.json"))
    if manifest_base.with_suffix(".csv").exists() and not bool(config.get("force", False)):
        return {"manifest_base": manifest_base, "split_base": split_base, "summary_path": summary_path}

    dataset_root = Path(config["dataset_root"]).resolve()
    note_search_roots = [Path(path).resolve() for path in config.get("note_search_roots", [])]
    scan_num_workers = _positive_int(config.get("scan_num_workers")) or 0
    if dataset_root.name.endswith(".zarr"):
        songs = _scan_zarr_root(
            dataset_root=dataset_root,
            note_search_roots=note_search_roots,
            num_workers=scan_num_workers,
        )
        backend = "zarr"
    else:
        songs = _scan_npy_root(
            dataset_root=dataset_root,
            note_search_roots=note_search_roots,
            num_workers=scan_num_workers,
        )
        backend = "npy_dir"
    if not songs:
        raise FileNotFoundError(f"No songs found under Sonata dataset root: {dataset_root}")

    songs = _apply_subset(songs=songs, config=config)
    split_df = _build_song_split_df(songs=songs, seed=int(config.get("split_seed", 0)), ratios=dict(config.get("split_ratios", {})))
    split_map = dict(zip(split_df["song_id"].astype(str), split_df["split"].astype(str)))

    manifest_rows: list[dict[str, Any]] = []
    per_song_episode_limit = _positive_int(config.get("max_episodes"))
    control_timestep = float(config.get("control_timestep", 0.05))
    for song in songs:
        episode_count = song["num_episodes"]
        if per_song_episode_limit is not None:
            episode_count = min(episode_count, per_song_episode_limit)
        split = split_map.get(song["song_id"], "train")
        for episode_index in range(episode_count):
            record = ManifestRecord(
                song_id=str(song["song_id"]),
                episode_id=f"{song['song_id']}__ep{episode_index:05d}",
                split=split,
                backend=backend,
                dataset_root=str(dataset_root),
                song_key=str(song.get("song_key", song["song_id"])),
                song_path=str(song.get("song_path", "")),
                episode_index=int(episode_index),
                note_path=str(song.get("note_path", "")),
                control_timestep=control_timestep,
                num_steps=int(song["num_steps"]),
                action_dim=int(song["dims"].get("actions", 0)),
                goal_dim=int(song["dims"].get("goals", 0)),
                piano_state_dim=int(song["dims"].get("piano_states", 0)),
                hand_joint_dim=int(song["dims"].get("hand_joints", 0)),
                hand_fingertip_dim=int(song["dims"].get("hand_fingertips", 0)),
                joint_velocity_dim=int(song["dims"].get("joint_velocities", 0)),
                wrist_pose_dim=int(song["dims"].get("wrist_pose", 0)),
                hand_pose_dim=int(song["dims"].get("hand_pose", 0)),
                has_actions=bool(song["dims"].get("actions", 0) > 0),
                has_goals=bool(song["dims"].get("goals", 0) > 0),
                has_piano_states=bool(song["dims"].get("piano_states", 0) > 0),
                has_hand_joints=bool(song["dims"].get("hand_joints", 0) > 0),
                has_hand_fingertips=bool(song["dims"].get("hand_fingertips", 0) > 0),
                has_joint_velocities=bool(song["dims"].get("joint_velocities", 0) > 0),
                has_wrist_pose=bool(song["dims"].get("wrist_pose", 0) > 0),
                has_hand_pose=bool(song["dims"].get("hand_pose", 0) > 0),
            )
            manifest_rows.append(record.as_row())

    manifest_df = pd.DataFrame(manifest_rows)
    write_table(manifest_df, manifest_base)
    write_table(split_df, split_base)

    split_counts = manifest_df["split"].value_counts().sort_index().to_dict() if not manifest_df.empty else {}
    summary = {
        "dataset_root": str(dataset_root),
        "backend": backend,
        "num_songs": int(len(songs)),
        "num_episodes": int(len(manifest_df)),
        "note_files_found": int(sum(1 for song in songs if song.get("note_path"))),
        "splits": {key: int(value) for key, value in split_counts.items()},
        "songs": [str(song["song_id"]) for song in songs],
        "subset_mode": str(config.get("subset_mode", "all")),
    }
    write_json(summary, summary_path)
    logger.info("Indexed %d songs and %d episodes from %s", len(songs), len(manifest_df), dataset_root)
    return {"manifest_base": manifest_base, "split_base": split_base, "summary_path": summary_path}


def _scan_zarr_root(
    dataset_root: Path,
    note_search_roots: list[Path],
    num_workers: int = 0,
) -> list[dict[str, Any]]:
    song_dirs = sorted(path for path in dataset_root.iterdir() if path.is_dir() and (path / ".zgroup").exists())
    songs = _parallel_map_ordered(
        song_dirs,
        lambda song_dir: _scan_zarr_song(song_dir, note_search_roots),
        num_workers=num_workers,
    )
    return [song for song in songs if song is not None]


def _scan_npy_root(
    dataset_root: Path,
    note_search_roots: list[Path],
    num_workers: int = 0,
) -> list[dict[str, Any]]:
    candidates = [path for path in sorted(dataset_root.iterdir()) if path.is_dir()]
    if not candidates and any((dataset_root / f"{name}.npy").exists() for name in ARRAY_NAMES):
        candidates = [dataset_root]
    songs = _parallel_map_ordered(
        candidates,
        lambda song_dir: _scan_npy_song(song_dir, note_search_roots),
        num_workers=num_workers,
    )
    return [song for song in songs if song is not None]


def _scan_zarr_song(song_dir: Path, note_search_roots: list[Path]) -> dict[str, Any] | None:
    dims: dict[str, int] = {}
    num_episodes = 0
    num_steps = 0
    for array_name in ARRAY_NAMES:
        zarray_path = song_dir / array_name / ".zarray"
        if not zarray_path.exists():
            continue
        metadata = json.loads(zarray_path.read_text())
        shape = list(metadata.get("shape", []))
        if len(shape) >= 2:
            num_episodes = max(num_episodes, int(shape[0]))
            num_steps = max(num_steps, int(shape[1]))
        dims[array_name] = int(shape[2]) if len(shape) >= 3 else 1
    if not dims:
        return None
    song_id = song_dir.name
    return {
        "song_id": song_id,
        "song_key": song_id,
        "song_path": "",
        "note_path": str(_find_note_path(song_id=song_id, note_search_roots=note_search_roots) or ""),
        "num_episodes": int(num_episodes),
        "num_steps": int(num_steps),
        "dims": dims,
    }


def _scan_npy_song(song_dir: Path, note_search_roots: list[Path]) -> dict[str, Any] | None:
    dims: dict[str, int] = {}
    num_episodes = 0
    num_steps = 0
    for array_name in ARRAY_NAMES:
        shape = _discover_npy_shape(song_dir=song_dir, array_name=array_name)
        if shape is None:
            continue
        if len(shape) >= 2:
            num_steps = max(num_steps, int(shape[-2]))
        if len(shape) >= 3:
            num_episodes = max(num_episodes, int(shape[0]))
            dims[array_name] = int(shape[2])
        elif len(shape) == 2:
            num_episodes = max(num_episodes, 1)
            dims[array_name] = int(shape[1])
    if not dims:
        return None
    song_id = song_dir.name
    return {
        "song_id": song_id,
        "song_key": song_id,
        "song_path": str(song_dir.resolve()),
        "note_path": str(_find_note_path(song_id=song_id, note_search_roots=note_search_roots) or ""),
        "num_episodes": int(num_episodes),
        "num_steps": int(num_steps),
        "dims": dims,
    }


def _discover_npy_shape(song_dir: Path, array_name: str) -> tuple[int, ...] | None:
    direct_file = song_dir / f"{array_name}.npy"
    if direct_file.exists():
        array = np.load(direct_file, mmap_mode="r")
        return tuple(int(item) for item in array.shape)

    array_dir = song_dir / array_name
    if array_dir.is_dir():
        first_file = next((path for path in sorted(array_dir.iterdir()) if path.suffix == ".npy"), None)
        if first_file is not None:
            array = np.load(first_file, mmap_mode="r")
            return (len([path for path in array_dir.iterdir() if path.suffix == ".npy"]),) + tuple(int(item) for item in array.shape)
    return None


def _apply_subset(songs: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    mode = str(config.get("subset_mode", "all"))
    selected = list(songs)
    limit = _positive_int(config.get("debug_num_songs")) or _positive_int(config.get("max_songs"))
    if limit is None:
        return selected
    if mode == "head":
        return selected[:limit]
    if mode == "random":
        rng = random.Random(int(config.get("split_seed", 0)))
        rng.shuffle(selected)
        return selected[:limit]
    return selected[:limit] if mode != "all" else selected


def _build_song_split_df(songs: list[dict[str, Any]], seed: int, ratios: dict[str, Any]) -> pd.DataFrame:
    song_ids = [str(song["song_id"]) for song in songs]
    if not song_ids:
        return pd.DataFrame(columns=["song_id", "split", "num_episodes", "note_path"])

    ordered = list(song_ids)
    random.Random(seed).shuffle(ordered)
    split_names = ["train", "val", "test"]
    ratios = {name: float(ratios.get(name, 0.0)) for name in split_names}

    if len(ordered) == 1:
        assignments = {ordered[0]: "train"}
    elif len(ordered) <= 3:
        sequence = ["train", "val", "test"]
        assignments = {song_id: sequence[min(index, len(sequence) - 1)] for index, song_id in enumerate(ordered)}
    else:
        counts = _allocate_split_counts(total=len(ordered), ratios=ratios)
        assignments = {}
        cursor = 0
        for split_name in split_names:
            for song_id in ordered[cursor : cursor + counts[split_name]]:
                assignments[song_id] = split_name
            cursor += counts[split_name]
        for song_id in ordered[cursor:]:
            assignments[song_id] = "train"

    rows = []
    by_song = {str(song["song_id"]): song for song in songs}
    for song_id in song_ids:
        song = by_song[song_id]
        rows.append(
            {
                "song_id": song_id,
                "split": assignments.get(song_id, "train"),
                "num_episodes": int(song["num_episodes"]),
                "num_steps": int(song["num_steps"]),
                "note_path": str(song.get("note_path", "")),
            }
        )
    return pd.DataFrame(rows)


def _allocate_split_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    split_names = ["train", "val", "test"]
    raw = {name: ratios.get(name, 0.0) * total for name in split_names}
    counts = {name: int(math.floor(raw[name])) for name in split_names}
    remainder = total - sum(counts.values())
    order = sorted(split_names, key=lambda name: (raw[name] - counts[name], ratios.get(name, 0.0)), reverse=True)
    for index in range(remainder):
        counts[order[index % len(order)]] += 1
    if counts["train"] == 0:
        for name in ("val", "test"):
            if counts[name] > 1:
                counts[name] -= 1
                counts["train"] += 1
                break
        if counts["train"] == 0:
            counts["train"] = 1
            if counts["val"] > 0:
                counts["val"] -= 1
            elif counts["test"] > 0:
                counts["test"] -= 1
    return counts


def _find_note_path(song_id: str, note_search_roots: list[Path]) -> Path | None:
    candidate_stems = []
    for stem in (song_id, re.sub(r"_[0-9]+$", "", song_id)):
        if stem and stem not in candidate_stems:
            candidate_stems.append(stem)
    for root in note_search_roots:
        for stem in candidate_stems:
            for suffix in (".proto", ".mid", ".midi"):
                candidate = root / f"{stem}{suffix}"
                if candidate.exists():
                    return candidate.resolve()
    return None


def _positive_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _parallel_map_ordered(
    items: list[Path],
    fn,
    *,
    num_workers: int,
) -> list[dict[str, Any] | None]:
    if num_workers <= 1 or len(items) <= 1:
        return [fn(item) for item in items]
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        return list(executor.map(fn, items))
