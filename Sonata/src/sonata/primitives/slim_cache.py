from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SEGMENT_INDEX_DEFAULTS: dict[str, Any] = {
    "chunk_path": "",
    "chunk_index": -1,
    "raw_chunk_path": "",
    "raw_chunk_index": -1,
    "gmr_target_name": "",
}
ONLINE_STORAGE_FORMATS = {"npz", "npz_shards"}


@dataclass(frozen=True)
class SlimCachePaths:
    root: Path
    feature_dir: Path
    gmr_target_dir: Path
    index_dir: Path
    manifest_dir: Path
    progress_dir: Path


def online_segment_processing_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("online_segment_processing", config.get("write_slim_cache", True)))


def save_raw_segment_chunks_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("save_raw_segment_chunks", config.get("write_full_segment_cache", False)))


def resolve_online_storage_format(config: dict[str, Any]) -> str:
    value = str(config.get("online_storage_format", "npz_shards"))
    if value not in ONLINE_STORAGE_FORMATS:
        raise ValueError(f"Unsupported Sonata online storage format: {value}")
    return value


def resolve_slim_cache_paths(output_dir: Path, config: dict[str, Any]) -> SlimCachePaths:
    root = output_dir / str(config.get("online_cache_dir", config.get("slim_cache_dir", "slim")))
    return SlimCachePaths(
        root=root,
        feature_dir=root / str(config.get("online_feature_dir", config.get("slim_feature_dir", "features"))),
        gmr_target_dir=root / str(config.get("online_gmr_target_dir", config.get("slim_gmr_target_dir", "gmr_targets"))),
        index_dir=root / str(config.get("online_index_dir", config.get("slim_index_dir", "index"))),
        manifest_dir=root / "manifests",
        progress_dir=root / "progress",
    )


def ensure_slim_dirs(paths: SlimCachePaths) -> None:
    for path in (
        paths.root,
        paths.feature_dir,
        paths.gmr_target_dir,
        paths.index_dir,
        paths.manifest_dir,
        paths.progress_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)


def slim_chunk_name(chunk_index: int) -> str:
    return f"slim_chunk_{int(chunk_index):05d}.npz"


def chunk_index_from_name(chunk_name: str | Path) -> int:
    stem = Path(chunk_name).stem
    return int(stem.rsplit("_", 1)[-1])


def is_slim_chunk_name(chunk_name: str | Path) -> bool:
    return Path(chunk_name).name.startswith("slim_chunk_")


def feature_chunk_path(paths: SlimCachePaths, chunk_name: str | Path) -> Path:
    return paths.feature_dir / Path(chunk_name).name


def gmr_target_chunk_path(paths: SlimCachePaths, chunk_name: str | Path) -> Path:
    return paths.gmr_target_dir / Path(chunk_name).name


def index_chunk_path(paths: SlimCachePaths, chunk_name: str | Path) -> Path:
    return paths.index_dir / f"{Path(chunk_name).stem}.csv"


def manifest_chunk_path(paths: SlimCachePaths, chunk_name: str | Path) -> Path:
    return paths.manifest_dir / f"{Path(chunk_name).stem}.json"


def compact_store_manifest_path(paths: SlimCachePaths) -> Path:
    return paths.root / "compact_store_manifest.json"


def episode_progress_path(paths: SlimCachePaths) -> Path:
    return paths.progress_dir / "episode_progress.jsonl"


def ensure_segment_index_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column, default in SEGMENT_INDEX_DEFAULTS.items():
        if column not in output.columns:
            output[column] = default
    if "chunk_index" in output.columns:
        output["chunk_index"] = output["chunk_index"].fillna(-1).astype(int)
    if "raw_chunk_index" in output.columns:
        output["raw_chunk_index"] = output["raw_chunk_index"].fillna(-1).astype(int)
    return output


def load_completed_episodes(paths: SlimCachePaths) -> set[str]:
    completed: set[str] = set()
    log_path = episode_progress_path(paths)
    if not log_path.exists():
        return completed
    with log_path.open() as handle:
        for raw_line in handle:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            payload = json.loads(raw_line)
            if payload.get("status") == "completed":
                completed.add(str(payload["episode_id"]))
    return completed


def append_episode_progress(paths: SlimCachePaths, payload: dict[str, Any]) -> None:
    log_path = episode_progress_path(paths)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")


def collect_slim_chunk_names(paths: SlimCachePaths, completed_only: bool = False) -> list[str]:
    names: set[str] = set()
    if paths.index_dir.exists():
        names.update(f"{path.stem}.npz" for path in paths.index_dir.glob("slim_chunk_*.csv"))
    if paths.feature_dir.exists():
        names.update(path.name for path in paths.feature_dir.glob("slim_chunk_*.npz"))
    if paths.gmr_target_dir.exists():
        names.update(path.name for path in paths.gmr_target_dir.glob("slim_chunk_*.npz"))
    if paths.manifest_dir.exists():
        names.update(f"{path.stem}.npz" for path in paths.manifest_dir.glob("slim_chunk_*.json"))
    ordered = sorted(names, key=chunk_index_from_name)
    if completed_only:
        ordered = [name for name in ordered if slim_chunk_complete(paths, name)]
    return ordered


def list_incomplete_slim_chunks(paths: SlimCachePaths) -> list[str]:
    return [name for name in collect_slim_chunk_names(paths, completed_only=False) if not slim_chunk_complete(paths, name)]


def load_slim_index_table(paths: SlimCachePaths) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for chunk_name in collect_slim_chunk_names(paths, completed_only=True):
        chunk_path = index_chunk_path(paths, chunk_name)
        if chunk_path.exists():
            frames.append(pd.read_csv(chunk_path))
    if not frames:
        return pd.DataFrame()
    return ensure_segment_index_columns(pd.concat(frames, ignore_index=True))


def compose_segment_index(existing_df: pd.DataFrame, slim_df: pd.DataFrame) -> pd.DataFrame:
    existing = ensure_segment_index_columns(existing_df)
    slim = ensure_segment_index_columns(slim_df)
    if slim.empty:
        return existing.reset_index(drop=True)
    if existing.empty:
        base = slim
    else:
        slim_ids = slim["segment_id"].astype(str)
        base = pd.concat(
            [existing.loc[~existing["segment_id"].astype(str).isin(slim_ids)], slim],
            ignore_index=True,
        )
    sort_columns = [column for column in ("song_id", "episode_id", "onset_step", "end_step", "segment_id") if column in base.columns]
    if sort_columns:
        base = base.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    else:
        base = base.reset_index(drop=True)
    return base


def next_slim_chunk_index(paths: SlimCachePaths) -> int:
    names = collect_slim_chunk_names(paths, completed_only=False)
    if not names:
        return 0
    return max(chunk_index_from_name(name) for name in names) + 1


def slim_chunk_complete(paths: SlimCachePaths, chunk_name: str | Path) -> bool:
    manifest_path = manifest_chunk_path(paths, chunk_name)
    if not manifest_path.exists():
        return False
    payload = json.loads(manifest_path.read_text())
    return (
        payload.get("status") == "completed"
        and feature_chunk_path(paths, chunk_name).exists()
        and gmr_target_chunk_path(paths, chunk_name).exists()
        and index_chunk_path(paths, chunk_name).exists()
    )


def build_gmr_target(arrays: dict[str, np.ndarray | None], config: dict[str, Any], resample_fn=None) -> tuple[np.ndarray, str]:
    del resample_fn
    from sonata.primitives.features import build_gmr_target_from_arrays

    return build_gmr_target_from_arrays(arrays=arrays, config=config)


def write_slim_chunk(
    paths: SlimCachePaths,
    chunk_name: str,
    segment_rows: list[dict[str, Any]],
    feature_matrix: np.ndarray,
    feature_names: list[str],
    gmr_targets: np.ndarray,
    target_names: list[str],
    source_raw_chunk: str | None = None,
    migrated: bool = False,
) -> list[dict[str, Any]]:
    ensure_slim_dirs(paths)
    if slim_chunk_complete(paths, chunk_name):
        return pd.read_csv(index_chunk_path(paths, chunk_name)).to_dict(orient="records")

    if not segment_rows:
        raise ValueError("Cannot write an empty slim chunk.")
    if feature_matrix.shape[0] != len(segment_rows):
        raise ValueError("Slim feature matrix row count does not match segment rows.")
    if gmr_targets.shape[0] != len(segment_rows):
        raise ValueError("Slim GMR target row count does not match segment rows.")
    if len(target_names) != len(segment_rows):
        raise ValueError("Slim target-name count does not match segment rows.")

    updated_rows: list[dict[str, Any]] = []
    for index, row in enumerate(segment_rows):
        updated = dict(row)
        updated["chunk_path"] = chunk_name
        updated["chunk_index"] = index
        updated["gmr_target_name"] = str(target_names[index])
        updated_rows.append(updated)

    rows_df = ensure_segment_index_columns(pd.DataFrame(updated_rows))
    segment_ids = rows_df["segment_id"].astype(str).to_numpy(dtype=object)

    _save_npz_atomic(
        feature_chunk_path(paths, chunk_name),
        segment_ids=segment_ids,
        feature_matrix=np.asarray(feature_matrix, dtype=np.float32),
        feature_names=np.asarray(feature_names, dtype=object),
    )
    _save_npz_atomic(
        gmr_target_chunk_path(paths, chunk_name),
        segment_ids=segment_ids,
        gmr_targets=np.asarray(gmr_targets, dtype=np.float32),
        target_names=np.asarray(target_names, dtype=object),
    )
    _write_csv_atomic(index_chunk_path(paths, chunk_name), rows_df)
    manifest_payload = {
        "chunk_name": chunk_name,
        "feature_path": str(feature_chunk_path(paths, chunk_name).resolve()),
        "gmr_target_path": str(gmr_target_chunk_path(paths, chunk_name).resolve()),
        "index_path": str(index_chunk_path(paths, chunk_name).resolve()),
        "num_segments": int(len(rows_df)),
        "num_features": int(feature_matrix.shape[1]) if feature_matrix.ndim == 2 else 0,
        "gmr_horizon": int(gmr_targets.shape[1]) if gmr_targets.ndim >= 2 else 0,
        "gmr_dim": int(gmr_targets.shape[2]) if gmr_targets.ndim == 3 else 0,
        "source_raw_chunk": source_raw_chunk or "",
        "migrated": bool(migrated),
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json_atomic(manifest_chunk_path(paths, chunk_name), manifest_payload)
    verify_slim_chunk(paths, chunk_name, segment_ids=segment_ids)
    manifest_payload.update(chunk_storage_bytes(paths, chunk_name))
    _write_json_atomic(manifest_chunk_path(paths, chunk_name), manifest_payload)
    return updated_rows


def verify_slim_chunk(paths: SlimCachePaths, chunk_name: str, segment_ids: np.ndarray) -> None:
    feature_bundle = np.load(feature_chunk_path(paths, chunk_name), allow_pickle=True)
    gmr_bundle = np.load(gmr_target_chunk_path(paths, chunk_name), allow_pickle=True)
    stored_feature_ids = np.asarray(feature_bundle["segment_ids"], dtype=object)
    stored_target_ids = np.asarray(gmr_bundle["segment_ids"], dtype=object)
    if not np.array_equal(stored_feature_ids, segment_ids):
        raise ValueError(f"Slim feature ids do not match expected ids for {chunk_name}")
    if not np.array_equal(stored_target_ids, segment_ids):
        raise ValueError(f"Slim GMR ids do not match expected ids for {chunk_name}")


def chunk_storage_bytes(paths: SlimCachePaths, chunk_name: str | Path) -> dict[str, int]:
    sizes = {
        "feature_bytes": _safe_file_size(feature_chunk_path(paths, chunk_name)),
        "gmr_target_bytes": _safe_file_size(gmr_target_chunk_path(paths, chunk_name)),
        "index_bytes": _safe_file_size(index_chunk_path(paths, chunk_name)),
        "manifest_bytes": _safe_file_size(manifest_chunk_path(paths, chunk_name)),
    }
    sizes["total_bytes"] = int(sum(sizes.values()))
    return sizes


def summarize_slim_cache(paths: SlimCachePaths) -> dict[str, Any]:
    chunk_names = collect_slim_chunk_names(paths, completed_only=True)
    manifests: list[dict[str, Any]] = []
    for chunk_name in chunk_names:
        chunk_manifest = manifest_chunk_path(paths, chunk_name)
        if chunk_manifest.exists():
            manifests.append(json.loads(chunk_manifest.read_text()))
    feature_dim = next((int(item.get("num_features", 0)) for item in manifests if int(item.get("num_features", 0)) > 0), 0)
    gmr_horizon = next((int(item.get("gmr_horizon", 0)) for item in manifests if int(item.get("gmr_horizon", 0)) > 0), 0)
    gmr_dim = next((int(item.get("gmr_dim", 0)) for item in manifests if int(item.get("gmr_dim", 0)) > 0), 0)
    return {
        "num_chunks": int(len(chunk_names)),
        "num_segments": int(sum(int(item.get("num_segments", 0)) for item in manifests)),
        "feature_dim": feature_dim,
        "gmr_horizon": gmr_horizon,
        "gmr_dim": gmr_dim,
        "total_bytes_on_disk": int(tree_storage_bytes(paths.root)),
        "incomplete_chunks": list_incomplete_slim_chunks(paths),
    }


def read_compact_store_manifest(paths: SlimCachePaths) -> dict[str, Any]:
    path = compact_store_manifest_path(paths)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def write_compact_store_manifest(paths: SlimCachePaths, payload: dict[str, Any]) -> None:
    ensure_slim_dirs(paths)
    _write_json_atomic(compact_store_manifest_path(paths), payload)


def _save_npz_atomic(path: Path, **arrays: Any) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp_path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    tmp_path.replace(path)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp_path.replace(path)


def _write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def _safe_file_size(path: Path) -> int:
    return int(path.stat().st_size) if path.exists() else 0


def tree_storage_bytes(root: Path) -> int:
    if not root.exists():
        return 0
    return int(sum(path.stat().st_size for path in root.rglob("*") if path.is_file()))
