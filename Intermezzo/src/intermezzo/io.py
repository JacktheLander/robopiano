from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any

import numpy as np


def filesystem_slug(text: str, *, max_len: int = 96) -> str:
    value = str(text).strip().replace(" ", "_")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    value = value.strip("_") or "intermezzo"
    return value[:max_len]


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def create_unique_run_dir(
    output_root: str | Path,
    *,
    run_name: str | None = None,
    prefix: str = "intermezzo",
    max_attempts: int = 1000,
) -> Path:
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    base_name = filesystem_slug(run_name) if run_name else f"{filesystem_slug(prefix)}_{utc_timestamp()}"
    for attempt in range(max(int(max_attempts), 1)):
        candidate = root / (base_name if attempt == 0 else f"{base_name}_{attempt:03d}")
        try:
            candidate.mkdir(parents=False, exist_ok=False)
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError(f"Could not create a unique Intermezzo run directory under {root}")


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def atomic_save_json(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, target)
    return target


def atomic_save_npz(path: str | Path, **payload: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.stem}.tmp{target.suffix}")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, target)
    return target
