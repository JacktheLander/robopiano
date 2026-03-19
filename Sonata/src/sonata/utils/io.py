from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def write_json(payload: dict[str, Any], path: str | Path) -> Path:
    resolved = Path(path).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return resolved


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def write_table(df: pd.DataFrame, path_without_suffix: str | Path) -> dict[str, Path]:
    base = Path(path_without_suffix).resolve()
    base.parent.mkdir(parents=True, exist_ok=True)
    csv_path = base.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    outputs = {"csv": csv_path}
    try:
        parquet_path = base.with_suffix(".parquet")
        df.to_parquet(parquet_path, index=False)
        outputs["parquet"] = parquet_path
    except Exception:
        pass
    return outputs


def read_table(path_without_suffix: str | Path) -> pd.DataFrame:
    base = Path(path_without_suffix)
    parquet_path = base.with_suffix(".parquet")
    csv_path = base.with_suffix(".csv")
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"No table found for base path: {base}")


def append_csv_row(path: str | Path, fieldnames: list[str], row: dict[str, Any]) -> None:
    resolved = Path(path).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    exists = resolved.exists()
    with resolved.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_npz(path: str | Path, **arrays: Any) -> Path:
    resolved = Path(path).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(resolved, **arrays)
    return resolved
