from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def atomic_write_text(path: str | Path, text: str) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    fd, tmp = tempfile.mkstemp(prefix=f".{p.name}.", dir=str(p.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def save_json(path: str | Path, obj: Any) -> None:
    text = json.dumps(obj, indent=2, sort_keys=True)
    atomic_write_text(path, text + "\n")


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_csv(path: str | Path, rows: Iterable[Mapping[str, Any]], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(str(key))
        fieldnames = keys
    p = Path(path)
    ensure_dir(p.parent)
    fd, tmp = tempfile.mkstemp(prefix=f".{p.name}.", dir=str(p.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe_stem(value: str) -> str:
    out = []
    for ch in str(value):
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    stem = "".join(out).strip("._")
    return stem or "song"

