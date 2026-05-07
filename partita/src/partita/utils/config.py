from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if yaml is None:
        raise RuntimeError("PyYAML is required to read Partita config files.")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    data["_config_path"] = str(config_path)
    data["_repo_root"] = str(find_repo_root(config_path))
    return data


def find_repo_root(start: str | Path) -> Path:
    path = Path(start).resolve()
    if path.is_file():
        path = path.parent
    for candidate in [path, *path.parents]:
        if (candidate / "partita").exists() and (candidate / "robopianist").exists():
            return candidate
        if candidate.name == "partita":
            return candidate.parent
    return Path.cwd().resolve()


def output_root(config: dict[str, Any]) -> Path:
    root = Path(config.get("outputs", {}).get("root", "partita/outputs"))
    if not root.is_absolute():
        root = Path(config.get("_repo_root", Path.cwd())) / root
    return root


def experiment_name(config: dict[str, Any]) -> str:
    return str(config.get("experiment_name", "partita_debug"))
