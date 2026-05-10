from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def find_repo_root(start: str | Path) -> Path:
    path = Path(start).resolve()
    if path.is_file():
        path = path.parent
    for candidate in [path, *path.parents]:
        if (candidate / "Variations").exists() and (candidate / "robopianist").exists():
            return candidate
        if candidate.name == "Variations":
            return candidate.parent
    return Path.cwd().resolve()


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if yaml is None:
        raise RuntimeError("PyYAML is required to read Variations config files.")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    data["_config_path"] = str(config_path)
    data["_repo_root"] = str(find_repo_root(config_path))
    return data


def save_config(path: str | Path, config: dict[str, Any]) -> None:
    if yaml is None:
        from variations.utils.io import save_json

        save_json(path, config)
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def repo_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(config.get("_repo_root", Path.cwd())).resolve() / path


def experiment_name(config: dict[str, Any]) -> str:
    return str(config.get("experiment_name", "debug"))


def rp1m_root(config: dict[str, Any]) -> Path:
    value = os.environ.get("RP1M_ROOT") or os.environ.get("RP1M_300_ROOT") or config.get("rp1m_root")
    if value is None:
        raise KeyError("Set rp1m_root in config or export RP1M_ROOT.")
    return Path(value)


def variations_output_root(config: dict[str, Any]) -> Path:
    value = os.environ.get("VARIATIONS_OUTPUT_ROOT")
    if value:
        return Path(value)
    return repo_path(config, config.get("outputs", {}).get("root", "Variations/outputs"))


def extraction_root(config: dict[str, Any]) -> Path:
    if config.get("extraction_root"):
        output_env = os.environ.get("VARIATIONS_OUTPUT_ROOT")
        if output_env:
            configured = Path(str(config["extraction_root"]))
            parts = configured.parts
            if len(parts) >= 4 and parts[-3:] == ("outputs", "extraction", parts[-1]):
                return Path(output_env) / "extraction" / parts[-1]
            if len(parts) >= 3 and parts[-2] == "extraction":
                return Path(output_env) / "extraction" / parts[-1]
        return repo_path(config, config["extraction_root"])
    return variations_output_root(config) / "extraction" / experiment_name(config)


def diffusion_run_root(config: dict[str, Any]) -> Path:
    output_root = Path(os.environ.get("VARIATIONS_DIFFUSION_ROOT", ""))
    if not str(output_root):
        variations_root = os.environ.get("VARIATIONS_OUTPUT_ROOT")
        if variations_root:
            output_root = Path(variations_root) / "diffusion"
        else:
            output_root = repo_path(config, config.get("output_root", "Variations/outputs/diffusion"))
    run_name = str(config.get("run_name") or experiment_name(config))
    return output_root / run_name


def deep_get(config: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = config
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur
