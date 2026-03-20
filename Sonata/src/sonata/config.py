from __future__ import annotations

import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import ast

try:  # pragma: no cover
    import yaml
except Exception:  # pragma: no cover
    yaml = None


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    raw_text = resolved.read_text()
    if yaml is not None:
        data = yaml.safe_load(raw_text)
    else:
        data = simple_yaml_load(raw_text)
    return data or {}


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def resolve_stage_config(stage: str, profile: str = "debug") -> Path:
    return (project_root() / "configs" / stage / f"{profile}.yaml").resolve()


def load_stage_config(
    stage: str,
    profile: str = "debug",
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path = Path(config_path).resolve() if config_path is not None else resolve_stage_config(stage, profile)
    config = load_yaml(path)
    config["config_path"] = str(path)
    config["config_stage"] = stage
    if overrides:
        config = deep_update(config, overrides)
    return config


def load_pipeline_config(profile: str = "debug", config_path: str | Path | None = None) -> dict[str, Any]:
    path = (
        Path(config_path).resolve()
        if config_path is not None
        else (project_root() / "configs" / "pipeline" / f"{profile}.yaml").resolve()
    )
    config = load_yaml(path)
    config["config_path"] = str(path)
    return config


def resolve_path(path_like: str | Path | None, base_dir: str | Path | None = None) -> Path | None:
    if path_like is None:
        return None
    if isinstance(path_like, str) and not path_like.strip():
        return None
    expanded = _expand_path_like(path_like)
    path = Path(expanded)
    if path.is_absolute():
        return path
    base = Path(base_dir).resolve() if base_dir is not None else project_root()
    return (base / path).resolve()


def _expand_path_like(path_like: str | Path) -> str:
    raw = str(path_like)
    expanded = os.path.expanduser(os.path.expandvars(raw))
    unresolved = re.findall(r"\$(?:[A-Za-z_][A-Za-z0-9_]*|\{[^}]+\})", expanded)
    if unresolved:
        names = ", ".join(unresolved)
        raise ValueError(f"Unresolved environment variable(s) in Sonata path: {names}")
    return expanded


def simple_yaml_load(text: str) -> dict[str, Any]:
    lines: list[tuple[int, str]] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        lines.append((indent, stripped))
    parsed, _ = _parse_mapping(lines, start=0, indent=0)
    return parsed


def _parse_mapping(lines: list[tuple[int, str]], start: int, indent: int) -> tuple[dict[str, Any], int]:
    mapping: dict[str, Any] = {}
    index = start
    while index < len(lines):
        line_indent, content = lines[index]
        if line_indent < indent:
            break
        if line_indent > indent:
            raise ValueError(f"Invalid indentation near: {content}")
        if content.startswith("- "):
            raise ValueError(f"Unexpected list item in mapping near: {content}")
        key, separator, value = content.partition(":")
        if separator != ":":
            raise ValueError(f"Invalid config line: {content}")
        key = key.strip()
        value = value.strip()
        if not value:
            next_index = index + 1
            if next_index >= len(lines) or lines[next_index][0] <= indent:
                mapping[key] = {}
                index += 1
                continue
            if lines[next_index][1].startswith("- "):
                parsed, index = _parse_list(lines, start=next_index, indent=lines[next_index][0])
            else:
                parsed, index = _parse_mapping(lines, start=next_index, indent=lines[next_index][0])
            mapping[key] = parsed
        else:
            mapping[key] = _parse_scalar(value)
            index += 1
    return mapping, index


def _parse_list(lines: list[tuple[int, str]], start: int, indent: int) -> tuple[list[Any], int]:
    values: list[Any] = []
    index = start
    while index < len(lines):
        line_indent, content = lines[index]
        if line_indent < indent:
            break
        if line_indent != indent or not content.startswith("- "):
            break
        values.append(_parse_scalar(content[2:].strip()))
        index += 1
    return values, index


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    if value.startswith("[") or value.startswith("{") or value.startswith(("'", '"')):
        normalized = (
            value.replace(": true", ": True")
            .replace(": false", ": False")
            .replace(": null", ": None")
        )
        return ast.literal_eval(normalized)
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value
