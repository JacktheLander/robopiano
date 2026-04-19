from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def resolve_robopianist_import_root(robopianist_root: str | Path | None = None) -> Path | None:
    candidates: list[Path] = []
    if robopianist_root is not None:
        candidates.append(Path(robopianist_root).expanduser().resolve())
    env_override = os.environ.get("ROBOPIANIST_ROOT", "").strip()
    if env_override:
        candidates.append(Path(env_override).expanduser().resolve())
    candidates.append(repo_root())

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        package_root = candidate / "robopianist"
        if (package_root / "__init__.py").exists() and (package_root / "suite" / "__init__.py").exists():
            return candidate
        if (candidate / "__init__.py").exists() and (candidate / "suite" / "__init__.py").exists():
            return candidate.parent
    return None


def ensure_local_robopianist_on_path(robopianist_root: str | Path | None = None) -> Path | None:
    try:
        if importlib.util.find_spec("robopianist.suite") is not None:
            return None
    except ModuleNotFoundError:
        pass
    import_root = resolve_robopianist_import_root(robopianist_root)
    if import_root is None:
        return None
    resolved = str(import_root.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
    return import_root


def format_robopianist_import_error(exc: Exception, robopianist_root: str | Path | None = None) -> str:
    ensure_local_robopianist_on_path(robopianist_root)
    missing: list[str] = []
    for module_name in ("robopianist.suite", "dm_control", "mujoco", "mujoco_utils", "note_seq"):
        if importlib.util.find_spec(module_name) is None:
            missing.append(module_name)
    if not missing:
        return str(exc)
    return (
        f"{exc}. Missing evaluation dependencies: {', '.join(missing)}. "
        "Install RoboPianist and DM Control dependencies before running rollout evaluation."
    )
