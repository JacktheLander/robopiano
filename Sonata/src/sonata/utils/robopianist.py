from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def robopianist_package_dir() -> Path:
    return repo_root() / "robopianist"


def default_robopianist_search_roots() -> list[Path]:
    roots = [repo_root()]
    for candidate in (
        repo_root() / "anthony",
        repo_root() / "third_party",
        repo_root() / "external",
    ):
        if candidate.exists():
            roots.append(candidate)
    return roots


def resolve_robopianist_import_root(robopianist_root: str | Path | None = None) -> Path | None:
    candidates: list[Path] = []
    if robopianist_root is not None:
        candidates.append(Path(robopianist_root).expanduser().resolve())
    env_override = os.environ.get("ROBOPIANIST_ROOT", "").strip()
    if env_override:
        candidates.append(Path(env_override).expanduser().resolve())
    candidates.extend(default_robopianist_search_roots())

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        import_root = _candidate_import_root(candidate)
        if import_root is not None:
            return import_root
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
    import_root_str = str(import_root.resolve())
    if import_root_str not in sys.path:
        sys.path.insert(0, import_root_str)
    # Clear any namespace-only robopianist module so Python re-resolves from the repo root.
    existing = sys.modules.get("robopianist")
    if existing is not None and getattr(existing, "__file__", None) is None:
        sys.modules.pop("robopianist", None)
    return import_root


def format_robopianist_import_error(exc: Exception, robopianist_root: str | Path | None = None) -> str:
    ensure_local_robopianist_on_path(robopianist_root)
    missing: list[str] = []
    for module_name in ("robopianist.suite", "dm_control", "mujoco", "mujoco_utils", "note_seq"):
        if importlib.util.find_spec(module_name) is None:
            missing.append(module_name)
    if not missing:
        return str(exc)
    missing_text = ", ".join(missing)
    resolved_root = resolve_robopianist_import_root(robopianist_root)
    root_hint = (
        f"Resolved RoboPianist import root: {resolved_root}. "
        if resolved_root is not None
        else "No usable RoboPianist clone was discovered automatically. "
    )
    return (
        f"{exc}. Missing evaluation dependencies: {missing_text}. "
        f"{root_hint}"
        "Pass --robopianist-root /path/to/clone-root or set ROBOPIANIST_ROOT to the directory "
        "that contains the RoboPianist Python package. Install Sonata full requirements plus the "
        "local RoboPianist package/environment before running DM Control evaluation."
    )


def _candidate_import_root(candidate: Path) -> Path | None:
    # Candidate already points at the clone root, e.g. `/path/to/clone-root`.
    package_root = candidate / "robopianist"
    if _is_robopianist_package_root(package_root):
        return candidate
    # Candidate points at the package directory itself, e.g. `/path/to/clone-root/robopianist`.
    if _is_robopianist_package_root(candidate):
        return candidate.parent
    return None


def _is_robopianist_package_root(path: Path) -> bool:
    return (
        path.exists()
        and (path / "__init__.py").exists()
        and (path / "suite" / "__init__.py").exists()
    )
