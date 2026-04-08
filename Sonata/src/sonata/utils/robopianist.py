from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def anthony_root() -> Path:
    return repo_root() / "anthony"


def ensure_local_robopianist_on_path() -> Path | None:
    if importlib.util.find_spec("robopianist.suite") is not None:
        return None
    candidate = anthony_root()
    package_dir = candidate / "robopianist"
    if not package_dir.exists():
        return None
    candidate_str = str(candidate.resolve())
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)
    # Clear the namespace-only package from the repo root so Python re-resolves submodules.
    existing = sys.modules.get("robopianist")
    if existing is not None and getattr(existing, "__file__", None) is None:
        sys.modules.pop("robopianist", None)
    return candidate


def format_robopianist_import_error(exc: Exception) -> str:
    ensure_local_robopianist_on_path()
    missing: list[str] = []
    for module_name in ("robopianist.suite", "dm_control", "mujoco", "mujoco_utils", "note_seq"):
        if importlib.util.find_spec(module_name) is None:
            missing.append(module_name)
    if not missing:
        return str(exc)
    missing_text = ", ".join(missing)
    return (
        f"{exc}. Missing evaluation dependencies: {missing_text}. "
        "Install Sonata full requirements plus the local RoboPianist package/environment before running DM Control evaluation."
    )
