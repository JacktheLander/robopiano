from __future__ import annotations

from pathlib import Path
import sys


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def ensure_repo_paths() -> Path:
    """Make sibling project packages importable without relying on editable installs."""
    sys.dont_write_bytecode = True
    repo = repo_root()
    for path in (
        repo / "Bagatelle" / "src",
        repo / "Intermezzo" / "src",
        repo / "Variations" / "src",
        repo / "Variations",
        repo / "partita" / "src",
        repo,
    ):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    return repo
