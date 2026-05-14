"""Regression: unset VARIATIONS_DIFFUSION_ROOT must not resolve to cwd ('.') — see diffusion_run_root."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from variations.utils.config import diffusion_run_root, load_config


def test_diffusion_run_root_without_diffusion_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("VARIATIONS_DIFFUSION_ROOT", raising=False)
    monkeypatch.delenv("VARIATIONS_OUTPUT_ROOT", raising=False)
    cfg_path = ROOT / "configs" / "diffusion" / "debug.yaml"
    config = load_config(cfg_path)
    config["_repo_root"] = str(tmp_path)
    got = diffusion_run_root(config)
    assert got.name == "debug"
    assert "outputs" in got.parts
    assert got.parts[-2] == "diffusion"
