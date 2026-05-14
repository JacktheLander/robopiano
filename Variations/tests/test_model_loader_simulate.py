from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
VARIATIONS_DIR = ROOT
for p in (SRC, VARIATIONS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def test_load_simulation_model_unknown_type():
    pytest.importorskip("torch")
    from simulate.model_loader import load_simulation_model

    with pytest.raises(ValueError, match="Unsupported model_type"):
        load_simulation_model("/tmp/none.pt", "transformer")


def test_load_simulation_model_rejects_fingerpred():
    pytest.importorskip("torch")
    from simulate.model_loader import load_simulation_model

    with pytest.raises(ValueError, match="FingerPred checkpoints predict 30D fingertip"):
        load_simulation_model("/tmp/none.pt", "fingerpred")


def test_online_eval_rejects_fingerpred():
    from variations.online_eval import normalize_model_type

    with pytest.raises(ValueError, match="fingerpred predicts 30D fingertip"):
        normalize_model_type("fingerpred")
