from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from variations.losses.supervised_pose_loss import supervised_pose_loss
from variations.models.mlp_baseline import PressPoseMLP


def test_press_pose_mlp_forward_backward_and_checkpoint(tmp_path):
    batch = 4
    model = PressPoseMLP(input_dim=88, output_dim=46, hidden_dims=[32, 32], dropout=0.0)
    target_keys = torch.randn(batch, 88)
    target = torch.randn(batch, 46)
    pred = model(target_keys)
    assert pred.shape == (batch, 46)
    assert torch.isfinite(pred).all()
    loss = supervised_pose_loss(pred, target)["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    path = tmp_path / "checkpoint.pt"
    torch.save({"model_state_dict": model.state_dict()}, path)
    loaded = PressPoseMLP(input_dim=88, output_dim=46, hidden_dims=[32, 32], dropout=0.0)
    loaded.load_state_dict(torch.load(path, map_location="cpu")["model_state_dict"])
    assert loaded(target_keys).shape == (batch, 46)
