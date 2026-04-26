from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sonata.diffusion.note_surrogate import build_note_state_targets, fit_linear_note_surrogate
from sonata.diffusion.trainer import compose_final_action


def test_build_note_state_targets_tracks_hold_and_sustain() -> None:
    roll = np.zeros((3, 89), dtype=np.float32)
    roll[0, 0] = 1.0
    roll[1, 0] = 1.0
    roll[1, 2] = 1.0
    roll[2, 2] = 1.0
    roll[1:, 88] = 1.0
    previous = np.zeros((89,), dtype=np.float32)
    previous[0] = 1.0

    note_target, hold_target, sustain_target = build_note_state_targets(
        roll,
        action_horizon=3,
        previous_frame=previous,
    )

    assert note_target.shape == (3, 88)
    assert hold_target.shape == (3, 88)
    assert sustain_target.shape == (3, 1)
    np.testing.assert_array_equal(note_target[:, :3], np.asarray([[1, 0, 0], [1, 0, 1], [0, 0, 1]], dtype=np.float32))
    np.testing.assert_array_equal(hold_target[:, :3], np.asarray([[1, 0, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32))
    np.testing.assert_array_equal(sustain_target[:, 0], np.asarray([0, 1, 1], dtype=np.float32))


def test_fit_linear_note_surrogate_recovers_simple_note_mapping() -> None:
    def make_sample(actions: np.ndarray) -> dict[str, np.ndarray]:
        note_target = np.zeros((actions.shape[0], 88), dtype=np.float32)
        note_target[:, 0] = (actions[:, 0] > 0.0).astype(np.float32)
        note_target[:, 1] = (actions[:, 1] > 0.0).astype(np.float32)
        previous = np.concatenate([note_target[:1], note_target[:-1]], axis=0)
        hold_target = note_target * previous
        sustain_target = ((actions[:, :1] + actions[:, 1:2]) > 0.5).astype(np.float32)
        return {
            "action_target": actions.astype(np.float32),
            "note_target": note_target,
            "hold_target": hold_target,
            "sustain_target": sustain_target,
        }

    samples = [
        make_sample(np.asarray([[0.9, -0.9], [0.8, -0.8], [-0.9, 0.9], [-0.8, 0.8]], dtype=np.float32)),
        make_sample(np.asarray([[-0.8, -0.8], [0.7, 0.7], [0.7, 0.7], [-0.8, -0.8]], dtype=np.float32)),
        make_sample(np.asarray([[0.6, 0.6], [0.6, -0.6], [-0.6, -0.6], [-0.6, 0.6]], dtype=np.float32)),
    ]

    surrogate, _ = fit_linear_note_surrogate(samples, action_dim=2, ridge_lambda=1e-4)
    probe = torch.from_numpy(samples[1]["action_target"][None, :, :])
    outputs = surrogate(probe)

    predicted_note = (outputs["note_logits"] > 0.0).cpu().numpy().astype(np.float32)
    predicted_hold = (outputs["hold_logits"] > 0.0).cpu().numpy().astype(np.float32)
    predicted_sustain = (outputs["sustain_logits"] > 0.0).cpu().numpy().astype(np.float32)

    np.testing.assert_array_equal(predicted_note[0, :, :2], samples[1]["note_target"][:, :2])
    np.testing.assert_array_equal(predicted_hold[0, :, :2], samples[1]["hold_target"][:, :2])
    np.testing.assert_array_equal(predicted_sustain[0], samples[1]["sustain_target"])


def test_compose_final_action_respects_residual_mode() -> None:
    prior = torch.tensor([[[0.25, -0.25]]], dtype=torch.float32)
    model_output = torch.tensor([[[0.5, 0.5]]], dtype=torch.float32)

    residual_action = compose_final_action(prior, model_output, predict_residual=True)
    direct_action = compose_final_action(prior, model_output, predict_residual=False)

    assert torch.allclose(residual_action, torch.tensor([[[0.75, 0.25]]], dtype=torch.float32))
    assert torch.allclose(direct_action, model_output)
