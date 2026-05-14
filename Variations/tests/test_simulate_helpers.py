from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
VARIATIONS_DIR = ROOT
for p in (SRC, VARIATIONS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from simulate.midi_keysets import (  # noqa: E402
    pitch_to_key_index,
    quantize_notes_to_target_keys,
    simulation_slug,
)
from simulate.render import rollout_variations_maestro_prediction  # noqa: E402


def test_pitch_to_key_index_piano_range():
    assert pitch_to_key_index(21) == 0
    assert pitch_to_key_index(108) == 87
    assert pitch_to_key_index(60) == 39
    assert pitch_to_key_index(20) is None
    assert pitch_to_key_index(109) is None


def test_quantize_overlap_and_max_steps():
    # One note active only on frame 1: [0.051, 0.15) overlaps t=1 with dt=0.05 -> [0.05, 0.1)
    dt = 0.05
    roll, meta = quantize_notes_to_target_keys(
        [(0.051, 0.15, 60)],
        control_timestep=dt,
        max_steps=5,
        duration_hint_s=1.0,
    )
    assert roll.shape == (5, 88)
    assert meta["num_steps"] == 5
    assert float(roll[0].sum()) == 0.0
    assert roll[1, 39] == 1.0


def test_quantize_ignores_out_of_range_pitch():
    dt = 0.05
    roll, _meta = quantize_notes_to_target_keys(
        [(0.0, 0.2, 10), (0.0, 0.2, 200), (0.0, 0.2, 60)],
        control_timestep=dt,
        max_steps=2,
    )
    # Valid note spans frames 0 and 1 relative to overlap with [t*dt, (t+1)*dt).
    assert roll[0, 39] == 1.0
    assert roll[1, 39] == 1.0
    assert float(roll.sum()) == 2.0


def test_simulation_slug():
    assert simulation_slug("foo/bar!") == "foo_bar"
    assert simulation_slug("normal_name") == "normal_name"


def test_variations_playback_does_not_inject_midi_key_states(monkeypatch, tmp_path):
    target_keys = np.zeros((3, 88), dtype=np.float32)
    target_keys[:, 40] = 1.0
    hand_joints = np.zeros((3, 46), dtype=np.float32)

    class DummyPhysics:
        def forward(self):
            pass

    class DummyPiano:
        activation = np.zeros((88,), dtype=bool)

        def _update_key_state(self, physics):
            pass

        def _update_key_color(self, physics):
            pass

    class DummyEnv:
        def reset(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr("simulate.render.write_goals_proto", lambda *args, **kwargs: tmp_path / "goals.proto")
    monkeypatch.setattr("simulate.render._load_env", lambda **kwargs: ("dummy-env", DummyEnv(), {}))
    monkeypatch.setattr("simulate.render._locate_task_physics_piano", lambda env: (object(), DummyPhysics(), DummyPiano()))
    monkeypatch.setattr("simulate.render._set_reduced_hand_qpos", lambda *args, **kwargs: 46)
    monkeypatch.setattr("simulate.render.render_frame", lambda *args, **kwargs: np.zeros((2, 2, 3), dtype=np.uint8))
    monkeypatch.setattr("simulate.render.write_video", lambda *args, **kwargs: (None, None, None))

    result = rollout_variations_maestro_prediction(
        target_keys=target_keys,
        hand_joints=hand_joints,
        song_name="dummy-song",
        output_dir=tmp_path,
    )

    assert result["midi_key_state_injection"] is False
    assert result["audio_source"] == "physical_piano_activation_after_pose_injection"
    assert result["against_goals"]["key_recall"] == 0.0
