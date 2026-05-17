from __future__ import annotations

from types import ModuleType, SimpleNamespace
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from intermezzo.online_eval import RolloutConfig, press_events_from_roll, rollout_hand_targets_headless, score_rollout  # noqa: E402


def test_press_event_matching_counts_misses_mispresses_and_timing() -> None:
    target = np.zeros((8, 88), dtype=np.float32)
    target[2:4, 10] = 1.0
    target[5:7, 20] = 1.0
    played = np.zeros_like(target)
    played[3:5, 10] = 1.0
    played[6:7, 30] = 1.0

    score = score_rollout(
        target_keys=target,
        played_keys=played,
        dt=0.05,
        threshold=0.5,
        timing_tolerance_s=0.075,
    )

    assert score["target_press_events"] == 2
    assert score["played_press_events"] == 2
    assert score["matched_press_events"] == 1
    assert score["missed_key_presses"] == 1
    assert score["mispresses"] == 1
    assert score["timing_abs_error_mean_s"] == pytest.approx(0.05)


def test_press_events_use_onsets_not_sustained_frames() -> None:
    roll = np.zeros((5, 88), dtype=np.float32)
    roll[1:4, 7] = 1.0

    events = press_events_from_roll(roll, dt=0.05, threshold=0.5)

    assert [(event.frame, event.key) for event in events] == [(1, 7)]


def test_rollout_injects_hand_qpos_and_steps_physics_for_contacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    applied_poses: list[np.ndarray] = []
    load_kwargs_seen: dict[str, object] = {}

    class FakeEnv:
        def __init__(self) -> None:
            self.reset_count = 0
            self.step_count = 0

        def reset(self) -> None:
            self.reset_count += 1

        def step(self, _action):
            self.step_count += 1
            return SimpleNamespace(last=lambda: False)

        def action_spec(self):
            return SimpleNamespace(shape=(3,), dtype=np.float32)

        def close(self) -> None:
            pass

    class FakePhysics:
        def __init__(self) -> None:
            self.forward_count = 0

        def forward(self) -> None:
            self.forward_count += 1

    class FakePiano:
        def __init__(self) -> None:
            self.activation = np.zeros(88, dtype=np.float32)

        def _update_key_state(self, _physics) -> None:
            self.activation[:] = 0.0
            self.activation[12] = 1.0

        def _update_key_color(self, _physics) -> None:
            pass

    fake_env = FakeEnv()
    fake_physics = FakePhysics()
    fake_piano = FakePiano()
    fake_task = SimpleNamespace()

    fake_rollout = ModuleType("partita.evaluation.rollout")
    fake_rollout.candidate_environment_names = lambda name: [name]

    def fake_write_goals_proto(_keys, path, *, dt, title):
        Path(path).write_bytes(b"fake proto")
        return Path(path)

    def fake_load_env(**kwargs):
        load_kwargs_seen.update(kwargs)
        return "fake-env", fake_env, {}

    def fake_locate(_env):
        return fake_task, fake_physics, fake_piano

    def fake_set_qpos(_task, _physics, hand_qpos):
        applied_poses.append(np.asarray(hand_qpos, dtype=np.float32).copy())
        return int(np.asarray(hand_qpos).size)

    fake_rollout.write_goals_proto = fake_write_goals_proto
    fake_rollout._load_env = fake_load_env
    fake_rollout._locate_task_physics_piano = fake_locate
    fake_rollout._set_reduced_hand_qpos = fake_set_qpos
    fake_rollout._capture_piano_activation = lambda _env: fake_piano.activation

    monkeypatch.setitem(sys.modules, "partita", ModuleType("partita"))
    monkeypatch.setitem(sys.modules, "partita.evaluation", ModuleType("partita.evaluation"))
    monkeypatch.setitem(sys.modules, "partita.evaluation.rollout", fake_rollout)

    hand_targets = np.arange(92, dtype=np.float32).reshape(2, 46)
    target_keys = np.zeros((2, 88), dtype=np.float32)
    target_keys[:, 12] = 1.0

    result = rollout_hand_targets_headless(
        hand_targets=hand_targets,
        target_keys=target_keys,
        output_dir=tmp_path,
        label="fake",
        config=RolloutConfig(),
    )

    assert fake_env.step_count == 2
    assert fake_physics.forward_count == 2
    assert len(applied_poses) == 2
    np.testing.assert_array_equal(applied_poses[0], hand_targets[0])
    np.testing.assert_array_equal(applied_poses[1], hand_targets[1])
    assert result["control_mode"] == "direct_hand_qpos_pose_injection"
    assert result["actions_executed"] == 0
    assert result["pose_frames_applied"] == 2
    assert result["physics_steps_applied"] == 2
    assert result["restored_hand_joint_count"] == 46
    assert result["played_keys_shape"] == [2, 88]
    extra_task_kwargs = load_kwargs_seen["extra_task_kwargs"]
    assert isinstance(extra_task_kwargs, dict)
    assert extra_task_kwargs["disable_hand_collisions"] is False
