from __future__ import annotations

import numpy as np

from etude.robopianist.state_mapping import StateMapping


class DummyActionSpec:
    minimum = -np.ones(10, dtype=np.float32)
    maximum = np.ones(10, dtype=np.float32)


def test_state_mapping_extracts_46d_and_clips_action() -> None:
    mapping = StateMapping.from_action_spec(
        DummyActionSpec(),
        action_indices=list(range(10)),
    )
    obs = {
        "qpos": np.arange(60, dtype=np.float32),
        "qvel": np.arange(60, dtype=np.float32) * 0.1,
    }
    q = mapping.extract_q(obs)
    qdot = mapping.extract_qdot(obs)
    action = mapping.action_from_joint_command(np.full(46, 5.0, dtype=np.float32))
    assert q.shape == (46,)
    assert qdot.shape == (46,)
    assert action.shape == (10,)
    assert np.all(action <= DummyActionSpec.maximum)
    assert np.all(action >= DummyActionSpec.minimum)


def test_state_mapping_json_roundtrip(tmp_path) -> None:
    mapping = StateMapping.from_action_spec(DummyActionSpec())
    path = tmp_path / "mapping.json"
    mapping.save(path)
    loaded = StateMapping.load(path)
    assert loaded.qpos_indices_46 == list(range(46))
    assert loaded.action_low.shape == (10,)
