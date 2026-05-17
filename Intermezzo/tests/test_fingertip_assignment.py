from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from intermezzo.fingertip_assignment import assign_active_fingertips  # noqa: E402
from intermezzo.kinematics import FakeHandKinematics  # noqa: E402


def _key_geometry() -> np.ndarray:
    out = np.zeros((88, 2), dtype=np.float32)
    out[:, 1] = np.arange(88, dtype=np.float32)
    return out


def test_assigns_active_keys_to_same_hand_fingertips_only() -> None:
    q = np.zeros((46,), dtype=np.float32)
    # Right fingertip 0 is near key 60, left fingertip 5 is near key 10.
    q[0:2] = [0.0, 60.0]
    q[23:25] = [0.0, 10.0]
    target = np.zeros((88,), dtype=np.float32)
    target[[10, 60]] = 1.0

    assignments = assign_active_fingertips(target, endpoint_hand_state=q, key_xy=_key_geometry(), kinematics=FakeHandKinematics())

    by_key = {item.key_index: item for item in assignments}
    assert by_key[10].hand == "left"
    assert by_key[10].fingertip_index >= 5
    assert by_key[60].hand == "right"
    assert by_key[60].fingertip_index < 5


def test_inactive_keys_do_not_create_assignments() -> None:
    target = np.zeros((88,), dtype=np.float32)
    target[60] = 1.0

    assignments = assign_active_fingertips(target, endpoint_hand_state=np.zeros((46,), dtype=np.float32), key_xy=_key_geometry(), kinematics=FakeHandKinematics())

    assert [item.key_index for item in assignments] == [60]
