from __future__ import annotations

NUM_PIANO_KEYS = 88
HAND_STATE_DIM = 46
REDUCED_HAND_DIM = 23

# Verified from RoboPianist reduced hand joint ordering:
# [right hand 23 joints, left hand 23 joints], each ending in forearm_tx, forearm_ty.
RIGHT_FOREARM_TY_INDEX = 22
LEFT_FOREARM_TY_INDEX = 45
FOREARM_TY_INDICES = (RIGHT_FOREARM_TY_INDEX, LEFT_FOREARM_TY_INDEX)

FOREARM_TY_MIN = 0.0
FOREARM_TY_MAX = 0.06
KEY_SPLIT_LEFT_RIGHT = 44
