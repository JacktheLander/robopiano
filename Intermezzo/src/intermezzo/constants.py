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

# Fingertip order is right thumb/index/middle/ring/little, then left thumb/index/middle/ring/little.
# These are the reduced hand-state joint indices that belong to each finger, excluding wrist and forearm joints.
FINGER_JOINT_INDICES = (
 (18, 19, 20),
 (2, 3, 4, 5),
 (6, 7, 8, 9),
 (10, 11, 12, 13),
 (14, 15, 16, 17),
 (41, 42, 43),
 (25, 26, 27, 28),
 (29, 30, 31, 32),
 (33, 34, 35, 36),
 (37, 38, 39, 40),
)
