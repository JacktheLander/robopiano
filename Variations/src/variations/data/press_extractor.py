from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PressCandidates:
    target_keys: np.ndarray
    hand_state: np.ndarray
    steps: np.ndarray
    num_keys_pressed: np.ndarray
    num_onsets: np.ndarray

    @property
    def count(self) -> int:
        return int(self.target_keys.shape[0])


def goal_fingerprint(target_keys: np.ndarray) -> bytes:
    goal = np.asarray(target_keys)[:88] > 0.5
    return np.packbits(goal.astype(np.bool_), bitorder="little").tobytes()


def _empty_candidates() -> PressCandidates:
    return PressCandidates(
        target_keys=np.zeros((0, 88), dtype=np.float32),
        hand_state=np.zeros((0, 76), dtype=np.float32),
        steps=np.zeros((0,), dtype=np.int32),
        num_keys_pressed=np.zeros((0,), dtype=np.int32),
        num_onsets=np.zeros((0,), dtype=np.int32),
    )


def extract_press_candidates(trajectory: dict[str, np.ndarray], threshold: float = 0.5) -> PressCandidates:
    required = ["goals", "piano_states", "hand_joints", "hand_fingertips"]
    missing = [name for name in required if name not in trajectory]
    if missing:
        raise KeyError(f"Trajectory is missing required arrays: {missing}")

    goals = np.asarray(trajectory["goals"])
    piano_states = np.asarray(trajectory["piano_states"])
    hand_joints = np.asarray(trajectory["hand_joints"])
    hand_fingertips = np.asarray(trajectory["hand_fingertips"])
    t = min(goals.shape[0], piano_states.shape[0], hand_joints.shape[0], hand_fingertips.shape[0])
    if t == 0:
        return _empty_candidates()

    goal = goals[:t, :88] > threshold
    played = piano_states[:t, :88] > threshold
    previous = np.concatenate([np.zeros((1, 88), dtype=bool), played[:-1]], axis=0)
    onset = played & ~previous
    exact = (played == goal).all(axis=1)
    nonempty = goal.any(axis=1)
    has_onset = onset.any(axis=1)
    keep = exact & nonempty & has_onset
    if not np.any(keep):
        return _empty_candidates()

    joints = hand_joints[:t][keep]
    fingertips = hand_fingertips[:t][keep]
    hand_state = np.concatenate([joints, fingertips], axis=1).astype(np.float32, copy=False)
    if hand_state.shape[1] != 76:
        raise ValueError(f"Expected hand_state width 76, got {hand_state.shape[1]}")

    target_keys = goal[keep].astype(np.float32, copy=False)
    steps = np.flatnonzero(keep).astype(np.int32)
    return PressCandidates(
        target_keys=target_keys,
        hand_state=hand_state,
        steps=steps,
        num_keys_pressed=goal[keep].sum(axis=1).astype(np.int32),
        num_onsets=onset[keep].sum(axis=1).astype(np.int32),
    )

