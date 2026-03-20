from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class ManifestRecord:
    song_id: str
    episode_id: str
    split: str
    backend: str
    dataset_root: str
    song_key: str
    song_path: str
    episode_index: int
    note_path: str
    control_timestep: float
    num_steps: int
    action_dim: int
    goal_dim: int
    piano_state_dim: int
    hand_joint_dim: int
    hand_fingertip_dim: int
    joint_velocity_dim: int
    wrist_pose_dim: int
    hand_pose_dim: int
    has_actions: bool
    has_goals: bool
    has_piano_states: bool
    has_hand_joints: bool
    has_hand_fingertips: bool
    has_joint_velocities: bool
    has_wrist_pose: bool
    has_hand_pose: bool

    def as_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EpisodeRecord:
    song_id: str
    episode_id: str
    split: str
    note_path: Path | None
    control_timestep: float
    actions: np.ndarray | None
    goals: np.ndarray | None
    piano_states: np.ndarray | None
    hand_joints: np.ndarray | None
    joint_velocities: np.ndarray | None
    hand_fingertips: np.ndarray | None
    wrist_pose: np.ndarray | None
    hand_pose: np.ndarray | None


@dataclass(slots=True)
class ScoreEvent:
    event_id: str
    song_id: str
    episode_id: str
    onset_step: int
    end_step: int
    start_time_sec: float
    end_time_sec: float
    key_numbers: tuple[int, ...]
    chord_size: int
    key_center: float
    inter_onset_steps: int
    source: str

    def as_row(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["key_numbers_json"] = json.dumps(list(self.key_numbers))
        payload.pop("key_numbers")
        return payload


@dataclass(slots=True)
class SegmentRecord:
    segment_id: str
    song_id: str
    episode_id: str
    onset_step: int
    end_step: int
    duration_steps: int
    segment_source: str
    score_event_id: str
    key_signature: str
    chunk_path: str
    chunk_index: int
    heuristic_family: str
    motion_energy: float
    chord_size: int
    key_center: float
    start_state_norm: float
    end_state_norm: float
    score_context_json: str

    def as_row(self) -> dict[str, Any]:
        return asdict(self)
