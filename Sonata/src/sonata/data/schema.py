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
    coarse_family: str
    motion_energy: float
    chord_size: int
    key_center: float
    start_state_norm: float
    end_state_norm: float
    score_context_json: str
    proposal_size: int = 1
    proposal_span_steps: int = 0
    boundary_energy: float = 0.0
    boundary_alignment_score: float = 0.0
    duplicate_iou: float = 0.0
    merge_count: int = 0
    split_count: int = 0
    target_key_count: int = 0
    target_key_signature: str = ""
    target_onset_step: int = -1
    next_onset_gap_steps: int = -1
    truncated_by_next_onset: bool = False
    raw_chunk_path: str = ""
    raw_chunk_index: int = -1
    gmr_target_name: str = ""
    causal_segment: bool = False
    segment_alignment: str = ""
    inactive_start: bool = False
    activation_after_start: bool = False
    contact_near_onset: bool = False
    causal_press_score: float = 0.0
    rejection_reason: str = ""
    target_key_ids: str = "[]"
    target_key_ids_json: str = "[]"
    chord_center_key_id: float = 0.0
    chord_center_key_id_normalized: float = 0.0
    chord_span_semitones: float = 0.0
    interval_pattern: str = "[]"
    interval_pattern_bucket: str = "none"
    white_black_pattern: str = "[]"
    key_world_positions: str = "[]"
    wrist_world_position: str = "[]"
    wrist_world_orientation: str = "[]"
    wrist_velocity: str = "[]"
    wrist_to_chord_center_offset: str = "[]"
    wrist_to_each_target_key_offset: str = "[]"
    relative_wrist_anchor: str = "[]"
    fingertip_world_positions: str = "[]"
    fingertip_to_target_key_offsets: str = "[]"
    nearest_finger_to_each_target_key: str = "[]"
    nearest_finger_labels: str = "[]"
    contact_finger_ids: str = "[]"
    finger_set_id: str = "unknown"
    finger_set: str = "unknown"
    fingertip_height_above_key: str = "[]"
    lateral_fingertip_key_offsets: str = "[]"
    start_joint_state: str = "[]"
    start_joint_velocity: str = "[]"
    normalized_joint_state: str = "[]"
    hand_side: str = "unknown"
    key_inactive_at_segment_start: bool = False
    key_activation_onset_step: int = -1
    contact_step: int = -1
    duration_bucket: int = 0
    dynamics_value: float = 0.0
    dynamics_bucket: int = 0
    motion_family: str = ""
    primitive_frame_mode: str = "absolute"

    def as_row(self) -> dict[str, Any]:
        return asdict(self)
