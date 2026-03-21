from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from sonata.data.loading import build_manifest_lookup, load_episode_record, load_stage1_source_manifest
from sonata.utils.io import read_json, read_table


def parse_score_context(raw: str) -> np.ndarray:
    payload = json.loads(raw)
    histogram = np.asarray(payload.get("goal_histogram", [0.0] * 12), dtype=np.float32)
    scalars = np.asarray(
        [
            float(payload.get("active_ratio", 0.0)),
            float(payload.get("future_density", 0.0)),
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(np.concatenate([histogram, scalars], axis=0).astype(np.float32))


def build_planner_context(row: pd.Series) -> np.ndarray:
    score = parse_score_context(str(row["score_context_json"]))
    scalars = np.asarray(
        [
            float(row["duration_steps"]),
            float(row["motion_energy"]),
            float(row["chord_size"]),
            float(row["key_center"]),
            float(row["start_state_norm"]),
            float(row["end_state_norm"]),
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(np.concatenate([score, scalars], axis=0).astype(np.float32))


@dataclass
class PlannerMetadata:
    num_primitives: int
    num_duration_buckets: int
    num_dynamics_buckets: int
    score_dim: int
    pad_primitive: int
    pad_duration: int
    pad_dynamics: int


class PrimitiveSequenceDataset(Dataset):
    def __init__(self, token_df: pd.DataFrame, context_length: int, split: str):
        self.context_length = int(context_length)
        self.token_df = token_df[token_df["split"] == split].copy()
        self.score_dim = build_planner_context(self.token_df.iloc[0]).shape[0] if not self.token_df.empty else 20
        self.samples: list[dict[str, Any]] = []
        grouped = self.token_df.sort_values(["episode_id", "onset_step"]).groupby("episode_id", sort=True)
        for _, group in grouped:
            primitive = group["primitive_index"].astype(int).to_numpy()
            duration = group["duration_bucket"].astype(int).to_numpy()
            dynamics = group["dynamics_bucket"].astype(int).to_numpy()
            score_context = np.stack([build_planner_context(row) for _, row in group.iterrows()], axis=0).astype(np.float32)
            for target_index in range(1, len(group)):
                start = max(0, target_index - self.context_length)
                history_slice = slice(start, target_index)
                self.samples.append(
                    {
                        "primitive_history": primitive[history_slice],
                        "duration_history": duration[history_slice],
                        "dynamics_history": dynamics[history_slice],
                        "score_history": score_context[history_slice],
                        "target_primitive": primitive[target_index],
                        "target_duration": duration[target_index],
                        "target_dynamics": dynamics[target_index],
                        "episode_id": group.iloc[target_index]["episode_id"],
                        "position": np.arange(target_index - start, dtype=np.int64),
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]


class TransformerActionDataset(Dataset):
    def __init__(self, token_df: pd.DataFrame, primitive_root: Path, context_length: int, action_horizon: int, split: str):
        self.context_length = int(context_length)
        self.action_horizon = int(action_horizon)
        self.primitive_root = primitive_root
        self.token_df = token_df[token_df["split"] == split].copy()
        self.manifest_lookup = build_manifest_lookup(load_stage1_source_manifest(primitive_root))
        self.samples: list[dict[str, Any]] = []
        grouped = self.token_df.sort_values(["episode_id", "onset_step"]).groupby("episode_id", sort=True)
        for _, group in grouped:
            group_key = (str(group.iloc[0]["song_id"]), str(group.iloc[0]["episode_id"]))
            episode = load_episode_record(self.manifest_lookup[group_key])
            primitive = group["primitive_index"].astype(int).to_numpy()
            duration = group["duration_bucket"].astype(int).to_numpy()
            dynamics = group["dynamics_bucket"].astype(int).to_numpy()
            score_context = np.stack([build_planner_context(row) for _, row in group.iterrows()], axis=0).astype(np.float32)
            for target_index in range(1, len(group)):
                row = group.iloc[target_index]
                start = max(0, target_index - self.context_length)
                actions = slice_episode_array(episode.actions, int(row["onset_step"]), int(row["end_step"]))
                if actions is None:
                    continue
                action_target = resample_actions(actions, self.action_horizon)
                self.samples.append(
                    {
                        "primitive_history": primitive[start:target_index],
                        "duration_history": duration[start:target_index],
                        "dynamics_history": dynamics[start:target_index],
                        "score_history": score_context[start:target_index],
                        "action_target": action_target,
                        "episode_id": row["episode_id"],
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]


def resample_actions(actions: np.ndarray, horizon: int) -> np.ndarray:
    x_old = np.linspace(0.0, 1.0, actions.shape[0], dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, horizon, dtype=np.float32)
    output = np.zeros((horizon, actions.shape[1]), dtype=np.float32)
    for dim in range(actions.shape[1]):
        output[:, dim] = np.interp(x_new, x_old, actions[:, dim])
    return output


def slice_episode_array(array: np.ndarray | None, start: int, end: int) -> np.ndarray | None:
    if array is None:
        return None
    return np.asarray(array[start:end], dtype=np.float32)


def planner_collate_fn(batch: list[dict[str, Any]], metadata: PlannerMetadata) -> dict[str, torch.Tensor]:
    batch_size = len(batch)
    max_length = max(len(item["primitive_history"]) for item in batch)
    primitive = np.full((batch_size, max_length), metadata.pad_primitive, dtype=np.int64)
    duration = np.full((batch_size, max_length), metadata.pad_duration, dtype=np.int64)
    dynamics = np.full((batch_size, max_length), metadata.pad_dynamics, dtype=np.int64)
    score = np.zeros((batch_size, max_length, metadata.score_dim), dtype=np.float32)
    attention_mask = np.zeros((batch_size, max_length), dtype=np.float32)
    for index, item in enumerate(batch):
        length = len(item["primitive_history"])
        # Right padding keeps valid tokens in a causal prefix and avoids NaNs
        # from mixed-length batches inside the planner transformer.
        primitive[index, :length] = item["primitive_history"]
        duration[index, :length] = item["duration_history"]
        dynamics[index, :length] = item["dynamics_history"]
        score[index, :length] = item["score_history"]
        attention_mask[index, :length] = 1.0
    return {
        "primitive_history": torch.from_numpy(primitive),
        "duration_history": torch.from_numpy(duration),
        "dynamics_history": torch.from_numpy(dynamics),
        "score_history": torch.from_numpy(score),
        "attention_mask": torch.from_numpy(attention_mask),
        "target_primitive": torch.tensor([item["target_primitive"] for item in batch], dtype=torch.long),
        "target_duration": torch.tensor([item["target_duration"] for item in batch], dtype=torch.long),
        "target_dynamics": torch.tensor([item["target_dynamics"] for item in batch], dtype=torch.long),
    }


def action_collate_fn(batch: list[dict[str, Any]], metadata: PlannerMetadata) -> dict[str, torch.Tensor]:
    planner_batch = planner_collate_fn(
        [
            item
            | {
                "target_primitive": 0,
                "target_duration": 0,
                "target_dynamics": 0,
            }
            for item in batch
        ],
        metadata=metadata,
    )
    planner_batch["action_target"] = torch.from_numpy(np.stack([item["action_target"] for item in batch], axis=0).astype(np.float32))
    return planner_batch


def load_transformer_inputs(primitive_root: Path) -> tuple[pd.DataFrame, PlannerMetadata]:
    token_df = read_table(primitive_root / "tokens" / "primitive_tokens")
    vocabulary = read_json(primitive_root / "tokens" / "primitive_vocabulary.json")
    score_dim = build_planner_context(token_df.iloc[0]).shape[0] if not token_df.empty else 20
    duration_buckets = [int(value) for value in vocabulary.get("duration_buckets", [])]
    dynamics_buckets = [int(value) for value in vocabulary.get("dynamics_buckets", [])]
    num_duration_buckets = int(
        vocabulary.get(
            "num_duration_buckets",
            (max(duration_buckets) + 1) if duration_buckets else 1,
        )
    )
    num_dynamics_buckets = int(
        vocabulary.get(
            "num_dynamics_buckets",
            (max(dynamics_buckets) + 1) if dynamics_buckets else 1,
        )
    )
    metadata = PlannerMetadata(
        num_primitives=int(vocabulary["num_primitives"]),
        num_duration_buckets=num_duration_buckets,
        num_dynamics_buckets=num_dynamics_buckets,
        score_dim=score_dim,
        pad_primitive=int(vocabulary["num_primitives"]),
        pad_duration=num_duration_buckets,
        pad_dynamics=num_dynamics_buckets,
    )
    return token_df, metadata
