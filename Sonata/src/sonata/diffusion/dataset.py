from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from sonata.primitives.segmenters import load_segment_array
from sonata.transformer.dataset import build_planner_context, load_transformer_inputs, planner_collate_fn


@dataclass
class DiffusionMetadata:
    action_dim: int
    state_dim: int
    score_dim: int
    num_primitives: int
    num_duration_buckets: int
    num_dynamics_buckets: int
    pad_primitive: int
    pad_duration: int
    pad_dynamics: int


class DiffusionChunkDataset(Dataset):
    def __init__(
        self,
        token_df: pd.DataFrame,
        primitive_root: Path,
        split: str,
        context_length: int,
        action_horizon: int,
        state_context_steps: int,
    ) -> None:
        self.token_df = token_df[token_df["split"] == split].copy()
        self.primitive_root = primitive_root
        self.context_length = int(context_length)
        self.action_horizon = int(action_horizon)
        self.state_context_steps = int(state_context_steps)
        self.samples: list[dict[str, Any]] = []
        grouped = self.token_df.sort_values(["episode_id", "onset_step"]).groupby("episode_id", sort=True)
        for _, group in grouped:
            primitive = group["primitive_index"].astype(int).to_numpy()
            duration = group["duration_bucket"].astype(int).to_numpy()
            dynamics = group["dynamics_bucket"].astype(int).to_numpy()
            score_context = np.stack([build_planner_context(row) for _, row in group.iterrows()], axis=0).astype(np.float32)
            for target_index in range(1, len(group)):
                row = group.iloc[target_index]
                bundle = np.load(self.primitive_root / "segments" / str(row["chunk_path"]), allow_pickle=True)
                hand_joints = load_segment_array(bundle, "hand_joints", int(row["chunk_index"]))
                actions = load_segment_array(bundle, "actions", int(row["chunk_index"]))
                if hand_joints is None or actions is None:
                    continue
                state_context = resample_sequence(hand_joints, self.state_context_steps).reshape(-1)
                action_target = resample_sequence(actions, self.action_horizon)
                start = max(0, target_index - self.context_length)
                self.samples.append(
                    {
                        "primitive_history": primitive[start:target_index],
                        "duration_history": duration[start:target_index],
                        "dynamics_history": dynamics[start:target_index],
                        "score_history": score_context[start:target_index],
                        "primitive_index": int(row["primitive_index"]),
                        "duration_bucket": int(row["duration_bucket"]),
                        "dynamics_bucket": int(row["dynamics_bucket"]),
                        "score_context": build_planner_context(row),
                        "state_context": state_context.astype(np.float32),
                        "action_target": action_target.astype(np.float32),
                        "primitive_id": str(row["primitive_id"]),
                        "episode_id": str(row["episode_id"]),
                        "onset_step": int(row["onset_step"]),
                        "end_step": int(row["end_step"]),
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]


def resample_sequence(array: np.ndarray, steps: int) -> np.ndarray:
    if array.shape[0] == steps:
        return array.astype(np.float32)
    x_old = np.linspace(0.0, 1.0, array.shape[0], dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, steps, dtype=np.float32)
    output = np.zeros((steps, array.shape[1]), dtype=np.float32)
    for dim in range(array.shape[1]):
        output[:, dim] = np.interp(x_new, x_old, array[:, dim])
    return output


def build_prior_lookup(primitive_root: Path, action_horizon: int) -> dict[str, np.ndarray]:
    library_df = pd.read_parquet(primitive_root / "library" / "primitive_library.parquet") if (primitive_root / "library" / "primitive_library.parquet").exists() else pd.read_csv(primitive_root / "library" / "primitive_library.csv")
    lookup: dict[str, np.ndarray] = {}
    for row in library_df.itertuples(index=False):
        payload = np.load(Path(row.prior_path), allow_pickle=True)
        prior = np.asarray(payload["prior_mean"], dtype=np.float32)
        lookup[str(row.primitive_id)] = resample_sequence(prior, action_horizon)
    return lookup


def diffusion_collate_fn(batch: list[dict[str, Any]], metadata: DiffusionMetadata, prior_lookup: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    planner_batch = planner_collate_fn(
        [
            item
            | {
                "target_primitive": item["primitive_index"],
                "target_duration": item["duration_bucket"],
                "target_dynamics": item["dynamics_bucket"],
            }
            for item in batch
        ],
        metadata=metadata_to_planner(metadata),
    )
    planner_batch["primitive_index"] = torch.tensor([item["primitive_index"] for item in batch], dtype=torch.long)
    planner_batch["duration_bucket"] = torch.tensor([item["duration_bucket"] for item in batch], dtype=torch.long)
    planner_batch["dynamics_bucket"] = torch.tensor([item["dynamics_bucket"] for item in batch], dtype=torch.long)
    planner_batch["score_context"] = torch.from_numpy(np.stack([item["score_context"] for item in batch], axis=0).astype(np.float32))
    planner_batch["state_context"] = torch.from_numpy(np.stack([item["state_context"] for item in batch], axis=0).astype(np.float32))
    planner_batch["action_target"] = torch.from_numpy(np.stack([item["action_target"] for item in batch], axis=0).astype(np.float32))
    planner_batch["gmr_prior"] = torch.from_numpy(np.stack([prior_lookup[item["primitive_id"]] for item in batch], axis=0).astype(np.float32))
    planner_batch["episode_id"] = [item["episode_id"] for item in batch]
    planner_batch["onset_step"] = [int(item["onset_step"]) for item in batch]
    planner_batch["end_step"] = [int(item["end_step"]) for item in batch]
    return planner_batch


def metadata_to_planner(metadata: DiffusionMetadata):
    from sonata.transformer.dataset import PlannerMetadata

    return PlannerMetadata(
        num_primitives=metadata.num_primitives,
        num_duration_buckets=metadata.num_duration_buckets,
        num_dynamics_buckets=metadata.num_dynamics_buckets,
        score_dim=metadata.score_dim,
        pad_primitive=metadata.pad_primitive,
        pad_duration=metadata.pad_duration,
        pad_dynamics=metadata.pad_dynamics,
    )


def load_diffusion_inputs(primitive_root: Path, action_horizon: int, state_context_steps: int) -> tuple[pd.DataFrame, DiffusionMetadata, dict[str, np.ndarray]]:
    token_df, planner_metadata = load_transformer_inputs(primitive_root)
    sample_bundle = np.load(primitive_root / "segments" / str(token_df.iloc[0]["chunk_path"]), allow_pickle=True)
    sample_actions = load_segment_array(sample_bundle, "actions", int(token_df.iloc[0]["chunk_index"]))
    sample_hand = load_segment_array(sample_bundle, "hand_joints", int(token_df.iloc[0]["chunk_index"]))
    action_dim = int(sample_actions.shape[1]) if sample_actions is not None else 39
    state_dim = int(state_context_steps * sample_hand.shape[1]) if sample_hand is not None else int(state_context_steps * 46)
    metadata = DiffusionMetadata(
        action_dim=action_dim,
        state_dim=state_dim,
        score_dim=planner_metadata.score_dim,
        num_primitives=planner_metadata.num_primitives,
        num_duration_buckets=planner_metadata.num_duration_buckets,
        num_dynamics_buckets=planner_metadata.num_dynamics_buckets,
        pad_primitive=planner_metadata.pad_primitive,
        pad_duration=planner_metadata.pad_duration,
        pad_dynamics=planner_metadata.pad_dynamics,
    )
    prior_lookup = build_prior_lookup(primitive_root, action_horizon)
    return token_df, metadata, prior_lookup
