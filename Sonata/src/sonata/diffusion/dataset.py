from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from sonata.data.loading import build_manifest_lookup, load_episode_record, load_stage1_source_manifest
from sonata.transformer.dataset import (
    PlannerMetadata,
    build_goal_context,
    build_history_context,
    load_transformer_inputs,
    normalize_param_row,
    planner_collate_fn,
)


@dataclass
class DiffusionMetadata(PlannerMetadata):
    action_dim: int
    state_dim: int


class DiffusionChunkDataset(Dataset):
    def __init__(
        self,
        token_df: pd.DataFrame,
        metadata: DiffusionMetadata,
        primitive_root: Path,
        split: str,
        context_length: int,
        action_horizon: int,
        state_context_steps: int,
    ) -> None:
        self.token_df = token_df[token_df["split"] == split].copy()
        self.metadata = metadata
        self.primitive_root = primitive_root
        self.context_length = int(context_length)
        self.action_horizon = int(action_horizon)
        self.state_context_steps = int(state_context_steps)
        self.manifest_lookup = build_manifest_lookup(load_stage1_source_manifest(primitive_root))
        self.samples: list[dict[str, Any]] = []
        grouped = self.token_df.sort_values(["episode_id", "onset_step"]).groupby("episode_id", sort=True)
        for _, group in grouped:
            group_key = (str(group.iloc[0]["song_id"]), str(group.iloc[0]["episode_id"]))
            episode = load_episode_record(self.manifest_lookup[group_key])
            primitive = group["primitive_index"].astype(int).to_numpy(dtype=np.int64)
            family = group["primitive_family_index"].astype(int).to_numpy(dtype=np.int64)
            duration = group["duration_bucket"].astype(int).to_numpy(dtype=np.int64)
            dynamics = group["dynamics_bucket"].astype(int).to_numpy(dtype=np.int64)
            history_context = np.stack([build_history_context(row) for _, row in group.iterrows()], axis=0).astype(np.float32)
            goal_context = np.stack([build_goal_context(row) for _, row in group.iterrows()], axis=0).astype(np.float32)
            params = (
                np.stack([normalize_param_row(row, metadata) for _, row in group.iterrows()], axis=0).astype(np.float32)
                if metadata.continuous_param_dim > 0
                else np.zeros((len(group), 0), dtype=np.float32)
            )
            for target_index in range(1, len(group)):
                row = group.iloc[target_index]
                hand_joints = slice_episode_array(episode.hand_joints, int(row["onset_step"]), int(row["end_step"]))
                actions = slice_episode_array(episode.actions, int(row["onset_step"]), int(row["end_step"]))
                if hand_joints is None or actions is None:
                    continue
                state_context = resample_sequence(hand_joints, self.state_context_steps).reshape(-1)
                action_target = resample_sequence(actions, self.action_horizon)
                start = max(0, target_index - self.context_length)
                self.samples.append(
                    {
                        "primitive_history": primitive[start:target_index],
                        "family_history": family[start:target_index],
                        "duration_history": duration[start:target_index],
                        "dynamics_history": dynamics[start:target_index],
                        "history_context": history_context[start:target_index],
                        "planner_context": goal_context[target_index],
                        "score_context": goal_context[target_index],
                        "target_primitive": int(primitive[target_index]),
                        "target_family": int(family[target_index]),
                        "target_duration": int(duration[target_index]),
                        "target_dynamics": int(dynamics[target_index]),
                        "target_params": params[target_index],
                        "primitive_index": int(row["primitive_index"]),
                        "duration_bucket": int(row["duration_bucket"]),
                        "dynamics_bucket": int(row["dynamics_bucket"]),
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


def slice_episode_array(array: np.ndarray | None, start: int, end: int) -> np.ndarray | None:
    if array is None:
        return None
    return np.asarray(array[start:end], dtype=np.float32)


def build_prior_lookup(primitive_root: Path, action_horizon: int) -> dict[str, np.ndarray]:
    library_df = pd.read_parquet(primitive_root / "library" / "primitive_library.parquet") if (primitive_root / "library" / "primitive_library.parquet").exists() else pd.read_csv(primitive_root / "library" / "primitive_library.csv")
    lookup: dict[str, np.ndarray] = {}
    for row in library_df.itertuples(index=False):
        payload = np.load(Path(row.prior_path), allow_pickle=True)
        prior = np.asarray(payload["prior_mean"], dtype=np.float32)
        lookup[str(row.primitive_id)] = resample_sequence(prior, action_horizon)
    return lookup


def diffusion_collate_fn(batch: list[dict[str, Any]], metadata: DiffusionMetadata, prior_lookup: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    planner_batch = planner_collate_fn(batch, metadata=metadata)
    planner_batch["primitive_index"] = torch.tensor([item["primitive_index"] for item in batch], dtype=torch.long)
    planner_batch["duration_bucket"] = torch.tensor([item["duration_bucket"] for item in batch], dtype=torch.long)
    planner_batch["dynamics_bucket"] = torch.tensor([item["dynamics_bucket"] for item in batch], dtype=torch.long)
    planner_batch["state_context"] = torch.from_numpy(np.stack([item["state_context"] for item in batch], axis=0).astype(np.float32))
    planner_batch["action_target"] = torch.from_numpy(np.stack([item["action_target"] for item in batch], axis=0).astype(np.float32))
    planner_batch["gmr_prior"] = torch.from_numpy(np.stack([prior_lookup[item["primitive_id"]] for item in batch], axis=0).astype(np.float32))
    planner_batch["episode_id"] = [item["episode_id"] for item in batch]
    planner_batch["onset_step"] = [int(item["onset_step"]) for item in batch]
    planner_batch["end_step"] = [int(item["end_step"]) for item in batch]
    return planner_batch


def metadata_to_planner(metadata: DiffusionMetadata) -> PlannerMetadata:
    return PlannerMetadata(
        num_primitives=metadata.num_primitives,
        num_duration_buckets=metadata.num_duration_buckets,
        num_dynamics_buckets=metadata.num_dynamics_buckets,
        num_families=metadata.num_families,
        score_dim=metadata.score_dim,
        history_dim=metadata.history_dim,
        pad_primitive=metadata.pad_primitive,
        pad_duration=metadata.pad_duration,
        pad_dynamics=metadata.pad_dynamics,
        pad_family=metadata.pad_family,
        primitive_ids=list(metadata.primitive_ids),
        primitive_family_names=list(metadata.primitive_family_names),
        primitive_to_family=list(metadata.primitive_to_family),
        family_mapping_mode=metadata.family_mapping_mode,
        continuous_param_names=list(metadata.continuous_param_names),
        continuous_param_mean=list(metadata.continuous_param_mean),
        continuous_param_std=list(metadata.continuous_param_std),
        goal_context_features=list(metadata.goal_context_features),
        history_context_features=list(metadata.history_context_features),
    )


def load_diffusion_inputs(
    primitive_root: Path,
    action_horizon: int,
    state_context_steps: int,
    *,
    family_mapping_mode: str = "heuristic_stats",
    continuous_param_names: list[str] | tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, DiffusionMetadata, dict[str, np.ndarray]]:
    token_df, planner_metadata = load_transformer_inputs(
        primitive_root,
        family_mapping_mode=family_mapping_mode,
        continuous_param_names=continuous_param_names,
    )
    source_manifest = load_stage1_source_manifest(primitive_root)
    action_dim = int(source_manifest["action_dim"].replace(0, np.nan).dropna().iloc[0]) if "action_dim" in source_manifest.columns and not source_manifest["action_dim"].replace(0, np.nan).dropna().empty else 39
    hand_dim = int(source_manifest["hand_joint_dim"].replace(0, np.nan).dropna().iloc[0]) if "hand_joint_dim" in source_manifest.columns and not source_manifest["hand_joint_dim"].replace(0, np.nan).dropna().empty else 46
    state_dim = int(state_context_steps * hand_dim)
    metadata = DiffusionMetadata(
        num_primitives=planner_metadata.num_primitives,
        num_duration_buckets=planner_metadata.num_duration_buckets,
        num_dynamics_buckets=planner_metadata.num_dynamics_buckets,
        num_families=planner_metadata.num_families,
        score_dim=planner_metadata.score_dim,
        history_dim=planner_metadata.history_dim,
        pad_primitive=planner_metadata.pad_primitive,
        pad_duration=planner_metadata.pad_duration,
        pad_dynamics=planner_metadata.pad_dynamics,
        pad_family=planner_metadata.pad_family,
        primitive_ids=list(planner_metadata.primitive_ids),
        primitive_family_names=list(planner_metadata.primitive_family_names),
        primitive_to_family=list(planner_metadata.primitive_to_family),
        family_mapping_mode=planner_metadata.family_mapping_mode,
        continuous_param_names=list(planner_metadata.continuous_param_names),
        continuous_param_mean=list(planner_metadata.continuous_param_mean),
        continuous_param_std=list(planner_metadata.continuous_param_std),
        goal_context_features=list(planner_metadata.goal_context_features),
        history_context_features=list(planner_metadata.history_context_features),
        action_dim=action_dim,
        state_dim=state_dim,
    )
    prior_lookup = build_prior_lookup(primitive_root, action_horizon)
    return token_df, metadata, prior_lookup
