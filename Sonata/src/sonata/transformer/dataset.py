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
from sonata.transformer.families import PrimitiveFamilyMapping, derive_primitive_family_mapping
from sonata.utils.io import read_json, read_table

SCORE_HISTOGRAM_DIM = 12
GOAL_CONTEXT_SCALAR_COLUMNS = ("chord_size", "key_center", "start_state_norm")
HISTORY_CONTEXT_SCALAR_COLUMNS = (
    "duration_steps",
    "motion_energy",
    "chord_size",
    "key_center",
    "start_state_norm",
    "end_state_norm",
)
DEFAULT_CONTINUOUS_PARAM_COLUMNS = (
    "motion_energy",
    "chord_size",
    "start_state_norm",
    "end_state_norm",
)


def score_context_feature_names() -> list[str]:
    return [f"goal_histogram_{index:02d}" for index in range(SCORE_HISTOGRAM_DIM)] + [
        "goal_active_ratio",
        "goal_future_density",
    ]


def goal_context_feature_names() -> list[str]:
    return score_context_feature_names() + list(GOAL_CONTEXT_SCALAR_COLUMNS)


def history_context_feature_names() -> list[str]:
    return score_context_feature_names() + list(HISTORY_CONTEXT_SCALAR_COLUMNS)


def parse_score_context(raw: str) -> np.ndarray:
    payload = json.loads(raw)
    histogram = np.asarray(payload.get("goal_histogram", [0.0] * SCORE_HISTOGRAM_DIM), dtype=np.float32)
    scalars = np.asarray(
        [
            float(payload.get("active_ratio", 0.0)),
            float(payload.get("future_density", 0.0)),
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(np.concatenate([histogram, scalars], axis=0).astype(np.float32))


def build_goal_context(row: pd.Series) -> np.ndarray:
    score = parse_score_context(str(row["score_context_json"]))
    scalars = np.asarray([float(row.get(column, 0.0)) for column in GOAL_CONTEXT_SCALAR_COLUMNS], dtype=np.float32)
    return np.nan_to_num(np.concatenate([score, scalars], axis=0).astype(np.float32))


def build_history_context(row: pd.Series) -> np.ndarray:
    score = parse_score_context(str(row["score_context_json"]))
    scalars = np.asarray([float(row.get(column, 0.0)) for column in HISTORY_CONTEXT_SCALAR_COLUMNS], dtype=np.float32)
    return np.nan_to_num(np.concatenate([score, scalars], axis=0).astype(np.float32))


def build_planner_context(row: pd.Series) -> np.ndarray:
    return build_goal_context(row)


@dataclass
class PlannerMetadata:
    num_primitives: int
    num_duration_buckets: int
    num_dynamics_buckets: int
    num_families: int
    score_dim: int
    history_dim: int
    pad_primitive: int
    pad_duration: int
    pad_dynamics: int
    pad_family: int
    primitive_ids: list[str]
    primitive_family_names: list[str]
    primitive_to_family: list[int]
    family_mapping_mode: str
    continuous_param_names: list[str]
    continuous_param_mean: list[float]
    continuous_param_std: list[float]
    goal_context_features: list[str]
    history_context_features: list[str]

    @property
    def continuous_param_dim(self) -> int:
        return len(self.continuous_param_names)

    @property
    def primitive_family_labels(self) -> list[str]:
        return [self.primitive_family_names[index] for index in self.primitive_to_family]

    @property
    def family_to_primitives(self) -> list[list[int]]:
        grouped: list[list[int]] = [[] for _ in range(self.num_families)]
        for primitive_index, family_index in enumerate(self.primitive_to_family):
            grouped[int(family_index)].append(int(primitive_index))
        return grouped

    def to_payload(self) -> dict[str, Any]:
        return {
            "num_primitives": int(self.num_primitives),
            "num_duration_buckets": int(self.num_duration_buckets),
            "num_dynamics_buckets": int(self.num_dynamics_buckets),
            "num_families": int(self.num_families),
            "score_dim": int(self.score_dim),
            "history_dim": int(self.history_dim),
            "pad_primitive": int(self.pad_primitive),
            "pad_duration": int(self.pad_duration),
            "pad_dynamics": int(self.pad_dynamics),
            "pad_family": int(self.pad_family),
            "primitive_ids": list(self.primitive_ids),
            "primitive_family_names": list(self.primitive_family_names),
            "primitive_to_family": [int(value) for value in self.primitive_to_family],
            "family_mapping_mode": str(self.family_mapping_mode),
            "continuous_param_names": list(self.continuous_param_names),
            "continuous_param_mean": [float(value) for value in self.continuous_param_mean],
            "continuous_param_std": [float(value) for value in self.continuous_param_std],
            "goal_context_features": list(self.goal_context_features),
            "history_context_features": list(self.history_context_features),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "PlannerMetadata":
        return cls(
            num_primitives=int(payload["num_primitives"]),
            num_duration_buckets=int(payload["num_duration_buckets"]),
            num_dynamics_buckets=int(payload["num_dynamics_buckets"]),
            num_families=int(payload["num_families"]),
            score_dim=int(payload["score_dim"]),
            history_dim=int(payload.get("history_dim", payload["score_dim"])),
            pad_primitive=int(payload["pad_primitive"]),
            pad_duration=int(payload["pad_duration"]),
            pad_dynamics=int(payload["pad_dynamics"]),
            pad_family=int(payload.get("pad_family", payload["num_families"])),
            primitive_ids=[str(item) for item in payload.get("primitive_ids", [])],
            primitive_family_names=[str(item) for item in payload.get("primitive_family_names", [])],
            primitive_to_family=[int(item) for item in payload.get("primitive_to_family", [])],
            family_mapping_mode=str(payload.get("family_mapping_mode", "heuristic_stats")),
            continuous_param_names=[str(item) for item in payload.get("continuous_param_names", [])],
            continuous_param_mean=[float(item) for item in payload.get("continuous_param_mean", [])],
            continuous_param_std=[float(item) for item in payload.get("continuous_param_std", [])],
            goal_context_features=[str(item) for item in payload.get("goal_context_features", [])],
            history_context_features=[str(item) for item in payload.get("history_context_features", [])],
        )


@dataclass
class EpisodeSequence:
    primitive: np.ndarray
    family: np.ndarray
    duration: np.ndarray
    dynamics: np.ndarray
    history_context: np.ndarray
    goal_context: np.ndarray
    params: np.ndarray
    primitive_id: np.ndarray
    episode_id: str


class PrimitiveSequenceDataset(Dataset):
    def __init__(self, token_df: pd.DataFrame, metadata: PlannerMetadata, context_length: int, split: str):
        self.context_length = int(context_length)
        self.metadata = metadata
        self.token_df = token_df[token_df["split"] == split].copy()
        self.episodes: list[EpisodeSequence] = []
        self.samples: list[tuple[int, int]] = []
        self.target_primitive = np.zeros((0,), dtype=np.int64)
        self.target_family = np.zeros((0,), dtype=np.int64)
        self.target_duration = np.zeros((0,), dtype=np.int64)
        self.target_dynamics = np.zeros((0,), dtype=np.int64)
        if self.token_df.empty:
            return

        grouped = self.token_df.sort_values(["episode_id", "onset_step"]).groupby("episode_id", sort=True)
        target_primitive: list[int] = []
        target_family: list[int] = []
        target_duration: list[int] = []
        target_dynamics: list[int] = []
        for _, group in grouped:
            sequence = EpisodeSequence(
                primitive=group["primitive_index"].astype(int).to_numpy(dtype=np.int64),
                family=group["primitive_family_index"].astype(int).to_numpy(dtype=np.int64),
                duration=group["duration_bucket"].astype(int).to_numpy(dtype=np.int64),
                dynamics=group["dynamics_bucket"].astype(int).to_numpy(dtype=np.int64),
                history_context=np.stack([build_history_context(row) for _, row in group.iterrows()], axis=0).astype(np.float32),
                goal_context=np.stack([build_goal_context(row) for _, row in group.iterrows()], axis=0).astype(np.float32),
                params=np.stack(
                    [normalize_param_row(row, metadata) for _, row in group.iterrows()],
                    axis=0,
                ).astype(np.float32)
                if metadata.continuous_param_dim > 0
                else np.zeros((len(group), 0), dtype=np.float32),
                primitive_id=group["primitive_id"].astype(str).to_numpy(dtype=object),
                episode_id=str(group.iloc[0]["episode_id"]),
            )
            episode_index = len(self.episodes)
            self.episodes.append(sequence)
            for target_index in range(1, len(group)):
                self.samples.append((episode_index, target_index))
                target_primitive.append(int(sequence.primitive[target_index]))
                target_family.append(int(sequence.family[target_index]))
                target_duration.append(int(sequence.duration[target_index]))
                target_dynamics.append(int(sequence.dynamics[target_index]))
        self.target_primitive = np.asarray(target_primitive, dtype=np.int64)
        self.target_family = np.asarray(target_family, dtype=np.int64)
        self.target_duration = np.asarray(target_duration, dtype=np.int64)
        self.target_dynamics = np.asarray(target_dynamics, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode_index, target_index = self.samples[index]
        episode = self.episodes[episode_index]
        start = max(0, target_index - self.context_length)
        history_slice = slice(start, target_index)
        return {
            "primitive_history": episode.primitive[history_slice],
            "family_history": episode.family[history_slice],
            "duration_history": episode.duration[history_slice],
            "dynamics_history": episode.dynamics[history_slice],
            "history_context": episode.history_context[history_slice],
            "planner_context": episode.goal_context[target_index],
            "score_context": episode.goal_context[target_index],
            "target_primitive": int(episode.primitive[target_index]),
            "target_family": int(episode.family[target_index]),
            "target_duration": int(episode.duration[target_index]),
            "target_dynamics": int(episode.dynamics[target_index]),
            "target_params": episode.params[target_index],
            "target_primitive_id": str(episode.primitive_id[target_index]),
            "episode_id": episode.episode_id,
            "position": np.arange(target_index - start, dtype=np.int64),
        }

    def get_target_array(self, key: str) -> np.ndarray:
        if key == "primitive":
            return self.target_primitive
        if key == "family":
            return self.target_family
        if key == "duration":
            return self.target_duration
        if key == "dynamics":
            return self.target_dynamics
        raise KeyError(f"Unknown target array: {key}")


class TransformerActionDataset(Dataset):
    def __init__(
        self,
        token_df: pd.DataFrame,
        metadata: PlannerMetadata,
        primitive_root: Path,
        context_length: int,
        action_horizon: int,
        split: str,
    ):
        self.context_length = int(context_length)
        self.action_horizon = int(action_horizon)
        self.primitive_root = primitive_root
        self.metadata = metadata
        self.token_df = token_df[token_df["split"] == split].copy()
        self.manifest_lookup = build_manifest_lookup(load_stage1_source_manifest(primitive_root))
        self.samples: list[dict[str, Any]] = []
        self.target_primitive = np.zeros((0,), dtype=np.int64)
        self.target_family = np.zeros((0,), dtype=np.int64)
        if self.token_df.empty:
            return

        primitive_targets: list[int] = []
        family_targets: list[int] = []
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
                start = max(0, target_index - self.context_length)
                actions = slice_episode_array(episode.actions, int(row["onset_step"]), int(row["end_step"]))
                if actions is None:
                    continue
                action_target = resample_actions(actions, self.action_horizon)
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
                        "target_primitive_id": str(row["primitive_id"]),
                        "action_target": action_target,
                        "episode_id": str(row["episode_id"]),
                    }
                )
                primitive_targets.append(int(primitive[target_index]))
                family_targets.append(int(family[target_index]))
        self.target_primitive = np.asarray(primitive_targets, dtype=np.int64)
        self.target_family = np.asarray(family_targets, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]

    def get_target_array(self, key: str) -> np.ndarray:
        if key == "primitive":
            return self.target_primitive
        if key == "family":
            return self.target_family
        raise KeyError(f"Unknown target array: {key}")


def normalize_param_row(row: pd.Series, metadata: PlannerMetadata) -> np.ndarray:
    if metadata.continuous_param_dim == 0:
        return np.zeros((0,), dtype=np.float32)
    values = np.asarray([float(row.get(name, 0.0)) for name in metadata.continuous_param_names], dtype=np.float32)
    mean = np.asarray(metadata.continuous_param_mean, dtype=np.float32)
    std = np.asarray(metadata.continuous_param_std, dtype=np.float32)
    normalized = (values - mean) / np.clip(std, 1e-6, None)
    return np.nan_to_num(normalized.astype(np.float32))


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
    family = np.full((batch_size, max_length), metadata.pad_family, dtype=np.int64)
    duration = np.full((batch_size, max_length), metadata.pad_duration, dtype=np.int64)
    dynamics = np.full((batch_size, max_length), metadata.pad_dynamics, dtype=np.int64)
    history_context = np.zeros((batch_size, max_length, metadata.history_dim), dtype=np.float32)
    attention_mask = np.zeros((batch_size, max_length), dtype=np.float32)
    for index, item in enumerate(batch):
        length = len(item["primitive_history"])
        primitive[index, :length] = item["primitive_history"]
        family[index, :length] = item["family_history"]
        duration[index, :length] = item["duration_history"]
        dynamics[index, :length] = item["dynamics_history"]
        history_context[index, :length] = item["history_context"]
        attention_mask[index, :length] = 1.0
    planner_context = torch.from_numpy(
        np.stack([np.asarray(item["planner_context"], dtype=np.float32) for item in batch], axis=0).astype(np.float32)
    )
    target_params = (
        torch.from_numpy(np.stack([item["target_params"] for item in batch], axis=0).astype(np.float32))
        if metadata.continuous_param_dim > 0
        else torch.zeros((batch_size, 0), dtype=torch.float32)
    )
    collated = {
        "primitive_history": torch.from_numpy(primitive),
        "family_history": torch.from_numpy(family),
        "duration_history": torch.from_numpy(duration),
        "dynamics_history": torch.from_numpy(dynamics),
        "history_context": torch.from_numpy(history_context),
        "planner_context": planner_context,
        "score_history": torch.from_numpy(history_context),
        "score_context": planner_context,
        "attention_mask": torch.from_numpy(attention_mask),
        "target_primitive": torch.tensor([item["target_primitive"] for item in batch], dtype=torch.long),
        "target_family": torch.tensor([item["target_family"] for item in batch], dtype=torch.long),
        "target_duration": torch.tensor([item["target_duration"] for item in batch], dtype=torch.long),
        "target_dynamics": torch.tensor([item["target_dynamics"] for item in batch], dtype=torch.long),
        "target_params": target_params,
    }
    return collated


def action_collate_fn(batch: list[dict[str, Any]], metadata: PlannerMetadata) -> dict[str, torch.Tensor]:
    planner_batch = planner_collate_fn(batch, metadata=metadata)
    planner_batch["action_target"] = torch.from_numpy(np.stack([item["action_target"] for item in batch], axis=0).astype(np.float32))
    return planner_batch


def load_transformer_inputs(
    primitive_root: Path,
    *,
    family_mapping_mode: str = "heuristic_stats",
    continuous_param_names: list[str] | tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, PlannerMetadata]:
    token_df = read_table(primitive_root / "tokens" / "primitive_tokens")
    vocabulary = read_json(primitive_root / "tokens" / "primitive_vocabulary.json")
    library_path = primitive_root / "library" / "primitive_library"
    library_df = read_table(library_path) if library_path.with_suffix(".csv").exists() or library_path.with_suffix(".parquet").exists() else None
    score_dim = build_goal_context(token_df.iloc[0]).shape[0] if not token_df.empty else len(goal_context_feature_names())
    history_dim = build_history_context(token_df.iloc[0]).shape[0] if not token_df.empty else len(history_context_feature_names())
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
    primitive_ids = [str(item) for item in vocabulary.get("primitive_ids", [])]
    if not primitive_ids:
        primitive_ids = sorted(token_df["primitive_id"].astype(str).unique().tolist())

    family_mapping = derive_primitive_family_mapping(
        token_df,
        primitive_ids,
        library_df=library_df,
        mode=family_mapping_mode,
    )
    token_df = token_df.copy()
    token_df["primitive_family"] = token_df["primitive_id"].astype(str).map(family_mapping.primitive_to_family_name)
    token_df["primitive_family_index"] = token_df["primitive_id"].astype(str).map(family_mapping.primitive_to_family_index).astype(int)

    requested_param_names = list(continuous_param_names) if continuous_param_names is not None else list(DEFAULT_CONTINUOUS_PARAM_COLUMNS)
    resolved_param_names = [name for name in requested_param_names if name in token_df.columns]
    train_df = token_df[token_df["split"] == "train"].copy() if "split" in token_df.columns else token_df.copy()
    if train_df.empty:
        train_df = token_df
    param_mean: list[float] = []
    param_std: list[float] = []
    for name in resolved_param_names:
        values = train_df[name].to_numpy(dtype=np.float32) if name in train_df.columns else np.zeros((0,), dtype=np.float32)
        if values.size == 0:
            param_mean.append(0.0)
            param_std.append(1.0)
            continue
        mean = float(np.nanmean(values))
        std = float(np.nanstd(values))
        param_mean.append(mean)
        param_std.append(std if std > 1e-6 else 1.0)

    metadata = PlannerMetadata(
        num_primitives=int(vocabulary["num_primitives"]),
        num_duration_buckets=num_duration_buckets,
        num_dynamics_buckets=num_dynamics_buckets,
        num_families=len(family_mapping.family_names),
        score_dim=score_dim,
        history_dim=history_dim,
        pad_primitive=int(vocabulary["num_primitives"]),
        pad_duration=num_duration_buckets,
        pad_dynamics=num_dynamics_buckets,
        pad_family=len(family_mapping.family_names),
        primitive_ids=primitive_ids,
        primitive_family_names=list(family_mapping.family_names),
        primitive_to_family=[int(family_mapping.primitive_to_family_index[primitive_id]) for primitive_id in primitive_ids],
        family_mapping_mode=str(family_mapping_mode),
        continuous_param_names=resolved_param_names,
        continuous_param_mean=param_mean,
        continuous_param_std=param_std,
        goal_context_features=goal_context_feature_names(),
        history_context_features=history_context_feature_names(),
    )
    return token_df, metadata


def family_mapping_records(metadata: PlannerMetadata) -> list[dict[str, Any]]:
    return [
        {
            "primitive_id": primitive_id,
            "primitive_index": primitive_index,
            "primitive_family": metadata.primitive_family_names[metadata.primitive_to_family[primitive_index]],
            "primitive_family_index": int(metadata.primitive_to_family[primitive_index]),
        }
        for primitive_index, primitive_id in enumerate(metadata.primitive_ids)
    ]


def family_mask_tensor(metadata: PlannerMetadata, device: torch.device | None = None) -> torch.Tensor:
    mask = torch.zeros((metadata.num_families, metadata.num_primitives), dtype=torch.bool, device=device)
    for primitive_index, family_index in enumerate(metadata.primitive_to_family):
        mask[int(family_index), int(primitive_index)] = True
    return mask
