from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset


_GOAL_STEP_DIM = 89
_FINGERING_DIM = 10
_PIANO_STATE_DIM = 88
_SUSTAIN_DIM = 1
_HAND_JOINT_DIM = 26
_RP1M_HAND_JOINT_DIM = 23
_RP1M_ACTION_DIM = 39
_DEFAULT_ACTION_DIM = 45
_LOG_STD_MIN = -5.0
_LOG_STD_MAX = 2.0


def infer_obs_dim(n_steps_lookahead: int) -> int:
    goal_dim = (int(n_steps_lookahead) + 1) * _GOAL_STEP_DIM
    return goal_dim + _FINGERING_DIM + _PIANO_STATE_DIM + _SUSTAIN_DIM + (2 * _HAND_JOINT_DIM)


def infer_offsets(n_steps_lookahead: int) -> dict[str, int]:
    goal_dim = (int(n_steps_lookahead) + 1) * _GOAL_STEP_DIM
    finger_offset = goal_dim
    piano_offset = finger_offset + _FINGERING_DIM
    sustain_offset = piano_offset + _PIANO_STATE_DIM
    rh_offset = sustain_offset + _SUSTAIN_DIM
    lh_offset = rh_offset + _HAND_JOINT_DIM
    return {
        "goal": 0,
        "fingering": finger_offset,
        "piano": piano_offset,
        "sustain": sustain_offset,
        "rh": rh_offset,
        "lh": lh_offset,
    }


def build_bc_observation(
    *,
    goals_trajectory: np.ndarray,
    piano_state_t: np.ndarray,
    hand_joints_t: np.ndarray,
    timestep: int,
    n_steps_lookahead: int,
) -> np.ndarray:
    obs_dim = infer_obs_dim(n_steps_lookahead)
    offsets = infer_offsets(n_steps_lookahead)
    observation = np.zeros(obs_dim, dtype=np.float32)
    max_t = goals_trajectory.shape[0]

    for lookahead in range(int(n_steps_lookahead) + 1):
        if timestep + lookahead >= max_t:
            break
        start = offsets["goal"] + (lookahead * _GOAL_STEP_DIM)
        end = start + _GOAL_STEP_DIM
        observation[start:end] = goals_trajectory[timestep + lookahead].astype(np.float32)

    observation[offsets["piano"] : offsets["piano"] + _PIANO_STATE_DIM] = piano_state_t[:_PIANO_STATE_DIM]
    observation[offsets["sustain"]] = float(piano_state_t[_PIANO_STATE_DIM])
    observation[offsets["rh"] : offsets["rh"] + _RP1M_HAND_JOINT_DIM] = hand_joints_t[:_RP1M_HAND_JOINT_DIM]
    observation[offsets["lh"] : offsets["lh"] + _RP1M_HAND_JOINT_DIM] = hand_joints_t[
        _RP1M_HAND_JOINT_DIM : (2 * _RP1M_HAND_JOINT_DIM)
    ]
    return observation


def build_bc_action(rp1m_action: np.ndarray, *, action_dim: int = _DEFAULT_ACTION_DIM) -> np.ndarray:
    action = np.zeros(int(action_dim), dtype=np.float32)
    copy_dim = min(int(action_dim), _RP1M_ACTION_DIM, int(rp1m_action.shape[-1]))
    action[:copy_dim] = rp1m_action[:copy_dim]
    return action


class RP1MIterableDataset(IterableDataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        zarr_root: str | os.PathLike[str],
        *,
        n_steps_lookahead: int,
        action_dim: int = _DEFAULT_ACTION_DIM,
        max_songs: int | None = None,
        max_traj_per_song: int | None = None,
    ) -> None:
        self.zarr_root = str(zarr_root)
        self.n_steps_lookahead = int(n_steps_lookahead)
        self.action_dim = int(action_dim)
        self.max_songs = max_songs
        self.max_traj_per_song = max_traj_per_song

        try:
            import zarr
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise ImportError("Install zarr to use RP1MIterableDataset.") from exc

        root = zarr.open(self.zarr_root, mode="r")
        songs = list(root.keys())
        if self.max_songs is not None:
            songs = songs[: self.max_songs]
        self.song_names = songs

        total = 0
        for song_name in self.song_names:
            group = root[song_name]
            num_trajectories, horizon, _ = group["actions"].shape
            if self.max_traj_per_song is not None:
                num_trajectories = min(num_trajectories, self.max_traj_per_song)
            total += max(horizon - 1, 0) * num_trajectories
        self.total_examples = int(total)

    def __len__(self) -> int:
        return self.total_examples

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        import zarr

        root = zarr.open(self.zarr_root, mode="r")
        rng = np.random.default_rng()
        song_order = rng.permutation(len(self.song_names))
        for song_index in song_order:
            group = root[self.song_names[song_index]]
            actions = group["actions"][:]
            goals = group["goals"][:]
            piano_states = group["piano_states"][:]
            hand_joints = group["hand_joints"][:]

            num_trajectories, horizon, _ = actions.shape
            if self.max_traj_per_song is not None:
                num_trajectories = min(num_trajectories, self.max_traj_per_song)
            trajectory_order = rng.permutation(num_trajectories)

            for trajectory_index in trajectory_order:
                for timestep in range(max(horizon - 1, 0)):
                    observation = build_bc_observation(
                        goals_trajectory=goals[trajectory_index],
                        piano_state_t=piano_states[trajectory_index, timestep],
                        hand_joints_t=hand_joints[trajectory_index, timestep],
                        timestep=timestep,
                        n_steps_lookahead=self.n_steps_lookahead,
                    )
                    action = build_bc_action(actions[trajectory_index, timestep], action_dim=self.action_dim)
                    yield torch.from_numpy(observation), torch.from_numpy(action)


class BCPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(self.hidden_dim, self.act_dim)
        self.log_std_head = nn.Linear(self.hidden_dim, self.act_dim)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.net(observation)
        mean = self.mean_head(hidden)
        log_std = self.log_std_head(hidden).clamp(_LOG_STD_MIN, _LOG_STD_MAX)
        return mean, log_std


@dataclass(frozen=True)
class BCCheckpointMetadata:
    obs_dim: int
    act_dim: int
    n_steps_lookahead: int
    hidden_dim: int


def train_bc(
    *,
    model: BCPolicy,
    dataset: IterableDataset[tuple[torch.Tensor, torch.Tensor]],
    save_path: str | os.PathLike[str],
    epochs: int = 5,
    batch_size: int = 1024,
    learning_rate: float = 3e-4,
    use_amp: bool = True,
    device: str | torch.device = "auto",
) -> dict[str, Any]:
    resolved_device = _resolve_device(device)
    save_path = str(save_path)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=resolved_device.type == "cuda",
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and resolved_device.type == "cuda")
    autocast = torch.amp.autocast("cuda", enabled=use_amp and resolved_device.type == "cuda")
    model = model.to(resolved_device)
    history = {"epoch_loss": []}

    for epoch in range(epochs):
        batch_losses: list[float] = []
        epoch_start = time.time()
        model.train()
        for observation_batch, action_batch in loader:
            observation_batch = observation_batch.to(resolved_device, non_blocking=resolved_device.type == "cuda")
            action_batch = action_batch.to(resolved_device, non_blocking=resolved_device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            with autocast:
                mean, _ = model(observation_batch)
                prediction = torch.tanh(mean)
                loss = F.mse_loss(prediction[:, :_RP1M_ACTION_DIM], action_batch[:, :_RP1M_ACTION_DIM])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            batch_losses.append(float(loss.item()))
        history["epoch_loss"].append(float(np.mean(batch_losses)) if batch_losses else math.nan)
        history.setdefault("epoch_s", []).append(float(time.time() - epoch_start))

    metadata = BCCheckpointMetadata(
        obs_dim=model.obs_dim,
        act_dim=model.act_dim,
        n_steps_lookahead=((model.obs_dim - (_FINGERING_DIM + _PIANO_STATE_DIM + _SUSTAIN_DIM + (2 * _HAND_JOINT_DIM))) // _GOAL_STEP_DIM) - 1,
        hidden_dim=model.hidden_dim,
    )
    torch.save(
        {
            "actor": model.state_dict(),
            "metadata": metadata.__dict__,
        },
        save_path,
    )
    history["save_path"] = save_path
    return history


def load_bc_actor_weights(agent_or_actor: Any, checkpoint_path: str | os.PathLike[str]) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    actor = getattr(agent_or_actor, "actor", agent_or_actor)
    if hasattr(actor, "_orig_mod"):
        actor = actor._orig_mod
    actor.load_state_dict(checkpoint["actor"])
    return dict(checkpoint.get("metadata", {}))


def _resolve_device(requested: str | torch.device) -> torch.device:
    if isinstance(requested, torch.device):
        return requested
    if requested not in {"", "auto", None}:
        return torch.device(str(requested))
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
