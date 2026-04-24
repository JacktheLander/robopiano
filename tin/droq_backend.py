from __future__ import annotations

from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class RunningNormalizer:
    def __init__(self, shape: tuple[int, ...], *, clip: float = 5.0, eps: float = 1e-8) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 0
        self.clip = clip
        self.eps = eps

    def update(self, value: np.ndarray) -> None:
        array = np.asarray(value, dtype=np.float64)
        if np.any(np.isnan(array)) or np.any(np.isinf(array)):
            return
        self.count += 1
        delta = array - self.mean
        self.mean += delta / self.count
        delta2 = array - self.mean
        self.var = ((self.count - 1) * self.var + delta * delta2) / self.count

    def normalize(self, value: np.ndarray) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64)
        normalized = (array - self.mean) / (np.sqrt(self.var) + self.eps)
        return np.clip(normalized, -self.clip, self.clip).astype(np.float32)


class NStepReplayBuffer:
    def __init__(
        self,
        *,
        capacity: int,
        obs_dim: int,
        act_dim: int,
        batch_size: int,
        n_steps: int = 3,
        gamma: float = 0.99,
        device: str = "cpu",
    ) -> None:
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.batch_size = int(batch_size)
        self.n_steps = int(n_steps)
        self.gamma = float(gamma)
        self.device = torch.device(device)
        self.ptr = 0
        self.size = 0
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.effective_gammas = np.zeros((capacity, 1), dtype=np.float32)
        self.pending: deque[tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]] = deque()

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        done: float | bool,
    ) -> None:
        self.pending.append(
            (
                np.asarray(observation, dtype=np.float32).copy(),
                np.asarray(action, dtype=np.float32).copy(),
                float(reward),
                np.asarray(next_observation, dtype=np.float32).copy(),
                bool(done),
            )
        )
        if done:
            while self.pending:
                self._flush_oldest()
        elif len(self.pending) >= self.n_steps:
            self._flush_oldest()

    def is_ready(self) -> bool:
        return self.size >= self.batch_size

    def sample(self) -> tuple[torch.Tensor, ...]:
        indices = np.random.randint(0, self.size, size=self.batch_size)
        return (
            self._to_device(self.obs[indices]),
            self._to_device(self.actions[indices]),
            self._to_device(self.rewards[indices]),
            self._to_device(self.next_obs[indices]),
            self._to_device(self.dones[indices]),
            self._to_device(self.effective_gammas[indices]),
        )

    def _flush_oldest(self) -> None:
        if not self.pending:
            return
        obs0, action0, _, _, _ = self.pending[0]
        n_step_reward = 0.0
        last_next_obs = self.pending[-1][3]
        last_done = float(self.pending[-1][4])
        effective_gamma = self.gamma ** len(self.pending)

        for offset, (_, _, reward, next_obs, done) in enumerate(self.pending):
            n_step_reward += (self.gamma ** offset) * reward
            if done:
                last_next_obs = next_obs
                last_done = 1.0
                effective_gamma = self.gamma ** (offset + 1)
                break

        self.obs[self.ptr] = obs0
        self.actions[self.ptr] = action0
        self.rewards[self.ptr] = n_step_reward
        self.next_obs[self.ptr] = last_next_obs
        self.dones[self.ptr] = last_done
        self.effective_gammas[self.ptr] = effective_gamma
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.pending.popleft()

    def _to_device(self, array: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(array)
        if self.device.type == "cuda":
            tensor = tensor.pin_memory()
        return tensor.to(self.device, non_blocking=self.device.type == "cuda")


class DroQCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256, dropout: float = 0.01) -> None:
        super().__init__()
        self.q1 = self._build(obs_dim + act_dim, hidden, dropout)
        self.q2 = self._build(obs_dim + act_dim, hidden, dropout)

    @staticmethod
    def _build(input_dim: int, hidden: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.cat([observation, action], dim=-1)
        return self.q1(features), self.q2(features)

    def q_min(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(observation, action)
        return torch.min(q1, q2)


class DroQActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden, act_dim)
        self.log_std_head = nn.Linear(hidden, act_dim)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.net(observation)
        mean = self.mean_head(hidden)
        log_std = self.log_std_head(hidden).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(observation)
        distribution = Normal(mean, log_std.exp())
        latent = distribution.rsample()
        action = torch.tanh(latent)
        log_prob = (distribution.log_prob(latent) - torch.log(1 - action.pow(2) + 1e-6)).sum(-1, keepdim=True)
        return action, log_prob


@dataclass(frozen=True)
class DroQConfig:
    obs_dim: int
    act_dim: int
    device: str
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    batch_size: int = 1024
    hidden: int = 256
    dropout: float = 0.01
    use_amp: bool = True
    min_alpha: float = 0.05
    grad_clip: float = 1.0
    normalize_observations: bool = True
    normalize_rewards: bool = True
    observation_clip: float = 5.0
    reward_clip: float = 10.0
    normalizer_warmup_steps: int = 50


class DroQAgent:
    backend = "droq"

    def __init__(self, config: DroQConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.use_amp = bool(config.use_amp and self.device.type == "cuda")
        self.min_alpha = float(config.min_alpha)
        self.grad_clip = float(config.grad_clip)
        self.normalizer_warmup_steps = int(config.normalizer_warmup_steps)

        self.actor = DroQActor(config.obs_dim, config.act_dim, config.hidden).to(self.device)
        self.critic = DroQCritic(config.obs_dim, config.act_dim, config.hidden, config.dropout).to(self.device)
        self.critic_target = DroQCritic(config.obs_dim, config.act_dim, config.hidden, config.dropout).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.eval()

        self.log_alpha = torch.tensor(math.log(1.0), requires_grad=True, device=self.device)
        self.target_entropy = -float(config.act_dim)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=config.lr)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=config.lr)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.observation_normalizer = (
            RunningNormalizer((config.obs_dim,), clip=config.observation_clip)
            if config.normalize_observations
            else None
        )
        self.reward_normalizer = (
            RunningNormalizer((1,), clip=config.reward_clip)
            if config.normalize_rewards
            else None
        )

    def compile_models(self) -> bool:
        if self.device.type != "cuda" or not hasattr(torch, "compile"):
            return False
        try:
            self.actor = torch.compile(self.actor, mode="reduce-overhead")
            self.critic = torch.compile(self.critic, mode="reduce-overhead")
            self.critic_target = torch.compile(self.critic_target, mode="reduce-overhead")
            return True
        except Exception:
            return False

    def prepare_observation(self, observation: np.ndarray) -> np.ndarray:
        array = np.asarray(observation, dtype=np.float32)
        if self.observation_normalizer is None:
            return array
        if self.observation_normalizer.count <= self.normalizer_warmup_steps:
            return array
        return self.observation_normalizer.normalize(array)

    def update_normalizers(self, observation: np.ndarray, reward: float) -> None:
        if self.observation_normalizer is not None:
            self.observation_normalizer.update(np.asarray(observation, dtype=np.float32))
        if self.reward_normalizer is not None:
            self.reward_normalizer.update(np.asarray([reward], dtype=np.float32))

    def normalize_reward(self, reward: float) -> float:
        if self.reward_normalizer is None:
            return float(reward)
        return float(self.reward_normalizer.normalize(np.asarray([reward], dtype=np.float32))[0])

    def sample_actions(self, observation: np.ndarray) -> tuple[DroQAgent, np.ndarray]:
        action = self._act(self.prepare_observation(observation), deterministic=False)
        return self, action

    def sample_actions_batch(self, observations: np.ndarray) -> tuple[DroQAgent, np.ndarray]:
        return self, self._act_batch(self.prepare_observation_batch(observations), deterministic=False)

    def eval_actions(self, observation: np.ndarray) -> np.ndarray:
        return self._act(self.prepare_observation(observation), deterministic=True)

    def eval_actions_batch(self, observations: np.ndarray) -> np.ndarray:
        return self._act_batch(self.prepare_observation_batch(observations), deterministic=True)

    def update(self, transitions: tuple[torch.Tensor, ...]) -> tuple[DroQAgent, dict[str, float]]:
        observations, actions, rewards, next_observations, dones, effective_gammas = transitions
        amp = torch.amp.autocast("cuda", enabled=self.use_amp) if self.use_amp else nullcontext()

        with torch.no_grad(), amp:
            next_actions, next_log_prob = self.actor.sample(next_observations)
            target_q = self.critic_target.q_min(next_observations, next_actions)
            alpha = self._alpha_tensor()
            target = rewards + effective_gammas * (1 - dones) * (target_q - alpha * next_log_prob)

        self.critic.train()
        with amp:
            q1, q2 = self.critic(observations, actions)
            critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_opt.zero_grad(set_to_none=True)
        self.scaler.scale(critic_loss).backward()
        self.scaler.unscale_(self.critic_opt)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.scaler.step(self.critic_opt)
        self.scaler.update()

        self.critic.eval()
        with amp:
            sampled_actions, log_prob = self.actor.sample(observations)
            actor_loss = (self._alpha_tensor().detach() * log_prob - self.critic.q_min(observations, sampled_actions)).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        self.scaler.scale(actor_loss).backward()
        self.scaler.unscale_(self.actor_opt)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.scaler.step(self.actor_opt)
        self.scaler.update()

        with amp:
            alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        self.scaler.scale(alpha_loss).backward()
        self.scaler.step(self.alpha_opt)
        self.scaler.update()

        with torch.no_grad():
            self.log_alpha.clamp_(min=math.log(self.min_alpha))
            for parameter, target_parameter in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_parameter.data.mul_(1 - self.config.tau).add_(self.config.tau * parameter.data)

        return self, {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(self._alpha_tensor().item()),
        }

    def _act(self, observation: np.ndarray, *, deterministic: bool) -> np.ndarray:
        with torch.no_grad():
            observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
            if deterministic:
                mean, _ = self.actor(observation_tensor)
                action = torch.tanh(mean)
            else:
                action, _ = self.actor.sample(observation_tensor)
        return action.squeeze(0).cpu().numpy().astype(np.float32)

    def _act_batch(self, observations: np.ndarray, *, deterministic: bool) -> np.ndarray:
        with torch.no_grad():
            observation_tensor = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
            if deterministic:
                mean, _ = self.actor(observation_tensor)
                action = torch.tanh(mean)
            else:
                action, _ = self.actor.sample(observation_tensor)
        return action.cpu().numpy().astype(np.float32)

    def _alpha_tensor(self) -> torch.Tensor:
        return self.log_alpha.exp().clamp(min=self.min_alpha)

    def prepare_observation_batch(self, observations: np.ndarray) -> np.ndarray:
        array = np.asarray(observations, dtype=np.float32)
        if array.ndim == 1:
            return self.prepare_observation(array)
        if self.observation_normalizer is None or self.observation_normalizer.count <= self.normalizer_warmup_steps:
            return array
        return np.stack([self.observation_normalizer.normalize(observation) for observation in array], axis=0)
