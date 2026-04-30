from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Callable, Optional, Tuple
import time

import numpy as np

try:
    import dm_env_wrappers as wrappers
except ImportError:  # pragma: no cover
    wrappers = None

try:
    import replay
except ImportError:  # pragma: no cover
    replay = None

try:
    import robopianist.wrappers as robopianist_wrappers
except ImportError:  # pragma: no cover
    robopianist_wrappers = None

try:
    import sac
except ImportError:  # pragma: no cover
    sac = None

try:
    import specs
except ImportError:  # pragma: no cover
    specs = None

try:
    from robopianist import suite
except ImportError:  # pragma: no cover
    suite = None

try:
    from tin.bc import load_bc_actor_weights
except ImportError:  # pragma: no cover
    from bc import load_bc_actor_weights
from tin.droq_backend import DroQAgent, DroQConfig, NStepReplayBuffer
from tqdm import tqdm


def _default_sac_config() -> Any:
    return sac.SACConfig() if sac is not None else None


@dataclass(frozen=True)
class TrainArgs:
    root_dir: str = "/tmp/robopianist"
    seed: int = 42
    max_steps: int = 1_000_000
    warmstart_steps: int = 5_000
    log_interval: int = 1_000
    eval_interval: int = 10_000
    eval_episodes: int = 1
    batch_size: int = 256
    discount: float = 0.99
    tqdm_bar: bool = False
    replay_capacity: int = 1_000_000
    project: str = "robopianist"
    entity: str = "tnguyen31-santa-clara-university"
    name: str = ""
    tags: str = ""
    notes: str = ""
    mode: str = "online"
    environment_name: str = "RoboPianist-debug-TwinkleTwinkleRousseau-v0"
    n_steps_lookahead: int = 10
    trim_silence: bool = False
    gravity_compensation: bool = False
    reduced_action_space: bool = False
    control_timestep: float = 0.05
    stretch_factor: float = 1.0
    shift_factor: int = 0
    wrong_press_termination: bool = False
    disable_fingering_reward: bool = False
    disable_forearm_reward: bool = False
    disable_colorization: bool = False
    disable_hand_collisions: bool = False
    disable_key_proximity_reward: bool = False
    disable_smooth_motion_reward: bool = False
    disable_anticipation_reward: bool = False
    primitive_fingertip_collisions: bool = False
    frame_stack: int = 1
    clip: bool = True
    record_dir: Optional[Path] = None
    record_every: int = 1
    record_resolution: Tuple[int, int] = (480, 640)
    camera_id: Optional[str | int] = "piano/back"
    action_reward_observation: bool = False
    device: str = "auto"
    agent_backend: str = "auto"
    utd_ratio: int = 20
    n_step_return: int = 3
    droq_hidden_dim: int = 256
    droq_dropout: float = 0.01
    droq_tau: float = 0.005
    droq_lr: float = 3e-4
    droq_min_alpha: float = 0.05
    droq_grad_clip: float = 1.0
    normalize_observations: bool = True
    normalize_rewards: bool = True
    normalizer_warmup_steps: int = 50
    observation_normalizer_clip: float = 5.0
    reward_normalizer_clip: float = 10.0
    compile_models: bool = True
    bc_checkpoint: Optional[Path] = None
    use_mjx: bool = False
    n_mjx_envs: int = 4
    mjx_prefer_warp: bool = False
    agent_config: Any = field(default_factory=_default_sac_config)


def prefix_dict(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}/{key}": value for key, value in values.items()}


def resolve_device(requested: str) -> str:
    if requested and requested != "auto":
        return requested
    try:
        import torch
    except Exception:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_backend(requested: str, device: str) -> str:
    if requested and requested != "auto":
        return requested
    return "droq" if device.startswith("cuda") else "sac"


def get_env(
    args: TrainArgs,
    *,
    record_dir: Optional[Path] = None,
    midi_file: Optional[Path] = None,
    enable_midi_metrics: bool = False,
):
    _require_env_runtime()
    env = suite.load(
        environment_name=args.environment_name,
        midi_file=midi_file,
        seed=args.seed,
        stretch=args.stretch_factor,
        shift=args.shift_factor,
        task_kwargs=dict(
            n_steps_lookahead=args.n_steps_lookahead,
            trim_silence=args.trim_silence,
            gravity_compensation=args.gravity_compensation,
            reduced_action_space=args.reduced_action_space,
            control_timestep=args.control_timestep,
            wrong_press_termination=args.wrong_press_termination,
            disable_fingering_reward=args.disable_fingering_reward,
            disable_forearm_reward=args.disable_forearm_reward,
            disable_colorization=args.disable_colorization,
            disable_hand_collisions=args.disable_hand_collisions,
            disable_key_proximity_reward=args.disable_key_proximity_reward,
            disable_smooth_motion_reward=args.disable_smooth_motion_reward,
            disable_anticipation_reward=args.disable_anticipation_reward,
            primitive_fingertip_collisions=args.primitive_fingertip_collisions,
            change_color_on_activation=True,
        ),
    )
    if record_dir is not None:
        env = robopianist_wrappers.PianoSoundVideoWrapper(
            environment=env,
            record_dir=record_dir,
            record_every=args.record_every,
            camera_id=args.camera_id,
            height=args.record_resolution[0],
            width=args.record_resolution[1],
        )
    deque_size = args.record_every if record_dir is not None else 1
    env = wrappers.EpisodeStatisticsWrapper(environment=env, deque_size=deque_size)
    if record_dir is not None or enable_midi_metrics:
        env = robopianist_wrappers.MidiEvaluationWrapper(environment=env, deque_size=deque_size)
    if args.action_reward_observation:
        env = wrappers.ObservationActionRewardWrapper(env)
    env = wrappers.ConcatObservationWrapper(env)
    if args.frame_stack > 1:
        env = wrappers.FrameStackingWrapper(env, num_frames=args.frame_stack, flatten=True)
    env = wrappers.CanonicalSpecWrapper(env, clip=args.clip)
    env = wrappers.SinglePrecisionWrapper(env)
    env = wrappers.DmControlWrapper(env)
    return env


def get_train_env(
    args: TrainArgs,
    *,
    midi_file: Optional[Path] = None,
):
    if args.use_mjx:
        try:
            from tin.mjx_env import MJXBatchedEnv
        except ImportError:  # pragma: no cover
            from mjx_env import MJXBatchedEnv
        return MJXBatchedEnv(
            args,
            midi_file=midi_file,
            n_envs=max(int(args.n_mjx_envs), 1),
            prefer_warp=bool(args.mjx_prefer_warp),
        )
    return get_env(args, midi_file=midi_file)


def initialize_agent_and_replay(args: TrainArgs, env: Any) -> tuple[Any, Any, Any, dict[str, Any]]:
    requested_device = args.device or "auto"
    resolved_device = resolve_device(requested_device)
    requested_backend = args.agent_backend or "auto"
    resolved_backend = resolve_backend(requested_backend, resolved_device)
    device_info = {
        "requested_device": requested_device,
        "resolved_device": resolved_device,
        "requested_backend": requested_backend,
        "resolved_backend": resolved_backend,
    }

    if resolved_backend == "droq":
        initial_timestep = env.reset()
        observation_dim, action_dim = _infer_env_dims(env, initial_timestep)
        agent = DroQAgent(
            DroQConfig(
                obs_dim=observation_dim,
                act_dim=action_dim,
                device=resolved_device,
                gamma=args.discount,
                tau=args.droq_tau,
                lr=args.droq_lr,
                batch_size=args.batch_size,
                hidden=args.droq_hidden_dim,
                dropout=args.droq_dropout,
                min_alpha=args.droq_min_alpha,
                grad_clip=args.droq_grad_clip,
                normalize_observations=args.normalize_observations,
                normalize_rewards=args.normalize_rewards,
                observation_clip=args.observation_normalizer_clip,
                reward_clip=args.reward_normalizer_clip,
                normalizer_warmup_steps=args.normalizer_warmup_steps,
            )
        )
        bc_metadata = {}
        if args.bc_checkpoint:
            bc_metadata = load_bc_actor_weights(agent, args.bc_checkpoint)
        compiled = bool(args.compile_models and agent.compile_models())
        replay_buffer = NStepReplayBuffer(
            capacity=args.replay_capacity,
            obs_dim=observation_dim,
            act_dim=action_dim,
            batch_size=args.batch_size,
            n_steps=args.n_step_return,
            gamma=args.discount,
            device=resolved_device,
        )
        device_info.update(
            {
                "agent_device_applied": True,
                "agent_device_fields": "device",
                "droq_compiled": compiled,
                "observation_dim": observation_dim,
                "action_dim": action_dim,
                "bc_checkpoint": str(args.bc_checkpoint) if args.bc_checkpoint else None,
                "bc_metadata": bc_metadata,
            }
        )
        return None, agent, replay_buffer, device_info

    _require_sac_runtime()
    spec = specs.EnvironmentSpec.make(env)
    agent_config = deepcopy(args.agent_config)
    agent_config, applied_fields = _apply_agent_device(agent_config, resolved_device)
    agent = sac.SAC.initialize(
        spec=spec,
        config=agent_config,
        seed=args.seed,
        discount=args.discount,
    )
    replay_buffer = replay.Buffer(
        state_dim=spec.observation_dim,
        action_dim=spec.action_dim,
        max_size=args.replay_capacity,
        batch_size=args.batch_size,
    )
    device_info.update(
        {
            "agent_device_applied": bool(applied_fields),
            "agent_device_fields": ",".join(applied_fields),
            "droq_compiled": False,
            "observation_dim": spec.observation_dim,
            "action_dim": spec.action_dim,
        }
    )
    return spec, agent, replay_buffer, device_info


def run_eval_episodes(agent: Any, env: Any, num_episodes: int) -> dict[str, Any]:
    latest_filename = None
    for _ in range(num_episodes):
        timestep = env.reset()
        while not timestep.last():
            timestep = env.step(agent.eval_actions(timestep.observation))
        latest_filename = getattr(env, "latest_filename", latest_filename)
    statistics = env.get_statistics() if hasattr(env, "get_statistics") else {}
    music = env.get_musical_metrics() if hasattr(env, "get_musical_metrics") else {}
    return {
        "statistics": dict(statistics),
        "music": dict(music),
        "latest_filename": latest_filename,
    }


def train_online(
    *,
    args: TrainArgs,
    env: Any,
    spec: Any,
    agent: Any,
    replay_buffer: Any,
    eval_env: Any | None = None,
    on_episode_end: Callable[[int, dict[str, float]], None] | None = None,
    on_train_metrics: Callable[[int, dict[str, float]], None] | None = None,
    on_eval: Callable[[int, dict[str, Any]], None] | None = None,
    on_fps: Callable[[int, int], None] | None = None,
) -> tuple[Any, dict[str, float]]:
    if getattr(agent, "backend", "") == "droq":
        if getattr(env, "batched", False):
            return _train_online_droq_batched(
                args=args,
                env=env,
                agent=agent,
                replay_buffer=replay_buffer,
                eval_env=eval_env,
                on_episode_end=on_episode_end,
                on_train_metrics=on_train_metrics,
                on_eval=on_eval,
                on_fps=on_fps,
            )
        return _train_online_droq(
            args=args,
            env=env,
            agent=agent,
            replay_buffer=replay_buffer,
            eval_env=eval_env,
            on_episode_end=on_episode_end,
            on_train_metrics=on_train_metrics,
            on_eval=on_eval,
            on_fps=on_fps,
        )
    return _train_online_sac(
        args=args,
        env=env,
        spec=spec,
        agent=agent,
        replay_buffer=replay_buffer,
        eval_env=eval_env,
        on_episode_end=on_episode_end,
        on_train_metrics=on_train_metrics,
        on_eval=on_eval,
        on_fps=on_fps,
    )


def safe_close(env: Any) -> None:
    if env is None:
        return
    close = getattr(env, "close", None)
    if callable(close):
        close()


def _train_online_sac(
    *,
    args: TrainArgs,
    env: Any,
    spec: Any,
    agent: Any,
    replay_buffer: Any,
    eval_env: Any | None = None,
    on_episode_end: Callable[[int, dict[str, float]], None] | None = None,
    on_train_metrics: Callable[[int, dict[str, float]], None] | None = None,
    on_eval: Callable[[int, dict[str, Any]], None] | None = None,
    on_fps: Callable[[int, int], None] | None = None,
) -> tuple[Any, dict[str, float]]:
    timestep = env.reset()
    replay_buffer.insert(timestep, None)
    start_time = time.time()

    for step in tqdm(range(1, args.max_steps + 1), disable=not args.tqdm_bar):
        if step < args.warmstart_steps:
            action = spec.sample_action(random_state=env.random_state)
        else:
            agent, action = agent.sample_actions(timestep.observation)

        timestep = env.step(action)
        replay_buffer.insert(timestep, action)

        if timestep.last():
            if on_episode_end is not None:
                on_episode_end(step, dict(env.get_statistics()))
            timestep = env.reset()
            replay_buffer.insert(timestep, None)

        if step >= args.warmstart_steps and replay_buffer.is_ready():
            transitions = replay_buffer.sample()
            agent, metrics = agent.update(transitions)
            if step % args.log_interval == 0 and on_train_metrics is not None:
                on_train_metrics(step, dict(metrics))

        if eval_env is not None and args.eval_interval > 0 and step % args.eval_interval == 0:
            eval_payload = run_eval_episodes(agent, eval_env, max(args.eval_episodes, 1))
            if on_eval is not None:
                on_eval(step, eval_payload)

        if step % args.log_interval == 0 and on_fps is not None:
            elapsed = max(time.time() - start_time, 1e-6)
            on_fps(step, int(step / elapsed))

    elapsed_s = time.time() - start_time
    return agent, {
        "steps": float(args.max_steps),
        "elapsed_s": float(elapsed_s),
        "fps": float(args.max_steps / max(elapsed_s, 1e-6)),
    }


def _train_online_droq(
    *,
    args: TrainArgs,
    env: Any,
    agent: DroQAgent,
    replay_buffer: NStepReplayBuffer,
    eval_env: Any | None = None,
    on_episode_end: Callable[[int, dict[str, float]], None] | None = None,
    on_train_metrics: Callable[[int, dict[str, float]], None] | None = None,
    on_eval: Callable[[int, dict[str, Any]], None] | None = None,
    on_fps: Callable[[int, int], None] | None = None,
) -> tuple[DroQAgent, dict[str, float]]:
    timestep = env.reset()
    raw_observation = _to_numpy_observation(timestep.observation)
    start_time = time.time()

    for step in tqdm(range(1, args.max_steps + 1), disable=not args.tqdm_bar):
        normalized_observation = agent.prepare_observation(raw_observation)
        if step < args.warmstart_steps:
            action = _sample_random_action(env)
        else:
            _, action = agent.sample_actions(raw_observation)

        timestep = env.step(action)
        next_raw_observation = _to_numpy_observation(timestep.observation)
        reward = float(timestep.reward or 0.0)
        done = float(timestep.last())
        agent.update_normalizers(raw_observation, reward)
        next_observation = agent.prepare_observation(next_raw_observation)
        replay_buffer.add(
            normalized_observation,
            np.asarray(action, dtype=np.float32),
            agent.normalize_reward(reward),
            next_observation,
            done,
        )
        raw_observation = next_raw_observation

        if timestep.last():
            if on_episode_end is not None:
                on_episode_end(step, dict(env.get_statistics()))
            timestep = env.reset()
            raw_observation = _to_numpy_observation(timestep.observation)

        if step >= args.warmstart_steps and replay_buffer.is_ready():
            for _ in range(max(args.utd_ratio, 1)):
                transitions = replay_buffer.sample()
                agent, metrics = agent.update(transitions)
            if step % args.log_interval == 0 and on_train_metrics is not None:
                on_train_metrics(step, dict(metrics))

        if eval_env is not None and args.eval_interval > 0 and step % args.eval_interval == 0:
            eval_payload = run_eval_episodes(agent, eval_env, max(args.eval_episodes, 1))
            if on_eval is not None:
                on_eval(step, eval_payload)

        if step % args.log_interval == 0 and on_fps is not None:
            elapsed = max(time.time() - start_time, 1e-6)
            on_fps(step, int(step / elapsed))

    elapsed_s = time.time() - start_time
    return agent, {
        "steps": float(args.max_steps),
        "elapsed_s": float(elapsed_s),
        "fps": float(args.max_steps / max(elapsed_s, 1e-6)),
    }


def _train_online_droq_batched(
    *,
    args: TrainArgs,
    env: Any,
    agent: DroQAgent,
    replay_buffer: NStepReplayBuffer,
    eval_env: Any | None = None,
    on_episode_end: Callable[[int, dict[str, float]], None] | None = None,
    on_train_metrics: Callable[[int, dict[str, float]], None] | None = None,
    on_eval: Callable[[int, dict[str, Any]], None] | None = None,
    on_fps: Callable[[int, int], None] | None = None,
) -> tuple[DroQAgent, dict[str, float]]:
    observation_batch = np.asarray(env.reset(), dtype=np.float32)
    start_time = time.time()
    total_steps = 0
    metrics: dict[str, float] = {}
    batch_size = int(observation_batch.shape[0])

    while total_steps < args.max_steps:
        normalized_batch = agent.prepare_observation_batch(observation_batch)
        if total_steps < args.warmstart_steps:
            action_batch = _sample_random_actions_batch(env, batch_size)
        else:
            _, action_batch = agent.sample_actions_batch(observation_batch)

        next_observation_batch, reward_batch, done_batch = env.step(action_batch)
        next_observation_batch = np.asarray(next_observation_batch, dtype=np.float32)
        reward_batch = np.asarray(reward_batch, dtype=np.float32).reshape(-1)
        done_batch = np.asarray(done_batch, dtype=np.float32).reshape(-1)

        remaining_steps = max(args.max_steps - total_steps, 0)
        processed_steps = min(batch_size, remaining_steps)
        for index in range(processed_steps):
            reward = float(reward_batch[index])
            agent.update_normalizers(observation_batch[index], reward)
            replay_buffer.add(
                normalized_batch[index],
                np.asarray(action_batch[index], dtype=np.float32),
                agent.normalize_reward(reward),
                agent.prepare_observation(next_observation_batch[index]),
                float(done_batch[index]),
            )
        total_steps += processed_steps
        observation_batch = next_observation_batch

        if total_steps >= args.warmstart_steps and replay_buffer.is_ready():
            update_count = processed_steps * max(args.utd_ratio, 1)
            for _ in range(update_count):
                transitions = replay_buffer.sample()
                agent, metrics = agent.update(transitions)
            if total_steps % args.log_interval == 0 and on_train_metrics is not None:
                on_train_metrics(total_steps, dict(metrics))

        if bool(done_batch[:processed_steps].all()):
            if on_episode_end is not None:
                on_episode_end(
                    total_steps,
                    {
                        "episode_return": float(np.mean(reward_batch[:processed_steps])),
                        "episode_length": float(getattr(env, "_episode_horizon", processed_steps)),
                    },
                )
            observation_batch = np.asarray(env.reset(), dtype=np.float32)

        if eval_env is not None and args.eval_interval > 0 and total_steps % args.eval_interval < processed_steps:
            eval_payload = run_eval_episodes(agent, eval_env, max(args.eval_episodes, 1))
            if on_eval is not None:
                on_eval(total_steps, eval_payload)

        if total_steps % args.log_interval < processed_steps and on_fps is not None:
            elapsed = max(time.time() - start_time, 1e-6)
            on_fps(total_steps, int(total_steps / elapsed))

    elapsed_s = time.time() - start_time
    return agent, {
        "steps": float(total_steps),
        "elapsed_s": float(elapsed_s),
        "fps": float(total_steps / max(elapsed_s, 1e-6)),
    }


def _infer_env_dims(env: Any, timestep: Any) -> tuple[int, int]:
    observation = timestep.observation if hasattr(timestep, "observation") else timestep
    observation_array = np.asarray(observation, dtype=np.float32)
    if getattr(env, "batched", False) and observation_array.ndim > 1:
        observation_dim = int(observation_array.shape[-1])
    else:
        observation_dim = int(_to_numpy_observation(observation).size)
    action_spec = env.action_spec()
    action_dim = int(np.prod(action_spec.shape))
    return observation_dim, action_dim


def _sample_random_action(env: Any) -> np.ndarray:
    action_spec = env.action_spec()
    return np.random.uniform(action_spec.minimum, action_spec.maximum, action_spec.shape).astype(np.float32)


def _sample_random_actions_batch(env: Any, batch_size: int) -> np.ndarray:
    action_spec = env.action_spec()
    return np.random.uniform(action_spec.minimum, action_spec.maximum, size=(batch_size, *action_spec.shape)).astype(
        np.float32
    )


def _to_numpy_observation(observation: Any) -> np.ndarray:
    return np.asarray(observation, dtype=np.float32).reshape(-1)


def _require_env_runtime() -> None:
    missing = []
    if suite is None:
        missing.append("robopianist")
    if wrappers is None:
        missing.append("dm_env_wrappers")
    if robopianist_wrappers is None:
        missing.append("robopianist.wrappers")
    if missing:
        raise ImportError(f"Missing environment runtime dependencies: {', '.join(missing)}")


def _require_sac_runtime() -> None:
    missing = [name for name, module in (("sac", sac), ("replay", replay), ("specs", specs)) if module is None]
    if missing:
        raise ImportError(f"Missing SAC backend dependencies: {', '.join(missing)}")


def _apply_agent_device(agent_config: Any, device: str) -> tuple[Any, list[str]]:
    if agent_config is None:
        return None, []
    candidate_fields = [name for name in ("device", "torch_device") if _has_field(agent_config, name)]
    if not candidate_fields:
        return agent_config, []
    if is_dataclass(agent_config):
        updates = {name: device for name in candidate_fields}
        return replace(agent_config, **updates), candidate_fields
    for name in candidate_fields:
        setattr(agent_config, name, device)
    return agent_config, candidate_fields


def _has_field(agent_config: Any, name: str) -> bool:
    if is_dataclass(agent_config):
        return any(field.name == name for field in fields(agent_config))
    return hasattr(agent_config, name)
