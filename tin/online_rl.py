from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Callable, Optional, Tuple
import time

import dm_env_wrappers as wrappers
import numpy as np
import replay
import robopianist.wrappers as robopianist_wrappers
import sac
import specs
from robopianist import suite
from tqdm import tqdm


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
    agent_config: sac.SACConfig = sac.SACConfig()


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


def get_env(
    args: TrainArgs,
    *,
    record_dir: Optional[Path] = None,
    midi_file: Optional[Path] = None,
    enable_midi_metrics: bool = False,
):
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


def initialize_agent_and_replay(args: TrainArgs, env: Any) -> tuple[Any, Any, Any, dict[str, Any]]:
    spec = specs.EnvironmentSpec.make(env)
    requested_device = args.device or "auto"
    resolved_device = resolve_device(requested_device)
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
    return spec, agent, replay_buffer, {
        "requested_device": requested_device,
        "resolved_device": resolved_device,
        "agent_device_applied": bool(applied_fields),
        "agent_device_fields": ",".join(applied_fields),
    }


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


def safe_close(env: Any) -> None:
    if env is None:
        return
    close = getattr(env, "close", None)
    if callable(close):
        close()


def _apply_agent_device(agent_config: Any, device: str) -> tuple[Any, list[str]]:
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
