from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

import numpy as np


_GOAL_STEP_DIM = 89
_FINGERING_DIM = 10
_PIANO_STATE_DIM = 88
_SUSTAIN_DIM = 1
_HAND_JOINT_DIM = 26


@dataclass(frozen=True)
class MJXObservationLayout:
    obs_dim: int
    goal_offset: int
    piano_offset: int
    sustain_offset: int
    rh_offset: int
    lh_offset: int


def mjx_layout(n_steps_lookahead: int) -> MJXObservationLayout:
    goal_dim = (int(n_steps_lookahead) + 1) * _GOAL_STEP_DIM
    piano_offset = goal_dim + _FINGERING_DIM
    sustain_offset = piano_offset + _PIANO_STATE_DIM
    rh_offset = sustain_offset + _SUSTAIN_DIM
    lh_offset = rh_offset + _HAND_JOINT_DIM
    return MJXObservationLayout(
        obs_dim=lh_offset + _HAND_JOINT_DIM,
        goal_offset=0,
        piano_offset=piano_offset,
        sustain_offset=sustain_offset,
        rh_offset=rh_offset,
        lh_offset=lh_offset,
    )


def build_mjx_base_env(args: Any, *, midi_file: Path | None = None):
    try:
        from robopianist import suite
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError("Missing robopianist runtime for MJX environment setup.") from exc

    return suite.load(
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


class MJXBatchedEnv:
    batched = True

    def __init__(
        self,
        args: Any,
        *,
        midi_file: Path | None = None,
        n_envs: int = 4,
        prefer_warp: bool = False,
    ) -> None:
        try:
            import jax
            import jax.numpy as jnp
            import mujoco
            import mujoco.mjx as mjx
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise ImportError("MJX training requires jax and mujoco.mjx to be installed.") from exc

        self._jax = jax
        self._jnp = jnp
        self._mujoco = mujoco
        self._mjx = mjx
        self.args = args
        self.n = int(n_envs)
        self.layout = mjx_layout(args.n_steps_lookahead)

        self._ref_env = build_mjx_base_env(args, midi_file=midi_file)
        self._action_spec = self._ref_env.action_spec()
        self._action_min = np.asarray(self._action_spec.minimum, dtype=np.float32)
        self._action_max = np.asarray(self._action_spec.maximum, dtype=np.float32)

        mjcf_root = self._ref_env.task._arena._mjcf_root
        xml_string = mjcf_root.to_xml_string()
        xml_assets = mjcf_root.get_assets()
        self._mj_model = mujoco.MjModel.from_xml_string(xml_string, xml_assets)
        self._prune_collisions()

        if prefer_warp:
            try:
                self.mx = mjx.put_model(self._mj_model, impl="warp")
            except Exception:
                self.mx = mjx.put_model(self._mj_model)
        else:
            self.mx = mjx.put_model(self._mj_model)

        init_timestep = self._ref_env.reset()
        del init_timestep
        init_qpos = jnp.array(self._ref_env.physics.data.qpos, dtype=jnp.float32)
        init_qvel = jnp.array(self._ref_env.physics.data.qvel, dtype=jnp.float32)
        self._data_init = mjx.make_data(self.mx).replace(qpos=init_qpos, qvel=init_qvel)
        self._nu = int(self._mj_model.nu)
        self._goal_sequence = self._extract_goal_sequence()
        self._episode_horizon = int(self._goal_sequence.shape[0])
        self._piano_slice = slice(0, _PIANO_STATE_DIM)
        self._rh_slice, self._lh_slice = self._find_hand_slices()
        self._jit_step = jax.jit(jax.vmap(lambda data, ctrl: mjx.step(self.mx, data.replace(ctrl=ctrl))))
        self._data_batch = None
        self._step_counters = np.zeros(self.n, dtype=np.int32)
        self.reset()

    def action_spec(self):
        return self._action_spec

    def reset(self) -> np.ndarray:
        self._data_batch = self._jax.vmap(lambda _: self._data_init)(self._jnp.arange(self.n))
        self._step_counters[:] = 0
        return self._get_observation_batch()

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        clipped_actions = np.clip(np.asarray(actions, dtype=np.float32), self._action_min, self._action_max)
        ctrl = self._jnp.array(clipped_actions[:, : self._nu], dtype=self._jnp.float32)
        self._data_batch = self._jit_step(self._data_batch, ctrl)
        self._step_counters += 1
        observations = self._get_observation_batch()
        rewards = self._compute_f1_rewards()
        dones = self._step_counters >= self._episode_horizon
        return observations, rewards.astype(np.float32), dones.astype(np.float32)

    def _prune_collisions(self) -> None:
        keep_tokens = ("distal", "piano", "key")
        mujoco = self._mujoco
        for geom_index in range(self._mj_model.ngeom):
            body_id = self._mj_model.geom_bodyid[geom_index]
            body_name = mujoco.mj_id2name(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            if ("rh_" in body_name or "lh_" in body_name) and not any(token in body_name.lower() for token in keep_tokens):
                self._mj_model.geom_contype[geom_index] = 0
                self._mj_model.geom_conaffinity[geom_index] = 0

    def _extract_goal_sequence(self) -> np.ndarray:
        zero_action = np.zeros(self._action_spec.shape, dtype=np.float32)
        sequence: list[np.ndarray] = []
        timestep = self._ref_env.reset()
        while True:
            goal = np.asarray(timestep.observation.get("goal"), dtype=np.float32).reshape(-1)
            sequence.append(goal.copy())
            if timestep.last():
                break
            timestep = self._ref_env.step(zero_action)
        return np.stack(sequence, axis=0)

    def _find_hand_slices(self) -> tuple[slice, slice]:
        mujoco = self._mujoco
        rh_addresses: list[int] = []
        lh_addresses: list[int] = []
        for joint_index in range(self._mj_model.njnt):
            name = mujoco.mj_id2name(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_index) or ""
            qpos_address = int(self._mj_model.jnt_qposadr[joint_index])
            if name.startswith("rh_"):
                rh_addresses.append(qpos_address)
            elif name.startswith("lh_"):
                lh_addresses.append(qpos_address)

        if len(rh_addresses) >= _HAND_JOINT_DIM and len(lh_addresses) >= _HAND_JOINT_DIM:
            rh_start = min(rh_addresses)
            lh_start = min(lh_addresses)
            return slice(rh_start, rh_start + _HAND_JOINT_DIM), slice(lh_start, lh_start + _HAND_JOINT_DIM)

        return slice(_PIANO_STATE_DIM, _PIANO_STATE_DIM + _HAND_JOINT_DIM), slice(
            _PIANO_STATE_DIM + _HAND_JOINT_DIM,
            _PIANO_STATE_DIM + (2 * _HAND_JOINT_DIM),
        )

    def _get_observation_batch(self) -> np.ndarray:
        qpos = np.asarray(self._data_batch.qpos, dtype=np.float32)
        piano_state = qpos[:, self._piano_slice]
        rh_joints = qpos[:, self._rh_slice]
        lh_joints = qpos[:, self._lh_slice]
        goal_index = np.minimum(self._step_counters, self._episode_horizon - 1)
        goals = self._goal_sequence[goal_index]

        observation = np.zeros((self.n, self.layout.obs_dim), dtype=np.float32)
        observation[:, self.layout.goal_offset : self.layout.goal_offset + goals.shape[1]] = goals
        observation[:, self.layout.piano_offset : self.layout.piano_offset + _PIANO_STATE_DIM] = piano_state
        observation[:, self.layout.rh_offset : self.layout.rh_offset + _HAND_JOINT_DIM] = rh_joints
        observation[:, self.layout.lh_offset : self.layout.lh_offset + _HAND_JOINT_DIM] = lh_joints
        return np.nan_to_num(observation, copy=False)

    def _compute_f1_rewards(self) -> np.ndarray:
        qpos = np.asarray(self._data_batch.qpos, dtype=np.float32)
        piano_state = qpos[:, self._piano_slice]
        goal_index = np.minimum(self._step_counters - 1, self._episode_horizon - 1)
        targets = self._goal_sequence[goal_index, :_PIANO_STATE_DIM] > 0.1
        pressed = piano_state > 0.1
        true_positive = np.sum(pressed & targets, axis=1, dtype=np.float32)
        false_positive = np.sum(pressed & ~targets, axis=1, dtype=np.float32)
        false_negative = np.sum(~pressed & targets, axis=1, dtype=np.float32)
        return 2.0 * true_positive / (2.0 * true_positive + false_positive + false_negative + 1e-8)
