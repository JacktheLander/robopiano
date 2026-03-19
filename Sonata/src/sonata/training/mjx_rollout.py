from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class MJXAvailability:
    available: bool
    message: str


def mjx_availability() -> MJXAvailability:
    try:
        import jax  # noqa: F401
        from mujoco import mjx  # noqa: F401

        return MJXAvailability(True, "jax and mujoco.mjx are available")
    except Exception as exc:  # pragma: no cover
        return MJXAvailability(False, f"MJX unavailable: {exc}")


class MJXRolloutBackend:
    """Physics-only MJX collector for batched MuJoCo stepping.

    This intentionally stops short of reproducing RoboPianist's full dm_control task
    semantics. It is meant to move rollout physics stepping to GPU on HPC once the
    caller provides model XML plus observation/reward glue.
    """

    def __init__(self, xml_path: str | Path, batch_size: int):
        import jax
        import jax.numpy as jnp
        import mujoco
        from mujoco import mjx

        self.jax = jax
        self.jnp = jnp
        self.mujoco = mujoco
        self.mjx = mjx
        self.batch_size = int(batch_size)
        self.model = mujoco.MjModel.from_xml_path(str(Path(xml_path).resolve()))
        self.mjx_model = mjx.put_model(self.model)
        base_data = mjx.make_data(self.mjx_model)
        self.data = jax.tree_util.tree_map(lambda array: jnp.repeat(array[None, ...], self.batch_size, axis=0), base_data)

    def step(self, controls):
        controls = self.jnp.asarray(controls)
        data = self.data.replace(ctrl=controls)
        self.data = self.jax.vmap(self.mjx.step, in_axes=(None, 0))(self.mjx_model, data)
        return self.data

    def qpos(self):
        return self.data.qpos

    def qvel(self):
        return self.data.qvel
