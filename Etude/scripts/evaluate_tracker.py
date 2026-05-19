from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from etude.controllers.pd import PDController
from etude.data.trajectory_io import load_qpos_trajectory
from etude.evaluation.metrics import joint_metrics
from etude.robopianist.env_factory import make_robopianist_env
from etude.robopianist.state_mapping import resolve_mapping_from_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an Etude tracker in RoboPianist.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--trajectory", default=None)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--backend", default="dm_control")
    parser.add_argument("--render-video", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.trajectory is None:
        raise ValueError("--trajectory is required until RP1M eval sampling is wired in")
    traj = load_qpos_trajectory(args.trajectory)
    env = make_robopianist_env()
    mapping = resolve_mapping_from_env(env)
    controller = PDController(
        mapping,
        kp=float(config["pd"]["kp_init"]),
        kd=float(config["pd"]["kd_init"]),
        lookahead=int(config["trajectory"]["lookahead_steps"][0]),
    )
    from etude.evaluation.rollout import rollout_controller

    rollout = rollout_controller(env, controller, mapping, traj["q_ref"], traj["qdot_ref"])
    metrics = joint_metrics(rollout["q"], traj["q_ref"][: rollout["q"].shape[0]], rollout["qdot"], traj["qdot_ref"][: rollout["q"].shape[0]])
    metrics.update(controller.diagnostics())
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    np.savez_compressed(output_root / "rollout.npz", **rollout)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
