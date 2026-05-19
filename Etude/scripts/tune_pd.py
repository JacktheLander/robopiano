from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from etude.controllers.pd import PDController
from etude.evaluation.metrics import action_metrics
from etude.robopianist.state_mapping import StateMapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline PD gain sweep against Etude dataset states.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    manifest = np.genfromtxt(Path(args.dataset_root) / "manifest.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")
    first_path = Path(args.dataset_root) / str(np.atleast_1d(manifest["path"])[0])
    with np.load(first_path, allow_pickle=False) as episode:
        action_dim = episode["actions"].shape[1]
        low = -np.ones(action_dim, dtype=np.float32)
        high = np.ones(action_dim, dtype=np.float32)
        mapping = StateMapping(list(range(46)), list(range(46)), None, None, None, low, high)
        best = None
        for kp in (4.0, 8.0, float(config["pd"]["kp_init"]), 16.0):
            for kd in (0.2, float(config["pd"]["kd_init"]), 1.0):
                controller = PDController(mapping, kp=kp, kd=kd, lookahead=1)
                controller.reset(episode["q_ref"], episode["qdot_ref"], {"dt": float(episode["dt"])})
                actions = [
                    controller.act({"q": episode["q"][t], "qdot": episode["qdot"][t]}, t)
                    for t in range(episode["q"].shape[0])
                ]
                metrics = action_metrics(np.asarray(actions), low, high)
                score = metrics["control/action_clip_rate"]
                candidate = {"kp": kp, "kd": kd, **metrics}
                if best is None or score < best["control/action_clip_rate"]:
                    best = candidate
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "best_pd.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
