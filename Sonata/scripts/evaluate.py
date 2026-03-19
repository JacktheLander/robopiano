from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sonata.evaluation.offline import evaluate_offline_pipeline
from sonata.evaluation.rollout import evaluate_dm_control_rollout, evaluate_mjx_physics
from sonata.utils.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Sonata-3 offline and online.")
    parser.add_argument("--primitive-root", required=True)
    parser.add_argument("--diffusion-checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--backend", choices=["offline", "dm_control", "mjx_physics"], default="offline")
    parser.add_argument("--xml-path", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logger = configure_logging(args.log_level)
    primitive_root = Path(args.primitive_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    offline = evaluate_offline_pipeline(
        primitive_root=primitive_root,
        diffusion_checkpoint=Path(args.diffusion_checkpoint).resolve(),
        output_root=output_root / "offline",
        variant=args.variant,
        device=args.device,
    )
    logger.info("Offline metrics: %s", offline["metrics"])
    if args.backend == "dm_control":
        rollout = evaluate_dm_control_rollout(
            primitive_root=primitive_root,
            predictions_by_episode=offline["predictions_by_episode"],
            output_root=output_root / "rollout",
        )
        logger.info("DM Control rollout: %s", rollout)
    elif args.backend == "mjx_physics":
        if args.xml_path is None:
            raise ValueError("--xml-path is required for mjx_physics backend")
        first_episode = next(iter(offline["predictions_by_episode"].values()))
        import numpy as np

        actions = np.stack([item["predicted"] for item in first_episode], axis=0)
        rollout = evaluate_mjx_physics(xml_path=Path(args.xml_path), action_sequences=actions, output_root=output_root / "rollout")
        logger.info("MJX rollout: %s", rollout)


if __name__ == "__main__":
    main()
