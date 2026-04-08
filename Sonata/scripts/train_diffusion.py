from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sonata.config import load_stage_config, resolve_path
from sonata.diffusion.trainer import run_diffusion_training
from sonata.utils.logging import configure_logging
from sonata.utils.wandb import add_wandb_arguments, apply_wandb_cli_overrides


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Sonata-3 diffusion refiner.")
    parser.add_argument("--profile", default="debug")
    parser.add_argument("--config", default=None)
    parser.add_argument("--primitive-root", default=None)
    parser.add_argument("--planner-checkpoint", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--log-level", default="INFO")
    add_wandb_arguments(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_stage_config("diffusion", profile=args.profile, config_path=args.config)
    config = apply_wandb_cli_overrides(config, args)
    config_dir = resolve_path(config["config_path"]).parent
    cwd = Path.cwd()
    if args.primitive_root is not None:
        config["primitive_root"] = str(resolve_path(args.primitive_root, cwd))
    if args.planner_checkpoint is not None:
        config["planner_checkpoint"] = str(resolve_path(args.planner_checkpoint, cwd))
    if args.output_root is not None:
        config["output_root"] = str(resolve_path(args.output_root, cwd))
    if args.variant is not None:
        config["variant"] = args.variant
    config["primitive_root"] = str(resolve_path(config["primitive_root"], config_dir))
    if config.get("planner_checkpoint"):
        config["planner_checkpoint"] = str(resolve_path(config["planner_checkpoint"], config_dir))
    config["output_root"] = str(resolve_path(config["output_root"], config_dir))
    logger = configure_logging(args.log_level)
    outputs = run_diffusion_training(config=config, logger=logger, joint_refine=False)
    logger.info("Diffusion checkpoint saved to %s", outputs["best_checkpoint"])


if __name__ == "__main__":
    main()
