from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sonata.config import load_stage_config, resolve_path
from sonata.transformer.trainer import run_transformer_training
from sonata.utils.logging import configure_logging
from sonata.utils.wandb import add_wandb_arguments, apply_wandb_cli_overrides


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Sonata-3 factored transformer planner.")
    parser.add_argument("--profile", default="debug")
    parser.add_argument("--config", default=None)
    parser.add_argument("--primitive-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--log-level", default="INFO")
    add_wandb_arguments(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_stage_config("transformer", profile=args.profile, config_path=args.config)
    config = apply_wandb_cli_overrides(config, args)
    config_dir = resolve_path(config["config_path"]).parent
    if args.primitive_root is not None:
        config["primitive_root"] = args.primitive_root
    if args.output_root is not None:
        config["output_root"] = args.output_root
    config["primitive_root"] = str(resolve_path(config["primitive_root"], config_dir))
    config["output_root"] = str(resolve_path(config["output_root"], config_dir))
    logger = configure_logging(args.log_level)
    outputs = run_transformer_training(config=config, logger=logger)
    logger.info("Transformer checkpoint saved to %s", outputs["best_checkpoint"])


if __name__ == "__main__":
    main()
