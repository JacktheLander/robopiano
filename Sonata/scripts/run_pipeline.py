from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sonata.config import deep_update, load_pipeline_config, load_stage_config, resolve_path
from sonata.data.indexer import scan_dataset
from sonata.diffusion.trainer import run_diffusion_training
from sonata.evaluation.offline import evaluate_offline_pipeline
from sonata.primitives.discovery import run_primitive_pipeline
from sonata.transformer.trainer import run_transformer_training
from sonata.utils.logging import configure_logging
from sonata.utils.wandb import add_wandb_arguments, apply_wandb_cli_overrides


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full Sonata-3 pipeline.")
    parser.add_argument("--profile", default="debug")
    parser.add_argument("--config", default=None)
    parser.add_argument("--log-level", default="INFO")
    add_wandb_arguments(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logger = configure_logging(args.log_level)
    pipeline = load_pipeline_config(profile=args.profile, config_path=args.config)

    data_config = load_stage_config("data", profile=pipeline["profiles"]["data"])
    primitive_config = load_stage_config("primitive", profile=pipeline["profiles"]["primitive"])
    transformer_config = load_stage_config("transformer", profile=pipeline["profiles"]["transformer"])
    diffusion_config = load_stage_config("diffusion", profile=pipeline["profiles"]["diffusion"])
    pipeline_wandb = pipeline.get("wandb")
    if isinstance(pipeline_wandb, dict):
        primitive_config = deep_update(primitive_config, {"wandb": pipeline_wandb})
        transformer_config = deep_update(transformer_config, {"wandb": pipeline_wandb})
        diffusion_config = deep_update(diffusion_config, {"wandb": pipeline_wandb})
    primitive_config = apply_wandb_cli_overrides(primitive_config, args)
    transformer_config = apply_wandb_cli_overrides(transformer_config, args)
    diffusion_config = apply_wandb_cli_overrides(diffusion_config, args)

    data_dir = resolve_path(data_config["config_path"]).parent
    primitive_dir = resolve_path(primitive_config["config_path"]).parent
    transformer_dir = resolve_path(transformer_config["config_path"]).parent
    diffusion_dir = resolve_path(diffusion_config["config_path"]).parent

    data_config["output_root"] = str(resolve_path(data_config["output_root"], data_dir))
    primitive_config["output_root"] = str(resolve_path(primitive_config["output_root"], primitive_dir))
    transformer_config["output_root"] = str(resolve_path(transformer_config["output_root"], transformer_dir))
    diffusion_config["output_root"] = str(resolve_path(diffusion_config["output_root"], diffusion_dir))

    data_config["dataset_root"] = str(resolve_path(data_config["dataset_root"], data_dir))
    data_config["note_search_roots"] = [str(resolve_path(path, data_dir)) for path in data_config.get("note_search_roots", [])]
    scan_dataset(config=data_config, logger=logger)

    primitive_config["data_output_root"] = str(resolve_path(primitive_config["data_output_root"], primitive_dir))
    primitive_config["data_manifest_name"] = primitive_config.get("data_manifest_name", "dataset_manifest")
    primitive_config["data_config"] = data_config
    primitive_outputs = run_primitive_pipeline(config=primitive_config, logger=logger)

    transformer_config["primitive_root"] = str(resolve_path(transformer_config["primitive_root"], transformer_dir))
    transformer_outputs = run_transformer_training(config=transformer_config, logger=logger)

    diffusion_config["primitive_root"] = str(resolve_path(diffusion_config["primitive_root"], diffusion_dir))
    diffusion_config["planner_checkpoint"] = str(transformer_outputs["best_checkpoint"])
    diffusion_outputs = run_diffusion_training(config=diffusion_config, logger=logger, joint_refine=False)

    evaluate_offline_pipeline(
        primitive_root=Path(diffusion_config["primitive_root"]).resolve(),
        diffusion_checkpoint=Path(diffusion_outputs["best_checkpoint"]).resolve(),
        output_root=Path(diffusion_config["output_root"]).resolve() / "evaluation",
        variant=diffusion_config.get("variant"),
        device=diffusion_config["device"],
    )
    logger.info("Pipeline complete. Primitive outputs at %s", primitive_outputs["library_base"])
    logger.info("Transformer checkpoint at %s", transformer_outputs["best_checkpoint"])
    logger.info("Diffusion checkpoint at %s", diffusion_outputs["best_checkpoint"])


if __name__ == "__main__":
    main()
