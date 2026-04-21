from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sonata.config import deep_update, load_stage_config, load_yaml, resolve_path
from sonata.evaluation.primitive_online_eval import evaluate_primitives_online
from sonata.utils.logging import configure_logging
from sonata.utils.wandb import add_wandb_arguments, apply_wandb_cli_overrides


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Sonata-3 primitive library.")
    parser.add_argument("--profile", default="debug")
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--post-eval-config", default=None)
    parser.add_argument("--post-eval-output-root", default=None)
    parser.add_argument("--run-post-eval", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--robopianist-root", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    add_wandb_arguments(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from sonata.primitives.discovery import run_primitive_pipeline

    config = load_stage_config("primitive", profile=args.profile, config_path=args.config)
    config = apply_wandb_cli_overrides(config, args)
    config_dir = resolve_path(config["config_path"]).parent
    if args.output_root is not None:
        config["output_root"] = args.output_root
    config["output_root"] = str(resolve_path(config["output_root"], config_dir))
    config["data_output_root"] = str(resolve_path(config["data_output_root"], config_dir))
    if args.robopianist_root is not None:
        config["robopianist_root"] = str(resolve_path(args.robopianist_root, config_dir))
    data_config = load_stage_config("data", profile=args.profile)
    data_config_dir = resolve_path(data_config["config_path"]).parent
    data_config["dataset_root"] = str(resolve_path(data_config["dataset_root"], data_config_dir))
    data_config["output_root"] = str(resolve_path(data_config["output_root"], data_config_dir))
    data_config["note_search_roots"] = [str(resolve_path(path, data_config_dir)) for path in data_config.get("note_search_roots", [])]
    config["data_config"] = deep_update(data_config, {"force": bool(args.force or config.get("force", False))})
    config["force"] = bool(args.force or config.get("force", False))
    logger = configure_logging(args.log_level)
    outputs = run_primitive_pipeline(config=config, logger=logger)
    post_eval_cfg = dict(config.get("post_training_eval", {}))
    if args.post_eval_config is not None:
        post_eval_cfg["config_path"] = str(resolve_path(args.post_eval_config, config_dir))
    if args.run_post_eval is not None:
        post_eval_cfg["enabled"] = bool(args.run_post_eval)
    if args.post_eval_output_root is not None:
        post_eval_cfg["output_root"] = str(resolve_path(args.post_eval_output_root, config_dir))
    if bool(post_eval_cfg.get("enabled", False)):
        eval_config_path = resolve_path(post_eval_cfg.get("config_path"), config_dir)
        eval_config = load_yaml(eval_config_path) if eval_config_path is not None else {}
        eval_config["primitive_root"] = str(resolve_path(config["output_root"], config_dir))
        eval_config["output_root"] = str(
            resolve_path(
                post_eval_cfg.get("output_root") or eval_config.get("output_root") or Path(config["output_root"]) / "post_eval",
                config_dir,
            )
        )
        if config.get("robopianist_root") is not None:
            eval_config["robopianist_root"] = str(config["robopianist_root"])
        payload = evaluate_primitives_online(config=eval_config, logger=logger)
        logger.info("Primitive post-training evaluation ready at %s", payload["output_root"])
    logger.info("Primitive library ready at %s", outputs["library_base"])


if __name__ == "__main__":
    main()
