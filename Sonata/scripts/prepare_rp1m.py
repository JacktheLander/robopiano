from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sonata.config import load_stage_config, resolve_path
from sonata.data.indexer import scan_dataset
from sonata.utils.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare RP1M manifests for Sonata-3.")
    parser.add_argument("--profile", default="debug")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_stage_config("data", profile=args.profile, config_path=args.config)
    config_dir = resolve_path(config["config_path"]).parent
    if args.dataset_root is not None:
        config["dataset_root"] = args.dataset_root
    if args.output_root is not None:
        config["output_root"] = args.output_root
    config["dataset_root"] = str(resolve_path(config["dataset_root"], config_dir))
    config["output_root"] = str(resolve_path(config["output_root"], config_dir))
    config["note_search_roots"] = [str(resolve_path(path, config_dir)) for path in config.get("note_search_roots", [])]
    config["force"] = bool(args.force or config.get("force", False))
    logger = configure_logging(args.log_level)
    outputs = scan_dataset(config=config, logger=logger)
    logger.info("Prepared RP1M manifest at %s", outputs["manifest_base"])


if __name__ == "__main__":
    main()
