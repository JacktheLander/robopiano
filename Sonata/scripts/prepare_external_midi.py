from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sonata.data.indexer import index_external_midi_dataset
from sonata.utils.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Index an external MIDI corpus for Sonata unseen-song evaluation.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--control-timestep", type=float, default=0.05)
    parser.add_argument("--manifest-name", default="external_midi_manifest")
    parser.add_argument("--split-name", default="external_midi_splits")
    parser.add_argument("--summary-name", default="external_midi_summary.json")
    parser.add_argument("--split", default="test")
    parser.add_argument("--benchmark-name", default="external_midi_test")
    parser.add_argument("--recursive", action="store_true", default=True)
    parser.add_argument("--no-recursive", dest="recursive", action="store_false")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logger = configure_logging(args.log_level)
    outputs = index_external_midi_dataset(
        config={
            "dataset_root": str(Path(args.dataset_root).resolve()),
            "output_root": str(Path(args.output_root).resolve()),
            "control_timestep": float(args.control_timestep),
            "manifest_name": args.manifest_name,
            "split_name": args.split_name,
            "summary_name": args.summary_name,
            "split": args.split,
            "benchmark_name": args.benchmark_name,
            "recursive": bool(args.recursive),
            "force": bool(args.force),
        },
        logger=logger,
    )
    logger.info("Prepared external MIDI manifest at %s", outputs["manifest_base"])


if __name__ == "__main__":
    main()
