from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sonata.config import load_stage_config, resolve_path
from sonata.primitives.segmenters import migrate_existing_segment_chunks
from sonata.primitives.slim_cache import collect_slim_chunk_names, compose_segment_index, load_slim_index_table, resolve_slim_cache_paths
from sonata.utils.io import read_json, read_table, write_json, write_table
from sonata.utils.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Migrate Sonata Stage 1 raw segment chunks into the slim cache.")
    parser.add_argument("--profile", default="debug")
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--delete-raw", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_stage_config("primitive", profile=args.profile, config_path=args.config)
    config_dir = resolve_path(config["config_path"]).parent
    if args.output_root is not None:
        config["output_root"] = args.output_root
    config["output_root"] = str(resolve_path(config["output_root"], config_dir))
    config["write_slim_cache"] = True
    config["migrate_existing_segment_chunks"] = True
    if args.delete_raw:
        config["delete_raw_chunks_after_migration"] = True

    logger = configure_logging(args.log_level)
    primitive_root = Path(config["output_root"]).resolve()
    segments_dir = primitive_root / "segments"
    segment_table_base = segments_dir / "segment_index"
    if not segment_table_base.with_suffix(".csv").exists() and not segment_table_base.with_suffix(".parquet").exists():
        raise FileNotFoundError(f"Missing segment index table at {segment_table_base}")

    segment_df = read_table(segment_table_base)
    summary = migrate_existing_segment_chunks(
        segment_df=segment_df,
        segments_dir=segments_dir,
        output_dir=primitive_root,
        config=config,
        logger=logger,
    )
    slim_paths = resolve_slim_cache_paths(primitive_root, config)
    merged_df = compose_segment_index(segment_df, load_slim_index_table(slim_paths))
    write_table(merged_df, segment_table_base)

    manifest_path = segments_dir / "segment_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    manifest.update(
        {
            "write_slim_cache": True,
            "migrated_raw_chunks": int(summary["migrated_chunks"]),
            "deleted_raw_chunks": int(summary["deleted_raw_chunks"]),
            "slim_chunk_files": collect_slim_chunk_names(slim_paths),
        }
    )
    write_json(manifest, manifest_path)
    logger.info(
        "Migrated %d raw chunks into %d slim chunks.",
        summary["migrated_chunks"],
        len(collect_slim_chunk_names(slim_paths)),
    )


if __name__ == "__main__":
    main()
