from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sonata.config import deep_update, load_stage_config, resolve_path
from sonata.data.indexer import scan_dataset
from sonata.data.loading import load_episode_record, load_manifest
from sonata.primitives.discovery import _apply_stage1_defaults
from sonata.primitives.segmenters import (
    aggregate_budget_metrics,
    budget_segment_candidates,
    build_segmenter,
    estimate_segment_storage_bytes,
    slice_segment_arrays,
    _load_or_infer_score_events,
)
from sonata.utils.io import write_json
from sonata.utils.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dry-run Sonata Stage 1 segment budget without writing slim cache.")
    parser.add_argument("--profile", default="prepress_wrist_relative_medium")
    parser.add_argument("--config", default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--write-json", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def load_audit_config(*, profile: str, config_path: str | None, output_root: str | None = None) -> dict[str, Any]:
    config = load_stage_config("primitive", profile=profile, config_path=config_path)
    config = _apply_stage1_defaults(config)
    config_dir = resolve_path(config["config_path"]).parent
    if output_root is not None:
        config["output_root"] = output_root
    config["output_root"] = str(resolve_path(config["output_root"], config_dir))
    config["data_output_root"] = str(resolve_path(config["data_output_root"], config_dir))
    data_profile = str(config.get("data_profile", profile))
    data_config = load_stage_config("data", profile=data_profile)
    data_config_dir = resolve_path(data_config["config_path"]).parent
    data_config["dataset_root"] = str(resolve_path(data_config["dataset_root"], data_config_dir))
    data_config["output_root"] = str(config["data_output_root"])
    data_config["note_search_roots"] = [str(resolve_path(path, data_config_dir)) for path in data_config.get("note_search_roots", [])]
    config["data_config"] = deep_update(data_config, {"force": False})
    return config


def audit_segment_budget(config: dict[str, Any], *, max_episodes: int | None = None, logger=None) -> dict[str, Any]:
    data_output = Path(config["data_output_root"]).resolve()
    manifest_base = data_output / str(config["data_manifest_name"])
    if not manifest_base.with_suffix(".csv").exists() and not manifest_base.with_suffix(".parquet").exists():
        scan_dataset(config=config["data_config"], logger=logger)
    manifest_df = load_manifest(manifest_base)
    if max_episodes is not None:
        manifest_df = manifest_df.head(int(max_episodes)).reset_index(drop=True)

    segmenter = build_segmenter(config)
    accepted_rows: list[dict[str, Any]] = []
    aggregate_stats: dict[str, Any] = {
        "proposed_segments": 0,
        "accepted_segments_before_budget": 0,
        "dropped_by_budget": 0,
    }
    storage_bytes = 0
    for manifest_row in manifest_df.itertuples(index=False):
        episode = load_episode_record(manifest_row._asdict())
        score_events = _load_or_infer_score_events(episode=episode, config=config)
        candidates = segmenter.segment(episode, score_events)
        accepted, stats = budget_segment_candidates(
            candidates,
            config,
            song_id=str(manifest_row.song_id),
            episode_id=str(manifest_row.episode_id),
        )
        aggregate_stats["proposed_segments"] += int(stats.get("proposed_segments", 0))
        aggregate_stats["accepted_segments_before_budget"] += int(stats.get("accepted_segments_before_budget", 0))
        aggregate_stats["dropped_by_budget"] += int(stats.get("dropped_by_budget", 0))
        for candidate in accepted:
            duration = max(int(candidate.end_step) - int(candidate.onset_step), 1)
            accepted_rows.append(
                {
                    "song_id": str(manifest_row.song_id),
                    "episode_id": str(manifest_row.episode_id),
                    "target_onset_step": int(candidate.target_onset_step),
                    "duration_steps": int(duration),
                    "chord_size": int(candidate.chord_size),
                    "coarse_family": str(candidate.coarse_family),
                    "causal_press_score": float(candidate.causal_press_score),
                }
            )
            try:
                storage_bytes += estimate_segment_storage_bytes(slice_segment_arrays(episode, candidate.onset_step, candidate.end_step))
            except Exception:
                pass
    accepted_df = pd.DataFrame(accepted_rows)
    metrics = aggregate_budget_metrics(accepted_df, config, aggregate_stats)
    scale = float(len(load_manifest(manifest_base))) / max(float(len(manifest_df)), 1.0)
    metrics.update(
        {
            "episodes_audited": int(len(manifest_df)),
            "estimated_full_rp1m300_segment_count": int(round(metrics["accepted_segments_after_budget"] * scale)),
            "estimated_storage_bytes": int(round(storage_bytes * scale)),
            "estimated_runtime_relative_to_main": float(metrics["accepted_segments_after_budget"] * scale / 700000.0),
        }
    )
    return metrics


def main() -> None:
    args = build_parser().parse_args()
    logger = configure_logging(args.log_level)
    config = load_audit_config(profile=args.profile, config_path=args.config, output_root=args.output_root)
    metrics = audit_segment_budget(config, max_episodes=args.max_episodes, logger=logger)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if args.write_json:
        output_root = Path(config["output_root"]).resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        write_json(metrics, output_root / "stage1_segment_budget_audit.json")


if __name__ == "__main__":
    main()
