#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sonata.config import load_yaml, resolve_path
from sonata.evaluation.primitive_online_eval import evaluate_primitives_online
from sonata.primitives.discovery import run_primitive_pipeline
from sonata.utils.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Bounded closed-loop: Stage 1 + capped primitive online eval.")
    p.add_argument("--primitive-config", type=Path, required=True)
    p.add_argument("--eval-config", type=Path, required=True)
    p.add_argument("--profile", default="medium", help="Primitive profile name (unused if --primitive-config is set).")
    p.add_argument("--data-profile", default="medium", help="Data config profile for dataset manifest.")
    p.add_argument("--max-rounds", type=int, default=2)
    p.add_argument("--log-level", default="INFO")
    return p


def _quality_score(row: pd.Series) -> float:
    f1 = float(pd.to_numeric(row.get("intended_onset_f1", 0.0), errors="coerce") or 0.0)
    missed = float(pd.to_numeric(row.get("intended_missed_key_events", 0), errors="coerce") or 0.0)
    fp = float(pd.to_numeric(row.get("false_positive_key_events", 0), errors="coerce") or 0.0)
    mse = float(pd.to_numeric(row.get("piano_state_mse", 0.0), errors="coerce") or 0.0)
    return float(f1 - 0.08 * missed - 0.05 * fp - 0.02 * min(mse, 50.0))


def main() -> None:
    args = build_parser().parse_args()
    logger = configure_logging(args.log_level)
    from sonata.config import deep_update, load_stage_config

    prim_cfg = load_stage_config("primitive", profile=args.profile, config_path=args.primitive_config.resolve())
    cfg_dir = resolve_path(prim_cfg["config_path"]).parent
    prim_cfg["output_root"] = str(resolve_path(prim_cfg["output_root"], cfg_dir))
    prim_cfg["data_output_root"] = str(resolve_path(prim_cfg["data_output_root"], cfg_dir))
    data_cfg = load_stage_config("data", profile=args.data_profile)
    ddir = resolve_path(data_cfg["config_path"]).parent
    data_cfg["dataset_root"] = str(resolve_path(data_cfg["dataset_root"], ddir))
    data_cfg["output_root"] = str(resolve_path(data_cfg["output_root"], ddir))
    data_cfg["note_search_roots"] = [str(resolve_path(x, ddir)) for x in data_cfg.get("note_search_roots", [])]
    prim_cfg["data_config"] = deep_update(data_cfg, {"force": bool(prim_cfg.get("force", False))})

    logger.info("Round 1: running Stage 1 primitive pipeline")
    run_primitive_pipeline(config=prim_cfg, logger=logger)

    out_root = Path(prim_cfg["output_root"]).resolve() / "closed_loop"
    out_root.mkdir(parents=True, exist_ok=True)

    eval_path = args.eval_config.resolve()
    eval_cfg = load_yaml(eval_path)
    eval_cfg["primitive_root"] = str(Path(prim_cfg["output_root"]).resolve())
    eval_cfg["output_root"] = str(out_root / "eval_round_1")
    eval_cfg.setdefault("sampling", {})
    eval_cfg["sampling"].setdefault("instances_per_primitive", 2)
    eval_cfg["sampling"].setdefault("max_instances", 128)
    eval_cfg["sampling"].setdefault("max_instances_total", 128)

    for round_idx in range(1, int(args.max_rounds) + 1):
        eval_cfg["output_root"] = str(out_root / f"eval_round_{round_idx}")
        payload = evaluate_primitives_online(config=eval_cfg, logger=logger)
        metrics_path = Path(payload["output_root"]) / "primitive_instance_metrics.csv"
        summary_path = Path(payload["output_root"]) / "primitive_summary_metrics.csv"
        if not metrics_path.exists():
            logger.warning("Missing instance metrics at %s", metrics_path)
            continue
        mdf = pd.read_csv(metrics_path)
        usable = mdf.loc[mdf["status"].astype(str).isin({"completed", "terminated_early"})].copy()
        advisory: list[dict[str, object]] = []
        if not usable.empty and "primitive_id" in usable.columns:
            for pid, grp in usable.groupby("primitive_id", sort=True):
                row_mean = grp.mean(numeric_only=True)
                score = _quality_score(row_mean)
                reason = ""
                suggestion = ""
                if score < 0.35:
                    reason = "low_intended_f1_or_high_errors"
                    suggestion = "prune_or_merge"
                elif float(row_mean.get("piano_state_mse", 0.0) or 0.0) > 2.0:
                    reason = "high_piano_mse"
                    suggestion = "review_gmr_prior"
                advisory.append(
                    {
                        "primitive_id": str(pid),
                        "primitive_quality_score": score,
                        "primitive_prune_reason": reason,
                        "primitive_merge_suggestion": suggestion,
                        "n_instances": int(len(grp)),
                    }
                )
        adv_path = out_root / f"advisory_round_{round_idx}.json"
        adv_path.write_text(json.dumps(advisory, indent=2), encoding="utf-8")
        logger.info("Wrote advisory %s (rows=%d)", adv_path, len(advisory))
        if round_idx < int(args.max_rounds):
            logger.info("Additional rounds reuse the same primitive_root; re-run Stage 1 manually if you change configs.")

    logger.info("Closed-loop artifacts under %s", out_root)


if __name__ == "__main__":
    main()
