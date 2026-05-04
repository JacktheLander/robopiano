#!/usr/bin/env python3
"""Build metrics/stage2_primitive_quality.csv for Stage 2 primitive_filter.

The transformer loader expects a CSV with primitive_id and onset_f1 columns (optional
boolean key_press / target_hit). Raw primitive_library.csv has no onset F1 column, and
evaluation/primitives_online/primitive_summary_metrics.csv often omits onset_f1 rows.

This script merges Stage 1 library stats with online rollout summaries where available,
and derives a monotone onset surrogate from assignment confidence and weighted strike error
otherwise. Writes primitives/metrics/stage2_primitive_quality.csv under --primitive-root.

Example:
  python scripts/build_stage2_primitive_quality_csv.py \\
    --primitive-root /path/outputs_run4/primitives \\
    --online-summary /path/outputs_run4/evaluation/primitives_online/primitive_summary_metrics.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sonata.transformer.stage2_quality_table import DEFAULT_STAGE2_QUALITY_REL, build_stage2_quality_table_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit Stage 2 primitive_filter quality CSV under primitives/metrics/.")
    parser.add_argument("--primitive-root", type=Path, required=True)
    parser.add_argument(
        "--online-summary",
        type=Path,
        default=None,
        help="primitive_summary_metrics.csv from online rollout evaluation (optional).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: <primitive-root>/metrics/stage2_primitive_quality.csv).",
    )
    args = parser.parse_args()
    primitive_root = args.primitive_root.resolve()
    metrics_dir = primitive_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_path = (
        args.output.resolve()
        if args.output is not None
        else (metrics_dir / DEFAULT_STAGE2_QUALITY_REL.name).resolve()
    )

    summary = args.online_summary.resolve() if args.online_summary is not None else None
    df = build_stage2_quality_table_df(primitive_root=primitive_root, online_summary=summary)
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(df)} primitives).")


if __name__ == "__main__":
    main()
