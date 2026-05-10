from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

VARIATIONS_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = VARIATIONS_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from variations.utils.io import load_csv, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a Variations extraction dataset.")
    parser.add_argument("--extraction-root", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = Path(args.extraction_root)
    rows = load_csv(root / "manifest.csv")
    total_rows = 0
    total_candidates = 0
    duplicates = 0
    key_counts = []
    for row in rows:
        total_candidates += int(float(row.get("candidates_seen") or 0))
        duplicates += int(float(row.get("goal_duplicates_skipped") or 0))
        path = root / row["path"]
        if not path.exists():
            continue
        data = np.load(path, allow_pickle=False)
        targets = data["target_keys"]
        total_rows += int(targets.shape[0])
        if targets.size:
            key_counts.extend(targets.sum(axis=1).astype(int).tolist())
    summary = {
        "extraction_root": str(root),
        "num_song_files": len(rows),
        "total_candidates_seen": total_candidates,
        "total_rows_accepted": total_rows,
        "total_goal_duplicates_skipped": duplicates,
        "mean_keys_pressed": float(np.mean(key_counts)) if key_counts else 0.0,
        "max_keys_pressed": int(np.max(key_counts)) if key_counts else 0,
    }
    print(summary)
    output = Path(args.output) if args.output else root / "dataset_summary.json"
    save_json(output, summary)
    print(f"Saved summary: {output}")


if __name__ == "__main__":
    main()

