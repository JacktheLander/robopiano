from __future__ import annotations

import argparse
import sys
from pathlib import Path

VARIATIONS_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = VARIATIONS_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from variations.data.dataset import build_splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Build song-level train/val splits for Variations.")
    parser.add_argument("--extraction-root", required=True)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-pairs-per-split", type=int, default=1000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    path = build_splits(
        args.extraction_root,
        val_fraction=args.val_fraction,
        seed=args.seed,
        min_pairs_per_split=args.min_pairs_per_split,
        force=args.force,
    )
    print(f"Saved split index: {path}")


if __name__ == "__main__":
    main()

