from __future__ import annotations

import argparse
import sys
from pathlib import Path

VARIATIONS_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = VARIATIONS_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from variations.data.dataset import compute_norm_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute train-only hand_state normalization stats.")
    parser.add_argument("--extraction-root", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    path = compute_norm_stats(args.extraction_root, force=args.force)
    print(f"Saved norm stats: {path}")


if __name__ == "__main__":
    main()

