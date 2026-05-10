from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

VARIATIONS_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = VARIATIONS_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from variations.data.press_extractor import goal_fingerprint
from variations.utils.io import load_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify global target_keys uniqueness in an extraction root.")
    parser.add_argument("--extraction-root", required=True)
    args = parser.parse_args()
    root = Path(args.extraction_root)
    seen = {}
    duplicates = []
    for row in load_csv(root / "manifest.csv"):
        path = root / row["path"]
        if not path.exists():
            continue
        data = np.load(path, allow_pickle=False)
        for idx, target in enumerate(data["target_keys"]):
            fp = goal_fingerprint(target)
            here = f"{row['song_id']}:{idx}"
            if fp in seen:
                duplicates.append((seen[fp], here))
            else:
                seen[fp] = here
    if duplicates:
        for first, second in duplicates[:20]:
            print(f"duplicate {first} {second}")
        raise SystemExit(f"Found {len(duplicates)} duplicate goal fingerprints")
    print(f"OK: {len(seen)} unique goal fingerprints")


if __name__ == "__main__":
    main()

