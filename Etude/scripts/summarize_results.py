from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from etude.experiments import summarize_result_tree, write_summary_csv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Etude experiment result files into one summary CSV.")
    parser.add_argument("--root", required=True, help="Result root to scan recursively.")
    parser.add_argument("--out", required=True, help="Destination CSV path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = summarize_result_tree(args.root)
    destination = write_summary_csv(rows, args.out)
    print(
        json.dumps(
            {
                "root": str(Path(args.root).resolve()),
                "out": str(destination.resolve()),
                "rows": len(rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
