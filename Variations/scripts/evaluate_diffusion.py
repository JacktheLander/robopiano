from __future__ import annotations

import argparse
import sys
from pathlib import Path

VARIATIONS_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = VARIATIONS_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from variations.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Variations diffusion checkpoint on val songs.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    from variations.diffusion.trainer import evaluate_checkpoint

    config = load_config(args.config)
    result = evaluate_checkpoint(config, args.checkpoint)
    print(result["metrics"])
    print(f"Saved samples: {result['samples']}")


if __name__ == "__main__":
    main()
