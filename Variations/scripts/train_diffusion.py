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
    parser = argparse.ArgumentParser(description="Train Variations conditional diffusion.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    from variations.diffusion.trainer import run_training

    config = load_config(args.config)
    if args.no_wandb:
        config.setdefault("wandb", {})["enabled"] = False
    result = run_training(config)
    print(f"Run root: {result['run_root']}")
    print(f"Best checkpoint: {result['best_checkpoint']}")


if __name__ == "__main__":
    main()
