from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robodiffusion.config import load_stage_config, resolve_path
from robodiffusion.training.trainer import run_training


def main() -> None:
    profile = sys.argv[1] if len(sys.argv) > 1 else "debug"
    config = load_stage_config("training", profile=profile)
    config["dataset_root"] = str(resolve_path(config["dataset_root"], base_dir=ROOT))
    config["output_root"] = str(resolve_path(config["output_root"], base_dir=ROOT))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    result = run_training(config, logger=logging.getLogger("robodiffusion.train"))
    print(result["best_checkpoint"])


if __name__ == "__main__":
    main()
