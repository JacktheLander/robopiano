from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from robodiffusion.config import load_stage_config, resolve_path
from robodiffusion.data.windows import build_window_cache


def main() -> None:
    profile = sys.argv[1] if len(sys.argv) > 1 else "debug"
    config = load_stage_config("data", profile=profile)
    base_dir = Path(config["config_path"]).resolve().parent
    config["data_manifest_path"] = str(resolve_path(config["data_manifest_path"], base_dir=base_dir))
    config["output_root"] = str(resolve_path(config["output_root"], base_dir=ROOT))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    result = build_window_cache(config, logger=logging.getLogger("robodiffusion.preprocess"))
    print(result["metadata_path"])


if __name__ == "__main__":
    main()
