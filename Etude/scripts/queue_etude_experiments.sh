#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON:-python.exe}"
MODE="dry-run"
if [[ "${1:-}" == "--submit" ]]; then
  MODE="submit"
elif [[ "${1:-}" == "--dry-run" || -z "${1:-}" ]]; then
  MODE="dry-run"
else
  echo "Usage: bash scripts/queue_etude_experiments.sh [--dry-run|--submit]" >&2
  exit 1
fi

"$PYTHON_BIN" - "$MODE" <<'PY'
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path.cwd()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from etude.experiments import build_execution_plan, load_experiment_config

mode = sys.argv[1]
for config_path in sorted((ROOT / "configs" / "experiments").glob("*.yaml")):
    config = load_experiment_config(config_path)
    plan = build_execution_plan(config, config_path=config_path)
    script = plan.local_batch_script
    if not script:
        continue

    pretty_script = script.replace("\\", "/")
    pretty_config = str(config_path.relative_to(ROOT)).replace("\\", "/")
    if mode == "dry-run":
        print(f"echo {pretty_script} {pretty_config}")
        continue

    windows_script = str((ROOT / script).resolve())
    windows_config = str(config_path.resolve())
    subprocess.run(["cmd.exe", "/c", windows_script, windows_config], cwd=ROOT, check=True)
PY
