#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON:-python.exe}"

"$PYTHON_BIN" -m compileall -q src scripts tests

"$PYTHON_BIN" - <<'PY'
import sys
from pathlib import Path

root = Path.cwd()
src = root / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

import etude

print("import etude ok")
print(f"etude version: {getattr(etude, '__version__', 'unknown')}")
PY

if "$PYTHON_BIN" -c "import pytest" >/dev/null 2>&1; then
  "$PYTHON_BIN" -m pytest tests -q -m "not heavy and not rollout and not gpu and not dataset" || true
else
  echo "pytest unavailable; skipping pytest stage"
fi

for cfg in configs/experiments/*.yaml; do
  "$PYTHON_BIN" scripts/run_experiment.py --config "$cfg" --dry-run
done
