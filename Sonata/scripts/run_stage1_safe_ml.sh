#!/usr/bin/env bash
# Local or interactive node: run safe Stage 1 (use Slurm script for full RP1M 300 job).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export RP1M_300_ROOT="${RP1M_300_ROOT:-/WAVE/users/unix/jlanders/rp1m_300/rp1m_repertoire.zarr}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sonata

RUN_POST_EVAL="${RUN_POST_EVAL:-0}"
POST_FLAGS=""
if [[ "$RUN_POST_EVAL" == "1" ]]; then
  POST_FLAGS="--run-post-eval --post-eval-config configs/evaluation/primitives_online_safe.yaml"
fi

python scripts/train_primitives.py \
  --profile medium \
  --config configs/primitive/medium_safe_ml.yaml \
  ${POST_FLAGS}

echo "Done. Outputs under configs path resolved from medium_safe_ml.yaml (default ../../outputs/primitives/medium_safe_ml)."
