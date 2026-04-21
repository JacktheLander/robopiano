#!/bin/bash
#SBATCH --job-name=sonata_s1_safe
#SBATCH --partition=cmp
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --output=/WAVE/datasets/ccoelho_lab-jlanders/outputs_run3/stage1_safe_ml/logs/%x-%j.out
#SBATCH --error=/WAVE/datasets/ccoelho_lab-jlanders/outputs_run3/stage1_safe_ml/logs/%x-%j.err
#
# Submit:  sbatch /path/to/Sonata/scripts/slurm_stage1_safe_ml.sh
# Monitor: squeue -u "$USER"
# Logs:    /WAVE/datasets/ccoelho_lab-jlanders/outputs_run3/stage1_safe_ml/logs/
# Resume:  keep force: false in configs/primitive/medium_safe_ml.yaml; re-submit (completed slim episodes are skipped).
#
set -euo pipefail

OUT_BASE="${OUT_BASE:-/WAVE/datasets/ccoelho_lab-jlanders/outputs_run3/stage1_safe_ml}"
mkdir -p "${OUT_BASE}/logs"

SONATA_ROOT="${SONATA_ROOT:-/WAVE/projects/ECEN-524-Wi26/robopiano/Sonata}"
cd "$SONATA_ROOT"

export RP1M_300_ROOT="${RP1M_300_ROOT:-/WAVE/users/unix/jlanders/rp1m_300/rp1m_repertoire.zarr}"
export PYTHONPATH="${SONATA_ROOT}/src:${PYTHONPATH:-}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sonata

PRIM_OUT="${OUT_BASE}/primitives"
DATA_OUT="${OUT_BASE}/data_medium"
mkdir -p "$PRIM_OUT" "$DATA_OUT"

RUN_POST_EVAL="${RUN_POST_EVAL:-0}"
POST_FLAGS=()
if [[ "$RUN_POST_EVAL" == "1" ]]; then
  POST_FLAGS=(--run-post-eval --post-eval-config configs/evaluation/primitives_online_safe.yaml --post-eval-output-root "${OUT_BASE}/post_eval")
fi

python scripts/train_primitives.py \
  --profile medium \
  --config configs/primitive/medium_safe_ml.yaml \
  --output-root "$PRIM_OUT" \
  --data-output-root "$DATA_OUT" \
  ${ROBOPIANIST_ROOT:+--robopianist-root "$ROBOPIANIST_ROOT"} \
  "${POST_FLAGS[@]}"

echo "Stage 1 finished. Primitive root: $PRIM_OUT"
