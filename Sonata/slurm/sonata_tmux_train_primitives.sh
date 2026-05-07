#!/usr/bin/env bash
# sonata_tmux_train_primitives.sh
# SSH-friendly helper: start or attach a named tmux session; print the exact Slurm
# submission command. Does NOT run sbatch unless you uncomment the line at the end.
set -euo pipefail

SESSION_NAME="${SONATA_TMUX_SESSION:-sonata-primitives-stage1}"
# Match train_primitives_cmp_stage1.sbatch defaults; override for projects2/another clone
: "${SONATA_ROOT:=/WAVE/projects/ECEN-524-Wi26/robopiano/Sonata}"
SBATCH_PATH="${SONATA_ROOT}/slurm/train_primitives_cmp_stage1.sbatch"
: "${STAGE1_OUTPUT_ROOT:=/WAVE/datasets/ccoelho_lab-jlanders/outputs_run5}"
STAGE1_OUT="${STAGE1_OUTPUT_ROOT}"
LOG_DIR="${STAGE1_OUT}/logs"

mkdir -p "${LOG_DIR}"

echo "tmux session: ${SESSION_NAME}"
echo "SONATA_ROOT:          ${SONATA_ROOT}"
echo "STAGE1_OUTPUT_ROOT:  ${STAGE1_OUT}   (primitives+data+slim+W&B go here)"
echo "Logs dir:             ${LOG_DIR}"
echo ""
echo "Submit so the job inherits this run root (required for the correct on-disk path):"
echo "  export SONATA_ROOT=\"${SONATA_ROOT}\""
echo "  export STAGE1_OUTPUT_ROOT=\"${STAGE1_OUT}\""
echo "  sbatch --export=ALL \"${SBATCH_PATH}\""
echo "Optional:  export RP1M_300_ROOT=...  sbatch --export=ALL --cpus-per-task=24 \"${SBATCH_PATH}\""
echo "Post-train sim eval:  export RUN_PRIMITIVE_ONLINE_EVAL=1  before sbatch"
echo ""

# -- Auto-submit is disabled by design. To submit from this helper, uncomment:
# sbatch "${SBATCH_PATH}"

exec tmux new-session -A -s "${SESSION_NAME}" -c "${SONATA_ROOT}"
