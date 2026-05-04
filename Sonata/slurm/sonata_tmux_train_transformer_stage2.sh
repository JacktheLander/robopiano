#!/usr/bin/env bash
# sonata_tmux_train_transformer_stage2.sh
# SSH-friendly helper: start or attach a named tmux session; print the exact Slurm
# submission exports for Stage 2 medium planner training with outputs_run4 paths.
#
# Usage:
#   bash "${SONATA_ROOT}/slurm/sonata_tmux_train_transformer_stage2.sh"
set -euo pipefail

SESSION_NAME="${SONATA_TMUX_SESSION:-sonata-stage2-planner}"
: "${SONATA_ROOT:=/WAVE/projects/ECEN-524-Wi26/robopiano/Sonata}"
SBATCH_PATH="${SONATA_ROOT}/slurm/train_transformer_stage2_outputs_run4.sbatch"
: "${STAGE2_OUTPUT_ROOT:=/WAVE/datasets/ccoelho_lab-jlanders/outputs_run4}"
LOG_DIR="${STAGE2_OUTPUT_ROOT}/logs"

mkdir -p "${LOG_DIR}"

echo "tmux session: ${SESSION_NAME}"
echo "SONATA_ROOT:          ${SONATA_ROOT}"
echo "STAGE2_OUTPUT_ROOT:   ${STAGE2_OUTPUT_ROOT}   (primitives read; transformer2_mlp_dynhist_lowloss/ + wandb/ by default; override STAGE2_STAGE2_OUT_SUBDIR / STAGE2_TRANSFORMER_CONFIG if needed)"
echo "Logs dir:             ${LOG_DIR}"
echo ""
echo "Submit so the job inherits this run root (required for correct on-disk paths):"
echo "  export SONATA_ROOT=\"${SONATA_ROOT}\""
echo "  export STAGE2_OUTPUT_ROOT=\"${STAGE2_OUTPUT_ROOT}\""
echo "  sbatch --export=ALL \"${SBATCH_PATH}\""
echo ""

# sbatch intentionally not invoked here (same behavior as primitives helper).
exec tmux new-session -A -s "${SESSION_NAME}" -c "${SONATA_ROOT}"
