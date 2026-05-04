#!/bin/bash
# train_transformer_stage2_outputs_run4.sbatch
#
# Slurm: Stage 2 planner — MLP + remap, dynamics in history + low CE weight, no dynamics in plan embedding.
# Default: medium_remap_mlp_dyn_hist_lowloss_no_plan_embed.yaml → transformer2_mlp_dynhist_lowloss/
# Override STAGE2_STAGE2_OUT_SUBDIR or STAGE2_TRANSFORMER_CONFIG if needed.
#
# ------------------------------------------------------------------------------
# HOW TO RUN (recommended: tmux so SSH drops do not kill the shell)
#
# Option A — helper attaches tmux and prints the sbatch command:
#   bash "${SONATA_ROOT}/slurm/sonata_tmux_train_transformer_stage2.sh"
#
# Option B — direct submit (inherits run root via env):
#   export SONATA_ROOT=/WAVE/projects/ECEN-524-Wi26/robopiano/Sonata
#   export STAGE2_OUTPUT_ROOT=/WAVE/datasets/ccoelho_lab-jlanders/outputs_run4
#   sbatch --export=ALL "${SONATA_ROOT}/slurm/train_transformer_stage2_outputs_run4.sbatch"
#
# W&B is ON (see STAGE2_TRANSFORMER_CONFIG). Local cache under ${STAGE2_OUTPUT_ROOT}/wandb.
#
# HOW TO MONITOR
#   squeue -u "$USER"
#   tail -f "${STAGE2_OUTPUT_ROOT}/logs/slurm-stage2-%x-<jobid>.out"
#
# ------------------------------------------------------------------------------
# Slurm — adjust partition/gpu type for your cluster.
#SBATCH --job-name=sonata-s2-dynhist005
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
# Logs under the run tree (does not touch primitives/ or data/)
#SBATCH --output=/WAVE/datasets/ccoelho_lab-jlanders/outputs_run4/logs/slurm-stage2-%x-%j.out
#SBATCH --error=/WAVE/datasets/ccoelho_lab-jlanders/outputs_run4/logs/slurm-stage2-%x-%j.err
#SBATCH --export=ALL
#
set -euo pipefail

: "${SONATA_ROOT:=/WAVE/projects/ECEN-524-Wi26/robopiano/Sonata}"
STAGE2_OUTPUT_DEFAULT="/WAVE/datasets/ccoelho_lab-jlanders/outputs_run4"
export STAGE2_OUTPUT_ROOT="${STAGE2_OUTPUT_ROOT:-${STAGE2_OUTPUT_DEFAULT}}"

: "${STAGE2_STAGE2_OUT_SUBDIR:=transformer2_mlp_dynhist_lowloss}"
: "${STAGE2_TRANSFORMER_CONFIG:=${SONATA_ROOT}/configs/transformer/medium_remap_mlp_dyn_hist_lowloss_no_plan_embed.yaml}"

echo "SONATA_ROOT=${SONATA_ROOT}" >&2
echo "STAGE2_OUTPUT_ROOT=${STAGE2_OUTPUT_ROOT}" >&2
echo "STAGE2_STAGE2_OUT_SUBDIR=${STAGE2_STAGE2_OUT_SUBDIR}" >&2
echo "STAGE2_TRANSFORMER_CONFIG=${STAGE2_TRANSFORMER_CONFIG}" >&2

mkdir -p "${STAGE2_OUTPUT_ROOT}/logs" "${STAGE2_OUTPUT_ROOT}/${STAGE2_STAGE2_OUT_SUBDIR}" "${STAGE2_OUTPUT_ROOT}/wandb"

export PYTHONPATH="${SONATA_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

# W&B: keep run metadata next to Stage 1 (override: export WANDB_DIR=... before sbatch)
export WANDB_DIR="${WANDB_DIR:-${STAGE2_OUTPUT_ROOT}/wandb}"

# --- Conda (same discovery pattern as Stage 1 batch script) ---
_activated_conda=0
if [[ -n "${CONDA_SH:-}" && -f "${CONDA_SH}" ]]; then
  # shellcheck source=/dev/null
  source "${CONDA_SH}"
  _activated_conda=1
elif type module &>/dev/null && module load Anaconda3 2>/dev/null; then
  _activated_conda=1
elif [[ -f "${HOME}/.bashrc" ]]; then
  # shellcheck source=/dev/null
  source "${HOME}/.bashrc"
  _activated_conda=1
else
  for _condash in \
    "/WAVE/apps/x86_64/packages/Anaconda3/2025.12-2/app/etc/profile.d/conda.sh" \
    "/WAVE/apps/x86_64/packages/Anaconda3/2025.06-1/app/etc/profile.d/conda.sh" \
    "${HOME}/miniconda3/etc/profile.d/conda.sh" \
    "${HOME}/anaconda3/etc/profile.d/conda.sh"
  do
    if [[ -f "$_condash" ]]; then
      # shellcheck source=/dev/null
      source "$_condash"
      _activated_conda=1
      break
    fi
  done
fi
if [[ "${_activated_conda}" -eq 0 ]]; then
  echo "Could not initialize conda (set CONDA_SH or fix module load)." >&2
  exit 1
fi
if ! command -v conda &>/dev/null; then
  echo "conda not on PATH after init." >&2
  exit 1
fi

conda activate sonata

PY="${CONDA_PREFIX}/bin/python"
echo "Conda env: ${CONDA_PREFIX}" >&2
echo "Python: $($PY -V) (${PY})" >&2

if [[ ! -d "${STAGE2_OUTPUT_ROOT}/primitives/tokens" ]]; then
  echo "Missing primitive tokens at ${STAGE2_OUTPUT_ROOT}/primitives/tokens — run Stage 1 first." >&2
  exit 1
fi

cd "${SONATA_ROOT}"

# resume=false (default): new timestamped Sonata run; never overwrites primitives/ or data/.
"$PY" scripts/train_transformer.py \
  --config "${STAGE2_TRANSFORMER_CONFIG}" \
  --primitive-root "${STAGE2_OUTPUT_ROOT}/primitives" \
  --output-root "${STAGE2_OUTPUT_ROOT}/${STAGE2_STAGE2_OUT_SUBDIR}" \
  --log-level INFO

echo "Stage 2 finished. Planner runs under ${STAGE2_OUTPUT_ROOT}/${STAGE2_STAGE2_OUT_SUBDIR}/" >&2
