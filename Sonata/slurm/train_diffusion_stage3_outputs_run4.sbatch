#!/bin/bash
# train_diffusion_stage3_outputs_run4.sbatch
#
# Slurm: Stage 3 diffusion refiner (medium profile), GPU, separate job from Stage 2.
# Reads primitives from outputs_run4; writes under outputs_run4/diffusion3/ by default.
#
# Default planner checkpoint: best val/loss among completed MLP-style runs at time of writing
# (medium-planner-remap-mlp-no-dynamics …/checkpoints/best.pt). Override:
#   export STAGE3_PLANNER_CHECKPOINT=/path/to/planner/checkpoints/best.pt
#
# Usage:
#   export SONATA_ROOT=/WAVE/projects/ECEN-524-Wi26/robopiano/Sonata
#   export STAGE3_OUTPUT_ROOT=/WAVE/datasets/ccoelho_lab-jlanders/outputs_run4
#   sbatch --export=ALL "${SONATA_ROOT}/slurm/train_diffusion_stage3_outputs_run4.sbatch"
#
# W&B: configs/diffusion/medium.yaml (online). Cache: ${STAGE3_OUTPUT_ROOT}/wandb
#
#SBATCH --job-name=sonata-s3-diffusion
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/WAVE/datasets/ccoelho_lab-jlanders/outputs_run4/logs/slurm-stage3-diffusion-%x-%j.out
#SBATCH --error=/WAVE/datasets/ccoelho_lab-jlanders/outputs_run4/logs/slurm-stage3-diffusion-%x-%j.err
#SBATCH --export=ALL

set -euo pipefail

: "${SONATA_ROOT:=/WAVE/projects/ECEN-524-Wi26/robopiano/Sonata}"
STAGE3_OUTPUT_DEFAULT="/WAVE/datasets/ccoelho_lab-jlanders/outputs_run4"
export STAGE3_OUTPUT_ROOT="${STAGE3_OUTPUT_ROOT:-${STAGE3_OUTPUT_DEFAULT}}"

# Default planner checkpoint: prefer most recent finished run if its val/loss beats prior best.
# As of dynhist-lowloss run: best val/loss 0.240 (epoch 3) vs no-dynamics 0.289 — use dynhist.
DEFAULT_PLANNER="/WAVE/datasets/ccoelho_lab-jlanders/outputs_run4/transformer2_mlp_dynhist_lowloss/Sonata-3-transformer-Sonata-3-medium-planner-remap-mlp-dynhist-lowloss-noplanemb-seed7-20260503-211143/checkpoints/best.pt"
: "${STAGE3_PLANNER_CHECKPOINT:=${DEFAULT_PLANNER}}"
: "${STAGE3_DIFFUSION_OUT_SUBDIR:=diffusion3}"

echo "SONATA_ROOT=${SONATA_ROOT}" >&2
echo "STAGE3_OUTPUT_ROOT=${STAGE3_OUTPUT_ROOT}" >&2
echo "STAGE3_PLANNER_CHECKPOINT=${STAGE3_PLANNER_CHECKPOINT}" >&2
echo "STAGE3_DIFFUSION_OUT_SUBDIR=${STAGE3_DIFFUSION_OUT_SUBDIR}" >&2

if [[ ! -f "${STAGE3_PLANNER_CHECKPOINT}" ]]; then
  echo "Planner checkpoint missing: ${STAGE3_PLANNER_CHECKPOINT}" >&2
  exit 1
fi

mkdir -p "${STAGE3_OUTPUT_ROOT}/logs" "${STAGE3_OUTPUT_ROOT}/${STAGE3_DIFFUSION_OUT_SUBDIR}" "${STAGE3_OUTPUT_ROOT}/wandb"

export PYTHONPATH="${SONATA_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export WANDB_DIR="${WANDB_DIR:-${STAGE3_OUTPUT_ROOT}/wandb}"

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

if [[ ! -d "${STAGE3_OUTPUT_ROOT}/primitives/tokens" ]]; then
  echo "Missing primitive tokens at ${STAGE3_OUTPUT_ROOT}/primitives/tokens — run Stage 1 first." >&2
  exit 1
fi

cd "${SONATA_ROOT}"

"$PY" scripts/train_diffusion.py \
  --profile medium \
  --primitive-root "${STAGE3_OUTPUT_ROOT}/primitives" \
  --planner-checkpoint "${STAGE3_PLANNER_CHECKPOINT}" \
  --output-root "${STAGE3_OUTPUT_ROOT}/${STAGE3_DIFFUSION_OUT_SUBDIR}" \
  --log-level INFO

echo "Stage 3 finished. Diffusion runs under ${STAGE3_OUTPUT_ROOT}/${STAGE3_DIFFUSION_OUT_SUBDIR}/" >&2
