#!/bin/bash
# train_transformer_stage2_primitive_remap.sbatch
#
# Same as train_transformer_stage2_outputs_run4.sbatch but writes Planner runs under a
# dedicated folder (primitive-remap checkpoint selection + metrics defaults from medium.yaml).
#
# Primitive cache (read-only):  ${STAGE2_OUTPUT_ROOT}/primitives/
# Transformer outputs (write): ${STAGE2_OUTPUT_ROOT}/${STAGE2_STAGE2_OUT_SUBDIR}/
#
# Submit (inherit run root via env):
#   export SONATA_ROOT=/path/to/Sonata
#   export STAGE2_OUTPUT_ROOT=/path/to/outputs_run4
#   sbatch --export=ALL Sonata/slurm/train_transformer_stage2_primitive_remap.sbatch
#
# Override output subdir only:
#   export STAGE2_STAGE2_OUT_SUBDIR=transformer_primitive_remap_myrun
#   sbatch --export=ALL "${SONATA_ROOT}/slurm/train_transformer_stage2_primitive_remap.sbatch"
#
#SBATCH --job-name=sonata-s2-remap
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/WAVE/datasets/ccoelho_lab-jlanders/outputs_run4/logs/slurm-stage2-remap-%x-%j.out
#SBATCH --error=/WAVE/datasets/ccoelho_lab-jlanders/outputs_run4/logs/slurm-stage2-remap-%x-%j.err
#SBATCH --export=ALL

set -euo pipefail

: "${SONATA_ROOT:=/WAVE/projects/ECEN-524-Wi26/robopiano/Sonata}"
STAGE2_OUTPUT_DEFAULT="/WAVE/datasets/ccoelho_lab-jlanders/outputs_run4"
export STAGE2_OUTPUT_ROOT="${STAGE2_OUTPUT_ROOT:-${STAGE2_OUTPUT_DEFAULT}}"

_ts="$(date +%Y%m%d_%H%M%S)"
export STAGE2_STAGE2_OUT_SUBDIR="${STAGE2_STAGE2_OUT_SUBDIR:-transformer_primitive_remap_${_ts}}"

echo "SONATA_ROOT=${SONATA_ROOT}" >&2
echo "STAGE2_OUTPUT_ROOT=${STAGE2_OUTPUT_ROOT}" >&2
echo "STAGE2_STAGE2_OUT_SUBDIR=${STAGE2_STAGE2_OUT_SUBDIR}" >&2

mkdir -p "${STAGE2_OUTPUT_ROOT}/logs" "${STAGE2_OUTPUT_ROOT}/${STAGE2_STAGE2_OUT_SUBDIR}" "${STAGE2_OUTPUT_ROOT}/wandb"

export PYTHONPATH="${SONATA_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export WANDB_DIR="${WANDB_DIR:-${STAGE2_OUTPUT_ROOT}/wandb}"

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

"$PY" scripts/train_transformer.py \
  --profile medium \
  --primitive-root "${STAGE2_OUTPUT_ROOT}/primitives" \
  --output-root "${STAGE2_OUTPUT_ROOT}/${STAGE2_STAGE2_OUT_SUBDIR}" \
  --log-level INFO

echo "Stage 2 (primitive-remap planner) finished. Runs under ${STAGE2_OUTPUT_ROOT}/${STAGE2_STAGE2_OUT_SUBDIR}/" >&2
