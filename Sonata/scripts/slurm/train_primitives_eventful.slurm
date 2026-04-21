#!/bin/bash
#SBATCH --job-name=sonata-s1-eventful
#SBATCH --partition=cmp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=/WAVE/datasets/ccoelho_lab-jlanders/sonata_logs/%x-%j.out

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/WAVE/projects/ECEN-524-Wi26/robopiano}"
SONATA_ROOT="${SONATA_ROOT:-${PROJECT_ROOT}/Sonata}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-sonata}"
PERSIST_ROOT="${PERSIST_ROOT:-/WAVE/datasets/ccoelho_lab-jlanders}"
PRIMITIVE_PROFILE="${PRIMITIVE_PROFILE:-medium_eventful}"
PRIMITIVE_CONFIG="${PRIMITIVE_CONFIG:-${SONATA_ROOT}/configs/primitive/medium_eventful.yaml}"
PRIMITIVE_OUTPUT_ROOT="${PRIMITIVE_OUTPUT_ROOT:-${PERSIST_ROOT}/sonata/outputs/primitives/medium_eventful}"
POST_EVAL_CONFIG="${POST_EVAL_CONFIG:-${SONATA_ROOT}/configs/evaluation/primitives_online_maestro.yaml}"
POST_EVAL_OUTPUT_ROOT="${POST_EVAL_OUTPUT_ROOT:-${PERSIST_ROOT}/sonata/outputs/evaluation/primitives_online/medium_eventful_maestro}"
ROBOPIANIST_ROOT="${ROBOPIANIST_ROOT:-${PROJECT_ROOT}/robopianist}"
RP1M_300_ROOT="${RP1M_300_ROOT:?Set RP1M_300_ROOT to the shared RP1M zarr path before submission.}"
MAESTRO_MIDI_ROOT="${MAESTRO_MIDI_ROOT:?Set MAESTRO_MIDI_ROOT to the MAESTRO MIDI root before submission.}"

mkdir -p "${PERSIST_ROOT}/sonata_logs"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # Common fallback on shared clusters.
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
else
  echo "Unable to locate conda initialization script." >&2
  exit 1
fi

conda activate "${CONDA_ENV_NAME}"

cd "${SONATA_ROOT}"

echo "Running Stage 1 eventful primitive learning"
echo "  SONATA_ROOT=${SONATA_ROOT}"
echo "  PRIMITIVE_OUTPUT_ROOT=${PRIMITIVE_OUTPUT_ROOT}"
echo "  POST_EVAL_OUTPUT_ROOT=${POST_EVAL_OUTPUT_ROOT}"
echo "  ROBOPIANIST_ROOT=${ROBOPIANIST_ROOT}"
echo "  RP1M_300_ROOT=${RP1M_300_ROOT}"
echo "  MAESTRO_MIDI_ROOT=${MAESTRO_MIDI_ROOT}"

python scripts/train_primitives.py \
  --profile "${PRIMITIVE_PROFILE}" \
  --config "${PRIMITIVE_CONFIG}" \
  --output-root "${PRIMITIVE_OUTPUT_ROOT}" \
  --post-eval-config "${POST_EVAL_CONFIG}" \
  --post-eval-output-root "${POST_EVAL_OUTPUT_ROOT}" \
  --run-post-eval \
  --robopianist-root "${ROBOPIANIST_ROOT}" \
  --log-level INFO
