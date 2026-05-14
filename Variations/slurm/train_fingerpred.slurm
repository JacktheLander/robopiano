#!/bin/bash
#SBATCH -J variations_fingerpred
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --chdir=/WAVE/projects/ECEN-524-Wi26/robopiano
#SBATCH -o /WAVE/datasets/ccoelho_lab-jlanders/Variations/slurm_logs/variations_fingerpred_%j.out
#SBATCH -e /WAVE/datasets/ccoelho_lab-jlanders/Variations/slurm_logs/variations_fingerpred_%j.err

set -eo pipefail

source ~/.bashrc
conda activate sonata
set -u
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

export RUN_NAME=${RUN_NAME:-variations_fingerpred_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}
export OUTPUT_ROOT=${OUTPUT_ROOT:-/WAVE/datasets/ccoelho_lab-jlanders/Variations/$RUN_NAME}
export VARIATIONS_OUTPUT_ROOT=${VARIATIONS_OUTPUT_ROOT:-$OUTPUT_ROOT/variations}
export VARIATIONS_FINGERPRED_CONFIG=${VARIATIONS_FINGERPRED_CONFIG:-Variations/configs/fingerpred.yaml}
export VARIATIONS_FINGERPRED_EXTRA_ARGS=${VARIATIONS_FINGERPRED_EXTRA_ARGS:-}

EXTRA_ARGS=()
if [[ -n "${VARIATIONS_FINGERPRED_EXTRACTION_ROOT:-}" ]]; then
  EXTRA_ARGS+=(--extraction-root "$VARIATIONS_FINGERPRED_EXTRACTION_ROOT")
fi
if [[ -n "$VARIATIONS_FINGERPRED_EXTRA_ARGS" ]]; then
  read -r -a USER_EXTRA_ARGS <<< "$VARIATIONS_FINGERPRED_EXTRA_ARGS"
  EXTRA_ARGS+=("${USER_EXTRA_ARGS[@]}")
fi

mkdir -p "$OUTPUT_ROOT" "$VARIATIONS_OUTPUT_ROOT/logs"
python Variations/scripts/train_fingerpred.py --config "$VARIATIONS_FINGERPRED_CONFIG" "${EXTRA_ARGS[@]}" \
  2>&1 | tee "$VARIATIONS_OUTPUT_ROOT/logs/train_fingerpred_${SLURM_JOB_ID:-manual}.log"
