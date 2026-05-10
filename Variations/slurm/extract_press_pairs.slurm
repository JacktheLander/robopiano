#!/bin/bash
#SBATCH -J variations_extract
#SBATCH -p cmp
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=0-08:00:00
#SBATCH --chdir=/WAVE/projects/ECEN-524-Wi26/robopiano
#SBATCH -o /WAVE/datasets/ccoelho_lab-jlanders/variations_extract_%j.out
#SBATCH -e /WAVE/datasets/ccoelho_lab-jlanders/variations_extract_%j.err

set -eo pipefail

source ~/.bashrc
conda activate sonata
set -u

export RP1M_300_ROOT=${RP1M_300_ROOT:-/WAVE/users/unix/jlanders/rp1m_300/rp1m_repertoire.zarr}
export RP1M_ROOT=${RP1M_ROOT:-$RP1M_300_ROOT}
export RUN_NAME=${RUN_NAME:-variations_extract_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}
export OUTPUT_ROOT=${OUTPUT_ROOT:-/WAVE/datasets/ccoelho_lab-jlanders/$RUN_NAME}
export VARIATIONS_OUTPUT_ROOT=${VARIATIONS_OUTPUT_ROOT:-$OUTPUT_ROOT/variations}

mkdir -p "$OUTPUT_ROOT" "$VARIATIONS_OUTPUT_ROOT/logs"
python Variations/scripts/extract_press_pairs.py --config Variations/configs/extraction/medium.yaml \
  2>&1 | tee "$VARIATIONS_OUTPUT_ROOT/logs/extract_press_pairs_${SLURM_JOB_ID:-manual}.log"
