#!/bin/bash
#SBATCH -J variations_train
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --chdir=/WAVE/projects/ECEN-524-Wi26/robopiano
#SBATCH -o /WAVE/datasets/ccoelho_lab-jlanders/Variations/slurm_logs/variations_train_%j.out
#SBATCH -e /WAVE/datasets/ccoelho_lab-jlanders/Variations/slurm_logs/variations_train_%j.err

set -eo pipefail

source ~/.bashrc
conda activate sonata
set -u
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

export RUN_NAME=${RUN_NAME:-variations_train_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}
export OUTPUT_ROOT=${OUTPUT_ROOT:-/WAVE/datasets/ccoelho_lab-jlanders/Variations/$RUN_NAME}
export VARIATIONS_OUTPUT_ROOT=${VARIATIONS_OUTPUT_ROOT:-$OUTPUT_ROOT/variations}
export VARIATIONS_TRAIN_CONFIG=${VARIATIONS_TRAIN_CONFIG:-Variations/configs/diffusion/medium.yaml}

mkdir -p "$OUTPUT_ROOT" "$VARIATIONS_OUTPUT_ROOT/logs"
python Variations/scripts/train_diffusion.py --config "$VARIATIONS_TRAIN_CONFIG" \
  2>&1 | tee "$VARIATIONS_OUTPUT_ROOT/logs/train_diffusion_${SLURM_JOB_ID:-manual}.log"
