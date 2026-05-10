#!/bin/bash
#SBATCH -J variations_train
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --chdir=/WAVE/projects/ECEN-524-Wi26/robopiano
#SBATCH -o /WAVE/datasets/ccoelho_lab-jlanders/variations_train_%j.out
#SBATCH -e /WAVE/datasets/ccoelho_lab-jlanders/variations_train_%j.err

set -eo pipefail

source ~/.bashrc
conda activate sonata
set -u

export RUN_NAME=${RUN_NAME:-variations_train_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}
export OUTPUT_ROOT=${OUTPUT_ROOT:-/WAVE/datasets/ccoelho_lab-jlanders/$RUN_NAME}
export VARIATIONS_OUTPUT_ROOT=${VARIATIONS_OUTPUT_ROOT:-$OUTPUT_ROOT/variations}

mkdir -p "$OUTPUT_ROOT" "$VARIATIONS_OUTPUT_ROOT/logs"
python Variations/scripts/train_diffusion.py --config Variations/configs/diffusion/medium.yaml \
  2>&1 | tee "$VARIATIONS_OUTPUT_ROOT/logs/train_diffusion_${SLURM_JOB_ID:-manual}.log"
