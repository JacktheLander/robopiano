#!/bin/bash
#SBATCH -J variations_mlp
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --chdir=/WAVE/projects/ECEN-524-Wi26/robopiano
#SBATCH -o /WAVE/datasets/ccoelho_lab-jlanders/Variations/slurm_logs/variations_mlp_%j.out
#SBATCH -e /WAVE/datasets/ccoelho_lab-jlanders/Variations/slurm_logs/variations_mlp_%j.err

set -eo pipefail

source ~/.bashrc
conda activate sonata
set -u
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

export RUN_NAME=${RUN_NAME:-variations_mlp_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}
export OUTPUT_ROOT=${OUTPUT_ROOT:-/WAVE/datasets/ccoelho_lab-jlanders/Variations/$RUN_NAME}
export VARIATIONS_OUTPUT_ROOT=${VARIATIONS_OUTPUT_ROOT:-$OUTPUT_ROOT/variations}
export VARIATIONS_MLP_CONFIG=${VARIATIONS_MLP_CONFIG:-Variations/configs/mlp_baseline.yaml}

mkdir -p "$OUTPUT_ROOT" "$VARIATIONS_OUTPUT_ROOT/logs"
python Variations/scripts/train_mlp_baseline.py --config "$VARIATIONS_MLP_CONFIG" \
  2>&1 | tee "$VARIATIONS_OUTPUT_ROOT/logs/train_mlp_baseline_${SLURM_JOB_ID:-manual}.log"

