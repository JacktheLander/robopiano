#!/bin/bash
#SBATCH -J variations_compare
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --chdir=/WAVE/projects/ECEN-524-Wi26/robopiano
#SBATCH -o /WAVE/datasets/ccoelho_lab-jlanders/Variations/slurm_logs/variations_compare_%j.out
#SBATCH -e /WAVE/datasets/ccoelho_lab-jlanders/Variations/slurm_logs/variations_compare_%j.err

set -eo pipefail

source ~/.bashrc
conda activate sonata
set -u
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

export RUN_NAME=${RUN_NAME:-variations_compare_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}
export OUTPUT_ROOT=${OUTPUT_ROOT:-/WAVE/datasets/ccoelho_lab-jlanders/Variations/$RUN_NAME}
export VARIATIONS_OUTPUT_ROOT=${VARIATIONS_OUTPUT_ROOT:-$OUTPUT_ROOT/variations}
export VARIATIONS_EVAL_CONFIG=${VARIATIONS_EVAL_CONFIG:-Variations/configs/eval_press_pose.yaml}
export LATENT_MDN_CHECKPOINT=${LATENT_MDN_CHECKPOINT:-$VARIATIONS_OUTPUT_ROOT/latent_mdn/mdn/checkpoints/best.pt}
export MLP_CHECKPOINT=${MLP_CHECKPOINT:-$VARIATIONS_OUTPUT_ROOT/mlp_baseline/joints/checkpoints/best.pt}
export DIFFUSION_CHECKPOINT=${DIFFUSION_CHECKPOINT:-$VARIATIONS_OUTPUT_ROOT/diffusion/full/checkpoints/best.pt}
export LATENT_MDN_COMPARISON_OUTPUT=${LATENT_MDN_COMPARISON_OUTPUT:-$VARIATIONS_OUTPUT_ROOT/latent_mdn/comparisons/latent_mdn_metrics.csv}
export FINGERTIP_EVAL_MAX_SAMPLES=${FINGERTIP_EVAL_MAX_SAMPLES:-256}
export FINGERTIP_SETTLE_STEPS=${FINGERTIP_SETTLE_STEPS:-0}

mkdir -p "$OUTPUT_ROOT" "$VARIATIONS_OUTPUT_ROOT/logs"
python Variations/scripts/evaluate_latent_mdn.py \
  --config "$VARIATIONS_EVAL_CONFIG" \
  --latent-mdn-checkpoint "$LATENT_MDN_CHECKPOINT" \
  --mlp-checkpoint "$MLP_CHECKPOINT" \
  --diffusion-checkpoint "$DIFFUSION_CHECKPOINT" \
  --output "$LATENT_MDN_COMPARISON_OUTPUT" \
  --fingertip-eval-max-samples "$FINGERTIP_EVAL_MAX_SAMPLES" \
  --fingertip-settle-steps "$FINGERTIP_SETTLE_STEPS" \
  2>&1 | tee "$VARIATIONS_OUTPUT_ROOT/logs/evaluate_latent_mdn_${SLURM_JOB_ID:-manual}.log"
