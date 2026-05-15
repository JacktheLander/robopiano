#!/bin/bash
#SBATCH --job-name=mlp_precise_labels
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --chdir=/WAVE/projects/ECEN-524-Wi26/robopiano
#SBATCH --output=/WAVE/datasets/ccoelho_lab-jlanders/Variations/slurm_logs/mlp_precise_labels_%j.out
#SBATCH --error=/WAVE/datasets/ccoelho_lab-jlanders/Variations/slurm_logs/mlp_precise_labels_%j.err

set -eo pipefail
source ~/.bashrc
conda activate sonata
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
set -u

python Variations/scripts/generate_contact_refined_labels.py \
  --config Variations/configs/eval_press_pose.yaml \
  --extraction-root /WAVE/datasets/ccoelho_lab-jlanders/Variations/variations_rp1m_full_20260510_152232/variations/extraction/full \
  --model-type mlp_baseline \
  --checkpoint /WAVE/datasets/ccoelho_lab-jlanders/Variations/variations_tuned_20260512_190752/variations/mlp_baseline/joints_tuned/checkpoints/best.pt \
  --output /WAVE/datasets/ccoelho_lab-jlanders/Variations/variations_tuned_20260512_190752/variations/contact_refinement/mlp_precise_labels.npz \
  --max-samples 16025 \
  --max-iter 120 \
  --batch-size 256 \
  --device auto \
  --pose-reg-weight 0.003 \
  --smooth-weight 0.0 \
  --inactive-weight 0.05 \
  --inactive-margin-m 0.018 \
  --z-weight 1.0
