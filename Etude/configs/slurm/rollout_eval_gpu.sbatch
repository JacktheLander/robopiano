#!/bin/bash
#SBATCH --job-name=etude_rollout
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --output=logs/etude_rollout_%j.out

set -euo pipefail
cd /WAVE/projects/ECEN-524-Wi26/robopiano/Etude
conda activate sonata
MUJOCO_GL=egl python scripts/evaluate_tracker.py \
  --config configs/controller/debug.yaml \
  --checkpoint /WAVE/datasets/ccoelho_lab-jlanders/etude/outputs/controllers/debug_mlp/checkpoints/best.pt \
  --output-root /WAVE/datasets/ccoelho_lab-jlanders/etude/outputs/eval/debug_mlp \
  --backend dm_control \
  --render-video
