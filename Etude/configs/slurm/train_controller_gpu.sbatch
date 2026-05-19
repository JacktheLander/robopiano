#!/bin/bash
#SBATCH --job-name=etude_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/etude_train_%j.out

set -euo pipefail
cd /WAVE/projects/ECEN-524-Wi26/robopiano/Etude
conda activate sonata
python scripts/train_controller.py \
  --config configs/controller/debug.yaml \
  --output-root /WAVE/datasets/ccoelho_lab-jlanders/etude/outputs/controllers/debug_mlp \
  --wandb
