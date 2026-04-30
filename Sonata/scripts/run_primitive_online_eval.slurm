#!/bin/bash
#SBATCH --job-name=prim-online
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/WAVE/datasets/ccoelho_lab-jlanders/outputs_run4/logs/prim_online_%j.out
#SBATCH --error=/WAVE/datasets/ccoelho_lab-jlanders/outputs_run4/logs/prim_online_%j.err

set -euo pipefail

source ~/.bashrc
conda activate sonata

cd /WAVE/projects/ECEN-524-Wi26/robopiano/Sonata

export PYTHONPATH=$PWD/src:$PWD/../robopianist:$PYTHONPATH
export MUJOCO_GL=egl

mkdir -p /WAVE/datasets/ccoelho_lab-jlanders/outputs_run4/logs

python scripts/evaluate_primitives_online.py \
  --primitive-root /WAVE/datasets/ccoelho_lab-jlanders/outputs_run4/primitives \
  --output-root /WAVE/datasets/ccoelho_lab-jlanders/outputs_run4/evaluation/primitives_online \
  --instances-per-primitive 4 \
  --save-plots \
  --force
