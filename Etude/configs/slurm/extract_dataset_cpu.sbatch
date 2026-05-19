#!/bin/bash
#SBATCH --job-name=etude_extract
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-08:00:00
#SBATCH --output=logs/etude_extract_%j.out

set -euo pipefail
cd /WAVE/projects/ECEN-524-Wi26/robopiano/Etude
conda activate sonata
python scripts/extract_tracking_dataset.py \
  --rp1m-root /WAVE/users/unix/jlanders/rp1m_300/rp1m_repertoire.zarr \
  --profile debug \
  --output-root /WAVE/datasets/ccoelho_lab-jlanders/etude/data/debug \
  --force
