#!/bin/bash
#SBATCH --job-name=partita-val
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=/WAVE/projects/ECEN-524-Wi26/robopiano/partita/outputs/logs/partita_validation_%j.out
#SBATCH --error=/WAVE/projects/ECEN-524-Wi26/robopiano/partita/outputs/logs/partita_validation_%j.err

set -eo pipefail

source ~/.bashrc
set -u
conda activate sonata

cd /WAVE/projects/ECEN-524-Wi26/robopiano

export PYTHONPATH=$PWD/partita/src:$PWD:$PWD/robopianist:${PYTHONPATH:-}
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export MUJOCO_GL=egl

CONFIG=${PARTITA_CONFIG:-partita/configs/debug.yaml}
WHICH=${PARTITA_ROLLOUT_WHICH:-both}
WIDTH=${PARTITA_VIDEO_WIDTH:-640}
HEIGHT=${PARTITA_VIDEO_HEIGHT:-480}
FPS=${PARTITA_VIDEO_FPS:-20}
RENDER_EVERY=${PARTITA_RENDER_EVERY:-1}
MAX_STEPS_FLAG=()
if [[ -n "${PARTITA_MAX_STEPS:-}" ]]; then
  MAX_STEPS_FLAG=(--max-steps "${PARTITA_MAX_STEPS}")
fi

mkdir -p partita/outputs/logs

python partita/scripts/run_partita_debug.py --config "${CONFIG}"
python partita/scripts/simulate_rollout.py \
  --config "${CONFIG}" \
  --which "${WHICH}" \
  --width "${WIDTH}" \
  --height "${HEIGHT}" \
  --fps "${FPS}" \
  --render-every "${RENDER_EVERY}" \
  "${MAX_STEPS_FLAG[@]}"
python partita/scripts/evaluate_reconstruction.py --config "${CONFIG}"
