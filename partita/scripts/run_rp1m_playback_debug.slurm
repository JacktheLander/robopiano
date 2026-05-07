#!/bin/bash
#SBATCH --job-name=rp1m-playback
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --output=/WAVE/projects/ECEN-524-Wi26/robopiano/partita/outputs/logs/rp1m_playback_%j.out
#SBATCH --error=/WAVE/projects/ECEN-524-Wi26/robopiano/partita/outputs/logs/rp1m_playback_%j.err

set -eo pipefail

source ~/.bashrc
set -u
conda activate sonata

cd /WAVE/projects/ECEN-524-Wi26/robopiano

export PYTHONPATH=$PWD/partita/src:$PWD:$PWD/robopianist:${PYTHONPATH:-}
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export MUJOCO_GL=egl

CONFIG=${PARTITA_CONFIG:-partita/configs/debug.yaml}
WIDTH=${PARTITA_VIDEO_WIDTH:-640}
HEIGHT=${PARTITA_VIDEO_HEIGHT:-480}
FPS=${PARTITA_VIDEO_FPS:-20}
RENDER_FLAG=${PARTITA_RENDER_VIDEO:---render-video}
MAX_STEPS_FLAG=()
if [[ -n "${PARTITA_MAX_STEPS:-}" ]]; then
  MAX_STEPS_FLAG=(--max-steps "${PARTITA_MAX_STEPS}")
fi

mkdir -p partita/outputs/logs

# Ensure the selected target trajectory exists. This is cheap if outputs already exist.
if [[ ! -f partita/outputs/data/partita_sarabande_multitraj_debug/target_trajectory.npz ]]; then
  python partita/scripts/select_song_trajectories.py --config "${CONFIG}"
  python partita/scripts/extract_song_trajectories.py --config "${CONFIG}"
fi

python partita/scripts/debug_rp1m_playback.py \
  --config "${CONFIG}" \
  ${RENDER_FLAG} \
  --width "${WIDTH}" \
  --height "${HEIGHT}" \
  --fps "${FPS}" \
  "${MAX_STEPS_FLAG[@]}"
