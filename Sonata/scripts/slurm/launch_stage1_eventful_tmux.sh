#!/bin/bash
set -euo pipefail

SESSION_NAME="${SESSION_NAME:-sonata-stage1-eventful}"
PROJECT_ROOT="${PROJECT_ROOT:-/WAVE/projects/ECEN-524-Wi26/robopiano}"
SLURM_SCRIPT="${SLURM_SCRIPT:-${PROJECT_ROOT}/Sonata/scripts/slurm/train_primitives_eventful.slurm}"
PERSIST_ROOT="${PERSIST_ROOT:-/WAVE/datasets/ccoelho_lab-jlanders}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required for this launcher." >&2
  exit 1
fi

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is required for this launcher." >&2
  exit 1
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session ${SESSION_NAME} already exists." >&2
  exit 1
fi

mkdir -p "${PERSIST_ROOT}/sonata_logs"

tmux new-session -d -s "${SESSION_NAME}" "bash -lc 'export PROJECT_ROOT=\"${PROJECT_ROOT:-}\"; export RP1M_300_ROOT=\"${RP1M_300_ROOT:-}\"; export MAESTRO_MIDI_ROOT=\"${MAESTRO_MIDI_ROOT:-}\"; export ROBOPIANIST_ROOT=\"${ROBOPIANIST_ROOT:-}\"; export PERSIST_ROOT=\"${PERSIST_ROOT:-}\"; sbatch --export=ALL,PROJECT_ROOT=\"${PROJECT_ROOT:-}\",RP1M_300_ROOT=\"${RP1M_300_ROOT:-}\",MAESTRO_MIDI_ROOT=\"${MAESTRO_MIDI_ROOT:-}\",ROBOPIANIST_ROOT=\"${ROBOPIANIST_ROOT:-}\",PERSIST_ROOT=\"${PERSIST_ROOT:-}\" \"${SLURM_SCRIPT}\"; echo; echo \"Submitted from tmux session ${SESSION_NAME}.\"; echo \"Use squeue -u \$USER and tail -f ${PERSIST_ROOT}/sonata_logs/*.out to monitor.\"; exec bash'"
echo "Started tmux session ${SESSION_NAME} and submitted ${SLURM_SCRIPT}"
echo "Attach with: tmux attach -t ${SESSION_NAME}"
