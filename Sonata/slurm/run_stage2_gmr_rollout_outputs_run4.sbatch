#!/bin/bash
# run_stage2_gmr_rollout_outputs_run4.sbatch
#
# Online DM Control rollout: Stage 2 planner → GMR priors (no diffusion).
# Uses Stage 1 primitives + best Stage 2 checkpoint under outputs_run4 by default.
#
# Writes under:
#   ${STAGE2_OUTPUT_ROOT}/evaluation/${ROLL_RUN_TITLE}/${ROLL_TS}/
# Example:
#   .../outputs_run4/evaluation/stage2_gmr_online_outputs_run4/20260204-143022/
#
# Usage:
#   export SONATA_ROOT=/WAVE/projects/ECEN-524-Wi26/robopiano/Sonata
#   export STAGE2_OUTPUT_ROOT=/WAVE/datasets/ccoelho_lab-jlanders/outputs_run4
#   # optional:
#   export ROLL_RUN_TITLE=my_rollout_label
#   export EPISODE_ID='RoboPianist-repertoire-150-BalladeNo1-v0_500__ep00000'
#   export MAESTRO_MIDI_ROOT=/WAVE/datasets/ccoelho_lab-jlanders/maestro-v3.0.0
#   sbatch --export=ALL "${SONATA_ROOT}/slurm/run_stage2_gmr_rollout_outputs_run4.sbatch"
#
#SBATCH --job-name=s2-gmr-rollout
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --output=/WAVE/datasets/ccoelho_lab-jlanders/outputs_run4/logs/slurm-stage2-gmr-rollout-%x-%j.out
#SBATCH --error=/WAVE/datasets/ccoelho_lab-jlanders/outputs_run4/logs/slurm-stage2-gmr-rollout-%x-%j.err
#SBATCH --export=ALL

set -euo pipefail

: "${SONATA_ROOT:=/WAVE/projects/ECEN-524-Wi26/robopiano/Sonata}"
STAGE2_OUTPUT_DEFAULT="/WAVE/datasets/ccoelho_lab-jlanders/outputs_run4"
export STAGE2_OUTPUT_ROOT="${STAGE2_OUTPUT_ROOT:-${STAGE2_OUTPUT_DEFAULT}}"

DEFAULT_PLANNER="${STAGE2_OUTPUT_ROOT}/transformer2_mlp_dynhist_lowloss/Sonata-3-transformer-Sonata-3-medium-planner-remap-mlp-dynhist-lowloss-noplanemb-seed7-20260503-211143/checkpoints/best.pt"
: "${PLANNER_CHECKPOINT:=${DEFAULT_PLANNER}}"
: "${PRIMITIVE_ROOT:=${STAGE2_OUTPUT_ROOT}/primitives}"

: "${ROLL_RUN_TITLE:=stage2_gmr_online_outputs_run4}"
ROLL_TS="$(date +%Y%m%d-%H%M%S)"
OUTPUT_EVAL_ROOT="${STAGE2_OUTPUT_ROOT}/evaluation"
OUTPUT_ROOT="${OUTPUT_EVAL_ROOT}/${ROLL_RUN_TITLE}/${ROLL_TS}"

: "${SPLIT:=val}"
: "${LIMIT_EPISODES:=1}"
: "${DEVICE:=cuda}"
: "${ENVIRONMENT_NAME:=RoboPianist-debug-TwinkleTwinkleLittleStar-v0}"
: "${MUJOCO_GL:=egl}"

: "${MAESTRO_MIDI_ROOT:=/WAVE/datasets/ccoelho_lab-jlanders/maestro-v3.0.0}"
MIDI_ARGS=()
if [[ -d "${MAESTRO_MIDI_ROOT}" ]]; then
  MIDI_ARGS+=(--midi-root "${MAESTRO_MIDI_ROOT}")
fi

EPISODE_ARGS=()
if [[ -n "${EPISODE_ID:-}" ]]; then
  EPISODE_ARGS+=(--episode-id "${EPISODE_ID}")
fi

echo "SONATA_ROOT=${SONATA_ROOT}" >&2
echo "STAGE2_OUTPUT_ROOT=${STAGE2_OUTPUT_ROOT}" >&2
echo "PRIMITIVE_ROOT=${PRIMITIVE_ROOT}" >&2
echo "PLANNER_CHECKPOINT=${PLANNER_CHECKPOINT}" >&2
echo "OUTPUT_ROOT=${OUTPUT_ROOT}" >&2
echo "ROLL_RUN_TITLE=${ROLL_RUN_TITLE} ROLL_TS=${ROLL_TS}" >&2

if [[ ! -f "${PLANNER_CHECKPOINT}" ]]; then
  echo "Planner checkpoint missing: ${PLANNER_CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -d "${PRIMITIVE_ROOT}/tokens" ]]; then
  echo "Missing primitives tokens under ${PRIMITIVE_ROOT}/tokens" >&2
  exit 1
fi

mkdir -p "${OUTPUT_EVAL_ROOT}" "${STAGE2_OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}"

export PYTHONPATH="${SONATA_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

_activated_conda=0
if [[ -n "${CONDA_SH:-}" && -f "${CONDA_SH}" ]]; then
  # shellcheck source=/dev/null
  source "${CONDA_SH}"
  _activated_conda=1
elif type module &>/dev/null && module load Anaconda3 2>/dev/null; then
  _activated_conda=1
elif [[ -f "${HOME}/.bashrc" ]]; then
  # shellcheck source=/dev/null
  source "${HOME}/.bashrc"
  _activated_conda=1
else
  for _condash in \
    "/WAVE/apps/x86_64/packages/Anaconda3/2025.12-2/app/etc/profile.d/conda.sh" \
    "/WAVE/apps/x86_64/packages/Anaconda3/2025.06-1/app/etc/profile.d/conda.sh" \
    "${HOME}/miniconda3/etc/profile.d/conda.sh" \
    "${HOME}/anaconda3/etc/profile.d/conda.sh"
  do
    if [[ -f "$_condash" ]]; then
      # shellcheck source=/dev/null
      source "$_condash"
      _activated_conda=1
      break
    fi
  done
fi
if [[ "${_activated_conda}" -eq 0 ]]; then
  echo "Could not initialize conda (set CONDA_SH)." >&2
  exit 1
fi

conda activate sonata
PY="${CONDA_PREFIX}/bin/python"
echo "Python: $($PY -V) (${PY})" >&2

cd "${SONATA_ROOT}"

export MUJOCO_GL

"$PY" scripts/evaluate_stage2_rollout.py \
  --primitive-root "${PRIMITIVE_ROOT}" \
  --planner-checkpoint "${PLANNER_CHECKPOINT}" \
  --output-root "${OUTPUT_ROOT}" \
  --mode stage2_gmr \
  --backend dm_control \
  --split "${SPLIT}" \
  --device "${DEVICE}" \
  --limit-episodes "${LIMIT_EPISODES}" \
  --environment-name "${ENVIRONMENT_NAME}" \
  "${MIDI_ARGS[@]}" \
  "${EPISODE_ARGS[@]}" \
  --render-video \
  --max-render-episodes "${LIMIT_EPISODES}" \
  --log-level INFO

echo "Rollout finished. Artifacts: ${OUTPUT_ROOT}" >&2
