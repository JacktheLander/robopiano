#!/bin/bash
# train_primitives_cmp_stage1_resume.sbatch
#
# Copy of train_primitives_cmp_stage1.sbatch tuned for CONTINUE / RESUME after OOM or partial runs:
# - Higher Slurm RAM request (below) to finish GMM/GMR/refinement stages.
# - Explicitly avoids --force on train_primitives so existing slim/segment progress,
#   run_config-driven outputs, and non-destructive caches are preserved (Sonata skips
#   completed episodes per slim/progress; see STAGE1_ONLINE_PRIMITIVES.md).
#
# Submit with the SAME STAGE1_OUTPUT_ROOT and RP1M_300_ROOT as the run you want to resume.
#
# ------------------------------------------------------------------------------
# Slurm log paths MUST stay under ${STAGE1_OUTPUT_ROOT}/logs — edit lines below if that root changes.
#SBATCH --job-name=sonata-s1-ko-res
#SBATCH --partition=cmp
#SBATCH --mem=512G
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=16
# --- Logs ---
#SBATCH --output=/WAVE/datasets/ccoelho_lab-jlanders/outputs_run4/logs/slurm-%x-%j.out
#SBATCH --error=/WAVE/datasets/ccoelho_lab-jlanders/outputs_run4/logs/slurm-%x-%j.err
#SBATCH --export=ALL
#
set -euo pipefail

: "${SONATA_ROOT:=/WAVE/projects/ECEN-524-Wi26/robopiano/Sonata}"

STAGE1_OUTPUT_DEFAULT="/WAVE/datasets/ccoelho_lab-jlanders/outputs_run4"
export STAGE1_OUTPUT_ROOT="${STAGE1_OUTPUT_ROOT:-${STAGE1_OUTPUT_DEFAULT}}"
export RP1M_300_ROOT="${RP1M_300_ROOT:-/WAVE/users/unix/jlanders/rp1m_300/rp1m_repertoire.zarr}"

echo "[resume batch] SONATA_ROOT=${SONATA_ROOT}" >&2
echo "[resume batch] STAGE1_OUTPUT_ROOT=${STAGE1_OUTPUT_ROOT}" >&2
echo "[resume batch] Intended: continue from existing primitives under STAGE1; --force is NOT passed." >&2

# Concurrency (keep at or below SLURM_cpus_per_task).
: "${CPUS_PER_TASK:=16}"
if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
  export SONATA_PRIMITIVE_NUM_WORKERS="${SONATA_PRIMITIVE_NUM_WORKERS:-$SLURM_CPUS_PER_TASK}"
  export SONATA_DATA_SCAN_WORKERS="${SONATA_DATA_SCAN_WORKERS:-$SLURM_CPUS_PER_TASK}"
else
  export SONATA_PRIMITIVE_NUM_WORKERS="${SONATA_PRIMITIVE_NUM_WORKERS:-$CPUS_PER_TASK}"
  export SONATA_DATA_SCAN_WORKERS="${SONATA_DATA_SCAN_WORKERS:-$CPUS_PER_TASK}"
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

export PYTHONPATH="${SONATA_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export WANDB_DIR="${WANDB_DIR:-${STAGE1_OUTPUT_ROOT}/wandb}"

: "${RUN_PRIMITIVE_ONLINE_EVAL:=0}"

mkdir -p "${STAGE1_OUTPUT_ROOT}/logs" "${STAGE1_OUTPUT_ROOT}/primitives" "${STAGE1_OUTPUT_ROOT}/data" "${STAGE1_OUTPUT_ROOT}/wandb"

if [[ ! -d "${STAGE1_OUTPUT_ROOT}/primitives/slim" ]]; then
  echo "[resume batch] WARNING: no primitives/slim found — nothing to resume; first run start_fresh script or check STAGE1_OUTPUT_ROOT." >&2
fi

# --- Conda ---
_activated_conda=0
if [[ -n "${CONDA_SH:-}" && -f "${CONDA_SH}" ]]; then
  # shellcheck source=/dev/null
  source "${CONDA_SH}"
  _activated_conda=1
elif type module &>/dev/null; then
  if module load Anaconda3 2>/dev/null; then
    _activated_conda=1
  fi
fi
if [[ "${_activated_conda}" -eq 0 ]]; then
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
if ! command -v conda &>/dev/null; then
  echo "Could not find conda." >&2
  exit 1
fi

if [[ -z "${SONATA_CONDA_PREFIX:-}" ]]; then
  for _p in "${HOME}/.conda/envs/sonata" "/WAVE/users2/unix/jlanders/.conda/envs/sonata"; do
    if [[ -d "$_p" && -x "$_p/bin/python" ]]; then
      SONATA_CONDA_PREFIX="$_p"
      break
    fi
  done
fi
if [[ -z "${SONATA_CONDA_PREFIX:-}" || ! -x "${SONATA_CONDA_PREFIX}/bin/python" ]]; then
  echo "Could not find sonata env." >&2
  exit 1
fi
conda activate "${SONATA_CONDA_PREFIX}"
PY="${CONDA_PREFIX}/bin/python"
PIP="${CONDA_PREFIX}/bin/pip"
echo "Conda: ${CONDA_PREFIX}" >&2
echo "Python: $(${PY} -V) (${PY})" >&2

if ! "${PY}" -c "import wandb" 2>/dev/null; then
  "${PIP}" install "wandb>=0.17"
fi

cd "${SONATA_ROOT}"

HPC_CONFIG="${SONATA_ROOT}/configs/primitive/hpc_keyset_onset_stage1.yaml"
if [[ ! -f "${HPC_CONFIG}" ]]; then
  echo "Missing HPC config: ${HPC_CONFIG}" >&2
  exit 1
fi

# Append-only history (does not delete or replace slim data).
{
  echo "=== $(date -u '+%Y-%m-%dT%H:%M:%SZ') job_start ==="
  echo "STAGE1_OUTPUT_ROOT=${STAGE1_OUTPUT_ROOT}"
  echo "SONATA_ROOT=${SONATA_ROOT}"
  echo "RP1M_300_ROOT=${RP1M_300_ROOT}"
  echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
  echo "train_primitives: NO --force (resume-safe)"
} >> "${STAGE1_OUTPUT_ROOT}/primitives/stage1_launch_history.log"

echo "Stage1 outputs: primitives -> ${STAGE1_OUTPUT_ROOT}/primitives" >&2
echo "Stage1 outputs: data       -> ${STAGE1_OUTPUT_ROOT}/data" >&2
echo "W&B local dir:  ${WANDB_DIR}" >&2

export HPC_CONFIG_PATH="${HPC_CONFIG}"
"${PY}" - <<'PY' || exit 1
import os, sys
from pathlib import Path
root = os.environ.get("STAGE1_OUTPUT_ROOT", "").strip()
if not root:
    print("STAGE1_OUTPUT_ROOT is empty in Python", file=sys.stderr)
    raise SystemExit(1)
sys.path.insert(0, str(Path("src").resolve()))
from sonata.config import load_stage_config, resolve_path
cfg = load_stage_config("primitive", config_path=os.environ["HPC_CONFIG_PATH"])
cdir = Path(cfg["config_path"]).parent
out = Path(resolve_path(cfg["output_root"], cdir))
data = Path(resolve_path(cfg["data_output_root"], cdir))
exp_prim = (Path(root) / "primitives").resolve()
exp_data = (Path(root) / "data").resolve()
print("HPC preflight: output_root   ->", out, file=sys.stderr)
print("HPC preflight: data_output   ->", data, file=sys.stderr)
if out.resolve() != exp_prim or data.resolve() != exp_data:
    print("HPC preflight: expected", exp_prim, "and", exp_data, file=sys.stderr)
    raise SystemExit(1)
PY

# RESUME-SAFE: do not pass --force. Pipeline continues from slim/progress and existing artifacts.
"${PY}" scripts/train_primitives.py \
  --config "${HPC_CONFIG}" \
  --profile medium \
  --output-root "${STAGE1_OUTPUT_ROOT}/primitives" \
  --log-level INFO

if [[ "${RUN_PRIMITIVE_ONLINE_EVAL}" == "1" ]]; then
  EVAL_OUT="${STAGE1_OUTPUT_ROOT}/eval_primitives_online"
  mkdir -p "${EVAL_OUT}"
  "${PY}" scripts/evaluate_primitives_online.py \
    --primitive-root "${STAGE1_OUTPUT_ROOT}/primitives" \
    --output-root "${EVAL_OUT}" \
    --log-level INFO \
    ${ROBOPIANIST_ROOT:+--robopianist-root "${ROBOPIANIST_ROOT}"}
fi
