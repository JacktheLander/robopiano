#!/bin/bash
# train_primitives_cmp_stage1.sbatch
#
# Slurm: Stage 1 (keyset+onset) primitive pipeline for RP1M-300, partition cmp.
# ------------------------------------------------------------------------------
# HOW TO SUBMIT
#   1) Ensure the run root exists (creates subdirs: primitives/, data/, logs/):
#        mkdir -p /WAVE/datasets/ccoelho_lab-jlanders/outputs_run4/logs
#   2) Edit SONATA_ROOT below if your clone path differs.
#   3) Submit (add flags to override, e.g. more CPUs: sbatch --cpus-per-task=24 ...):
#        sbatch /path/to/Sonata/slurm/train_primitives_cmp_stage1.sbatch
#   4) Optional: auto-submit is intentionally NOT in this file; use tmux helper
#        `sonata_tmux_train_primitives.sh` to print the exact command without running it.
#
# HOW TO MONITOR
#   squeue -u "$USER"
#   squeue -j <jobid>        # or:  watch -n 30 squeue -u "$USER"
#   scontrol show job <jobid>
#
# HOW TO READ LOGS
#   Slurm stdout/stderr (paths below) plus Sonata’s own under:
#     ${STAGE1_OUTPUT_ROOT}/primitives/  (run_config.json, library/, eval/, etc.)
#     ${STAGE1_OUTPUT_ROOT}/data/         (dataset manifest, splits)
#   WandB (if enabled in config): check project/entity in configs/primitive/hpc_keyset_onset_stage1.yaml
#
# HOW TO RESUME SAFELY
#   - Do NOT pass --force unless you mean to re-scan the dataset and rebuild caches.
#   - Re-submit the same job: Stage 1 resumes from slim/progress, compact shards, and
#     will skip completed episodes (see STAGE1_ONLINE_PRIMITIVES.md).
#   - Point RP1M_300_ROOT and STAGE1_OUTPUT_ROOT at the same locations as the prior run.
#   - If you must restart clean, move or delete ${STAGE1_OUTPUT_ROOT} first, then resubmit.
#
# ------------------------------------------------------------------------------
# Slurm I/O: update these two paths if you change the default STAGE1_OUTPUT_ROOT below.
# They must stay under  ${STAGE1_OUTPUT_ROOT}/logs  for this run (same as mkdir -p in the job).
#SBATCH --job-name=sonata-s1-ko
#SBATCH --partition=cmp
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
# --- Logs: keep in sync with default STAGE1_OUTPUT_ROOT (see next block) -------
#SBATCH --output=/WAVE/datasets/ccoelho_lab-jlanders/outputs_run4/logs/slurm-%x-%j.out
#SBATCH --error=/WAVE/datasets/ccoelho_lab-jlanders/outputs_run4/logs/slurm-%x-%j.err
#SBATCH --export=ALL
#
set -euo pipefail

# --- Sonata code tree (set before sbatch if your clone is not in /WAVE/projects/...). Do not rely on
#     SLURM_SUBMIT_DIR — submitting from $HOME would point SONATA at the wrong place.
: "${SONATA_ROOT:=/WAVE/projects/ECEN-524-Wi26/robopiano/Sonata}"

# --- Canonical on-disk run root: ALL Stage 1 artifacts (primitives, data, W&B local cache). ---
#     Override:  export STAGE1_OUTPUT_ROOT=/path && sbatch this_script.sbatch
#     Use :- so an empty env var still falls back to the default.
STAGE1_OUTPUT_DEFAULT="/WAVE/datasets/ccoelho_lab-jlanders/outputs_run4"
export STAGE1_OUTPUT_ROOT="${STAGE1_OUTPUT_ROOT:-${STAGE1_OUTPUT_DEFAULT}}"
export RP1M_300_ROOT="${RP1M_300_ROOT:-/WAVE/users/unix/jlanders/rp1m_300/rp1m_repertoire.zarr}"
echo "SONATA_ROOT=${SONATA_ROOT}" >&2
echo "STAGE1_OUTPUT_ROOT=${STAGE1_OUTPUT_ROOT}" >&2

# Concurrency (keep at or below SLURM_cpus_per_task; override before sbatch if needed)
: "${CPUS_PER_TASK:=16}"
# Match worker pools to Slurm CPU allocation (when running under Slurm) or CPUS_PER_TASK
if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
  export SONATA_PRIMITIVE_NUM_WORKERS="${SONATA_PRIMITIVE_NUM_WORKERS:-$SLURM_CPUS_PER_TASK}"
  export SONATA_DATA_SCAN_WORKERS="${SONATA_DATA_SCAN_WORKERS:-$SLURM_CPUS_PER_TASK}"
else
  export SONATA_PRIMITIVE_NUM_WORKERS="${SONATA_PRIMITIVE_NUM_WORKERS:-$CPUS_PER_TASK}"
  export SONATA_DATA_SCAN_WORKERS="${SONATA_DATA_SCAN_WORKERS:-$CPUS_PER_TASK}"
fi

# Safer CPU BLAS / OpenMP with multiprocessing
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

# Sonata import path (script also inserts src, but this helps nested tools)
export PYTHONPATH="${SONATA_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

# W&B: keep local run files under this run root (override: export WANDB_DIR=... before sbatch)
export WANDB_DIR="${WANDB_DIR:-${STAGE1_OUTPUT_ROOT}/wandb}"

# W&B: default online; use offline on air-gapped nodes: export WANDB_MODE=offline

# Optional: RoboPianist checkout for post-hoc online eval
# export ROBOPIANIST_ROOT="/path/to/robopianist"
# export PYTHONPATH="${ROBOPIANIST_ROOT}:${PYTHONPATH:-}"

# Optional: run sim rollouts after Stage 1 (requires RoboPianist); 0=skip, 1=run
: "${RUN_PRIMITIVE_ONLINE_EVAL:=0}"

# Create run dirs early so Slurm can open -o / -e under ${STAGE1_OUTPUT_ROOT}/logs/ if needed
mkdir -p "${STAGE1_OUTPUT_ROOT}/logs" "${STAGE1_OUTPUT_ROOT}/primitives" "${STAGE1_OUTPUT_ROOT}/data" "${STAGE1_OUTPUT_ROOT}/wandb"

# --- Conda: WAVE module, site install, $HOME, or CONDA_SH
# Login nodes often have `module load Anaconda3`; batch may need a direct source.
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
  echo "Could not find conda. On WAVE: set CONDA_SH to .../etc/profile.d/conda.sh" >&2
  echo "  (e.g. /WAVE/apps/x86_64/packages/Anaconda3/2025.12-2/app/etc/profile.d/conda.sh)" >&2
  echo "  or ensure 'module load Anaconda3' works in your batch environment." >&2
  exit 1
fi

# Activate by path so batch jobs do not pick the wrong `python` (name-only is fragile).
# Override if your `sonata` env lives elsewhere: export SONATA_CONDA_PREFIX=/path/to/sonata
if [[ -z "${SONATA_CONDA_PREFIX:-}" ]]; then
  for _p in "${HOME}/.conda/envs/sonata" "/WAVE/users2/unix/jlanders/.conda/envs/sonata"; do
    if [[ -d "$_p" && -x "$_p/bin/python" ]]; then
      SONATA_CONDA_PREFIX="$_p"
      break
    fi
  done
fi
if [[ -z "${SONATA_CONDA_PREFIX:-}" || ! -x "${SONATA_CONDA_PREFIX}/bin/python" ]]; then
  echo "Could not find sonata env. Set SONATA_CONDA_PREFIX to the env with Sonata deps (wandb, zarr, ...)." >&2
  exit 1
fi
conda activate "${SONATA_CONDA_PREFIX}"
PY="${CONDA_PREFIX}/bin/python"
PIP="${CONDA_PREFIX}/bin/pip"
echo "Conda: ${CONDA_PREFIX}" >&2
echo "Python: $(${PY} -V) (${PY})" >&2

# W&B: install into the same env if missing (not base Anaconda)
if ! "${PY}" -c "import wandb" 2>/dev/null; then
  echo "wandb not in this env; pip installing into ${CONDA_PREFIX} ..." >&2
  "${PIP}" install "wandb>=0.17"
fi

cd "${SONATA_ROOT}"

HPC_CONFIG="${SONATA_ROOT}/configs/primitive/hpc_keyset_onset_stage1.yaml"
if [[ ! -f "${HPC_CONFIG}" ]]; then
  echo "Missing HPC config: ${HPC_CONFIG}" >&2
  exit 1
fi
# Marker: confirms this job is writing to this STAGE1 tree (before train_primitives starts).
{
  echo "STAGE1_OUTPUT_ROOT=${STAGE1_OUTPUT_ROOT}"
  echo "SONATA_ROOT=${SONATA_ROOT}"
  echo "RP1M_300_ROOT=${RP1M_300_ROOT}"
  date -u "+%Y-%m-%dT%H:%M:%SZ"
} > "${STAGE1_OUTPUT_ROOT}/primitives/.stage1_run_directory.txt"
echo "Stage1 outputs: primitives -> ${STAGE1_OUTPUT_ROOT}/primitives" >&2
echo "Stage1 outputs: data       -> ${STAGE1_OUTPUT_ROOT}/data" >&2
echo "W&B local dir:  ${WANDB_DIR}" >&2
# Fail fast if ${STAGE1_OUTPUT_ROOT} in YAML does not match this run root
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

# Primitive config: keyset+onset + online slim; data: medium (max_episodes: 20 per song)
# --output-root forces primitive_root to match this script even if YAML resolution differed.
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
