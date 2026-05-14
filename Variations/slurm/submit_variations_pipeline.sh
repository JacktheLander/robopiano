#!/usr/bin/env bash
# Submit extraction, three GPU training jobs (diffusion, MLP, latent MDN), then the
# three-way comparison. Training jobs start in parallel once extraction finishes;
# comparison runs after all three trainings succeed.
#
# Usage (from repo root, after conda setup is available to Slurm jobs as in other .slurm files):
#   export RUN_NAME=variations_$(date +%Y%m%d_%H%M%S)   # optional; default timestamped
#   bash Variations/slurm/submit_variations_pipeline.sh
#
# Defaults use the "full" extraction + diffusion configs so paths match mlp_baseline.yaml
# and latent_mdn.yaml (both read Variations/outputs/extraction/full). Override if needed:
#   export VARIATIONS_EXTRACT_CONFIG=Variations/configs/extraction/medium.yaml
#   export VARIATIONS_TRAIN_CONFIG=Variations/configs/diffusion/medium.yaml
#   export DIFFUSION_CHECKPOINT=$VARIATIONS_OUTPUT_ROOT/diffusion/medium/checkpoints/best.pt
#
set -euo pipefail

ROBOPIANO_ROOT=/WAVE/projects/ECEN-524-Wi26/robopiano
cd "$ROBOPIANO_ROOT"

mkdir -p /WAVE/datasets/ccoelho_lab-jlanders/Variations/slurm_logs

RUN_NAME=${RUN_NAME:-variations_$(date +%Y%m%d_%H%M%S)}
OUTPUT_ROOT=${OUTPUT_ROOT:-/WAVE/datasets/ccoelho_lab-jlanders/Variations/$RUN_NAME}
VARIATIONS_OUTPUT_ROOT=${VARIATIONS_OUTPUT_ROOT:-$OUTPUT_ROOT/variations}
RP1M_ROOT=${RP1M_ROOT:-/WAVE/users/unix/jlanders/rp1m_300/rp1m_repertoire.zarr}

VARIATIONS_EXTRACT_CONFIG=${VARIATIONS_EXTRACT_CONFIG:-Variations/configs/extraction/full.yaml}
VARIATIONS_TRAIN_CONFIG=${VARIATIONS_TRAIN_CONFIG:-Variations/configs/diffusion/full.yaml}

# Optional overrides for evaluate_latent_mdn.slurm (defaults match diffusion/full).
DIFFUSION_CHECKPOINT=${DIFFUSION_CHECKPOINT:-$VARIATIONS_OUTPUT_ROOT/diffusion/full/checkpoints/best.pt}
MLP_CHECKPOINT=${MLP_CHECKPOINT:-$VARIATIONS_OUTPUT_ROOT/mlp_baseline/joints/checkpoints/best.pt}
LATENT_MDN_CHECKPOINT=${LATENT_MDN_CHECKPOINT:-$VARIATIONS_OUTPUT_ROOT/latent_mdn/mdn/checkpoints/best.pt}
LATENT_MDN_COMPARISON_OUTPUT=${LATENT_MDN_COMPARISON_OUTPUT:-$VARIATIONS_OUTPUT_ROOT/latent_mdn/comparisons/latent_mdn_metrics.csv}

COMMON_EXPORT=(
  ALL
  "RUN_NAME=$RUN_NAME"
  "OUTPUT_ROOT=$OUTPUT_ROOT"
  "VARIATIONS_OUTPUT_ROOT=$VARIATIONS_OUTPUT_ROOT"
  "RP1M_ROOT=$RP1M_ROOT"
)

EXTRACT_EXPORT=(
  "${COMMON_EXPORT[@]}"
  "VARIATIONS_EXTRACT_CONFIG=$VARIATIONS_EXTRACT_CONFIG"
)
IFS=','
EXTRACT_EXPORT_STR="${EXTRACT_EXPORT[*]}"
unset IFS

TRAIN_EXPORT=(
  "${COMMON_EXPORT[@]}"
  "VARIATIONS_TRAIN_CONFIG=$VARIATIONS_TRAIN_CONFIG"
)
IFS=','
TRAIN_EXPORT_STR="${TRAIN_EXPORT[*]}"
unset IFS

COMPARE_EXPORT=(
  "${COMMON_EXPORT[@]}"
  "DIFFUSION_CHECKPOINT=$DIFFUSION_CHECKPOINT"
  "MLP_CHECKPOINT=$MLP_CHECKPOINT"
  "LATENT_MDN_CHECKPOINT=$LATENT_MDN_CHECKPOINT"
  "LATENT_MDN_COMPARISON_OUTPUT=$LATENT_MDN_COMPARISON_OUTPUT"
)
IFS=','
COMPARE_EXPORT_STR="${COMPARE_EXPORT[*]}"
unset IFS

echo "RUN_NAME=$RUN_NAME"
echo "VARIATIONS_OUTPUT_ROOT=$VARIATIONS_OUTPUT_ROOT"
echo "EXTRACT_CONFIG=$VARIATIONS_EXTRACT_CONFIG"
echo "DIFFUSION_TRAIN_CONFIG=$VARIATIONS_TRAIN_CONFIG"

EXTRACT_ID=$(sbatch --parsable --export="$EXTRACT_EXPORT_STR" Variations/slurm/extract_press_pairs.slurm)
echo "Submitted extraction job: $EXTRACT_ID"

DIFFUSION_ID=$(sbatch --parsable --dependency=afterok:"$EXTRACT_ID" --export="$TRAIN_EXPORT_STR" Variations/slurm/train_diffusion.slurm)
MLP_ID=$(sbatch --parsable --dependency=afterok:"$EXTRACT_ID" --export="$TRAIN_EXPORT_STR" Variations/slurm/train_mlp_baseline.slurm)
LATENT_ID=$(sbatch --parsable --dependency=afterok:"$EXTRACT_ID" --export="$TRAIN_EXPORT_STR" Variations/slurm/train_latent_mdn.slurm)

echo "Submitted training jobs (after extraction $EXTRACT_ID): diffusion=$DIFFUSION_ID mlp=$MLP_ID latent_mdn=$LATENT_ID"

COMPARE_ID=$(sbatch --parsable --dependency=afterok:"$DIFFUSION_ID:$MLP_ID:$LATENT_ID" --export="$COMPARE_EXPORT_STR" Variations/slurm/evaluate_latent_mdn.slurm)
echo "Submitted comparison job (after all trainings): $COMPARE_ID"

echo "Done. Tail logs under $VARIATIONS_OUTPUT_ROOT/logs/ and Slurm -o paths in each .slurm header."
