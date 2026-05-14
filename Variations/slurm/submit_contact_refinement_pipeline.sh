#!/usr/bin/env bash
# Generate contact-refined labels from the current tuned checkpoints, then
# post-train all three Variations model families against those labels.
#
# This reuses the completed RP1M extraction. Set VARIATIONS_OUTPUT_ROOT to the
# tuned run you want to refine before launching, or leave the default below.
set -euo pipefail

ROBOPIANO_ROOT=/WAVE/projects/ECEN-524-Wi26/robopiano
cd "$ROBOPIANO_ROOT"

mkdir -p /WAVE/datasets/ccoelho_lab-jlanders/Variations/slurm_logs

RUN_NAME=${RUN_NAME:-variations_tuned_20260512_190752}
OUTPUT_ROOT=${OUTPUT_ROOT:-/WAVE/datasets/ccoelho_lab-jlanders/Variations/$RUN_NAME}
VARIATIONS_OUTPUT_ROOT=${VARIATIONS_OUTPUT_ROOT:-$OUTPUT_ROOT/variations}
CONTACT_ROOT=${CONTACT_ROOT:-$VARIATIONS_OUTPUT_ROOT/contact_refinement}

MLP_CHECKPOINT=${MLP_CHECKPOINT:-$VARIATIONS_OUTPUT_ROOT/mlp_baseline/joints_tuned/checkpoints/best.pt}
LATENT_MDN_CHECKPOINT=${LATENT_MDN_CHECKPOINT:-$VARIATIONS_OUTPUT_ROOT/latent_mdn/mdn/checkpoints/best.pt}
DIFFUSION_CHECKPOINT=${DIFFUSION_CHECKPOINT:-$VARIATIONS_OUTPUT_ROOT/diffusion/full_tuned/checkpoints/best.pt}

CONTACT_MAX_SAMPLES=${CONTACT_MAX_SAMPLES:-16025}
CONTACT_MAX_ITER=${CONTACT_MAX_ITER:-40}
POSTTRAIN_EPOCHS=${POSTTRAIN_EPOCHS:-25}

COMMON_EXPORT="ALL,RUN_NAME=$RUN_NAME,OUTPUT_ROOT=$OUTPUT_ROOT,VARIATIONS_OUTPUT_ROOT=$VARIATIONS_OUTPUT_ROOT,CONTACT_ROOT=$CONTACT_ROOT,CONTACT_MAX_SAMPLES=$CONTACT_MAX_SAMPLES,CONTACT_MAX_ITER=$CONTACT_MAX_ITER,POSTTRAIN_EPOCHS=$POSTTRAIN_EPOCHS"

submit_generate() {
  local model_type=$1
  local checkpoint=$2
  local output=$3
  sbatch --parsable \
    --job-name="contact_labels_${model_type}" \
    --partition=gpu \
    --gres=gpu:1 \
    --cpus-per-task=8 \
    --mem=64G \
    --time=1-00:00:00 \
    --chdir="$ROBOPIANO_ROOT" \
    --output="/WAVE/datasets/ccoelho_lab-jlanders/Variations/slurm_logs/contact_labels_${model_type}_%j.out" \
    --error="/WAVE/datasets/ccoelho_lab-jlanders/Variations/slurm_logs/contact_labels_${model_type}_%j.err" \
    --export="$COMMON_EXPORT" \
    --wrap="source ~/.bashrc && conda activate sonata && export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\${LD_LIBRARY_PATH:-} && python Variations/scripts/generate_contact_refined_labels.py --config Variations/configs/eval_press_pose.yaml --model-type '$model_type' --checkpoint '$checkpoint' --output '$output' --max-samples '$CONTACT_MAX_SAMPLES' --max-iter '$CONTACT_MAX_ITER' --device auto"
}

submit_posttrain() {
  local model_type=$1
  local checkpoint=$2
  local labels=$3
  local output=$4
  local dependency=$5
  sbatch --parsable \
    --job-name="contact_train_${model_type}" \
    --partition=gpu \
    --gres=gpu:1 \
    --cpus-per-task=8 \
    --mem=64G \
    --time=0-08:00:00 \
    --chdir="$ROBOPIANO_ROOT" \
    --output="/WAVE/datasets/ccoelho_lab-jlanders/Variations/slurm_logs/contact_train_${model_type}_%j.out" \
    --error="/WAVE/datasets/ccoelho_lab-jlanders/Variations/slurm_logs/contact_train_${model_type}_%j.err" \
    --dependency="afterok:$dependency" \
    --export="$COMMON_EXPORT" \
    --wrap="source ~/.bashrc && conda activate sonata && export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\${LD_LIBRARY_PATH:-} && python Variations/scripts/posttrain_contact_refinement.py --model-type '$model_type' --checkpoint '$checkpoint' --labels '$labels' --output '$output' --epochs '$POSTTRAIN_EPOCHS' --batch-size 256 --learning-rate 3e-5 --device auto"
}

mkdir -p "$CONTACT_ROOT"

MLP_LABELS=$CONTACT_ROOT/mlp_contact_labels.npz
LATENT_LABELS=$CONTACT_ROOT/latent_mdn_contact_labels.npz
DIFFUSION_LABELS=$CONTACT_ROOT/diffusion_contact_labels.npz

MLP_LABEL_JOB=$(submit_generate mlp_baseline "$MLP_CHECKPOINT" "$MLP_LABELS")
LATENT_LABEL_JOB=$(submit_generate latent_mdn "$LATENT_MDN_CHECKPOINT" "$LATENT_LABELS")
DIFFUSION_LABEL_JOB=$(submit_generate diffusion "$DIFFUSION_CHECKPOINT" "$DIFFUSION_LABELS")

MLP_CONTACT_CKPT=$CONTACT_ROOT/mlp_baseline_contact.pt
LATENT_CONTACT_CKPT=$CONTACT_ROOT/latent_mdn_contact.pt
DIFFUSION_CONTACT_CKPT=$CONTACT_ROOT/diffusion_contact.pt

MLP_TRAIN_JOB=$(submit_posttrain mlp_baseline "$MLP_CHECKPOINT" "$MLP_LABELS" "$MLP_CONTACT_CKPT" "$MLP_LABEL_JOB")
LATENT_TRAIN_JOB=$(submit_posttrain latent_mdn "$LATENT_MDN_CHECKPOINT" "$LATENT_LABELS" "$LATENT_CONTACT_CKPT" "$LATENT_LABEL_JOB")
DIFFUSION_TRAIN_JOB=$(submit_posttrain diffusion "$DIFFUSION_CHECKPOINT" "$DIFFUSION_LABELS" "$DIFFUSION_CONTACT_CKPT" "$DIFFUSION_LABEL_JOB")

COMPARE_JOB=$(sbatch --parsable \
  --job-name="contact_compare" \
  --partition=gpu \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=64G \
  --time=0-04:00:00 \
  --chdir="$ROBOPIANO_ROOT" \
  --output="/WAVE/datasets/ccoelho_lab-jlanders/Variations/slurm_logs/contact_compare_%j.out" \
  --error="/WAVE/datasets/ccoelho_lab-jlanders/Variations/slurm_logs/contact_compare_%j.err" \
  --dependency="afterok:$MLP_TRAIN_JOB:$LATENT_TRAIN_JOB:$DIFFUSION_TRAIN_JOB" \
  --export="$COMMON_EXPORT" \
  --wrap="source ~/.bashrc && conda activate sonata && export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\${LD_LIBRARY_PATH:-} && python Variations/scripts/evaluate_latent_mdn.py --config Variations/configs/eval_press_pose.yaml --latent-mdn-checkpoint '$LATENT_CONTACT_CKPT' --mlp-checkpoint '$MLP_CONTACT_CKPT' --diffusion-checkpoint '$DIFFUSION_CONTACT_CKPT' --output '$CONTACT_ROOT/contact_comparison_metrics.csv' --fingertip-eval-max-samples 256 --fingertip-settle-steps 5")

ONLINE_JOB=$(sbatch --parsable \
  --job-name="contact_online_cmp" \
  --partition=gpu \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=64G \
  --time=0-04:00:00 \
  --chdir="$ROBOPIANO_ROOT" \
  --output="/WAVE/datasets/ccoelho_lab-jlanders/Variations/slurm_logs/contact_online_cmp_%j.out" \
  --error="/WAVE/datasets/ccoelho_lab-jlanders/Variations/slurm_logs/contact_online_cmp_%j.err" \
  --dependency="afterok:$MLP_TRAIN_JOB:$LATENT_TRAIN_JOB:$DIFFUSION_TRAIN_JOB" \
  --export="$COMMON_EXPORT" \
  --wrap="source ~/.bashrc && conda activate sonata && export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\${LD_LIBRARY_PATH:-} && python Variations/scripts/evaluate_online_rollout.py --mlp-checkpoint '$MLP_CONTACT_CKPT' --latent-mdn-checkpoint '$LATENT_CONTACT_CKPT' --diffusion-checkpoint '$DIFFUSION_CONTACT_CKPT' --output-root '$CONTACT_ROOT/online_evaluation' --diffusion-steps 75 --midi-selection shortest --max-duration-s 10 --batch-size 256")

cat > "$CONTACT_ROOT/submission_ids.txt" <<EOF
RUN_NAME=$RUN_NAME
VARIATIONS_OUTPUT_ROOT=$VARIATIONS_OUTPUT_ROOT
CONTACT_ROOT=$CONTACT_ROOT
MLP_LABEL_JOB=$MLP_LABEL_JOB
LATENT_LABEL_JOB=$LATENT_LABEL_JOB
DIFFUSION_LABEL_JOB=$DIFFUSION_LABEL_JOB
MLP_TRAIN_JOB=$MLP_TRAIN_JOB
LATENT_TRAIN_JOB=$LATENT_TRAIN_JOB
DIFFUSION_TRAIN_JOB=$DIFFUSION_TRAIN_JOB
COMPARE_JOB=$COMPARE_JOB
ONLINE_JOB=$ONLINE_JOB
MLP_CONTACT_CHECKPOINT=$MLP_CONTACT_CKPT
LATENT_CONTACT_CHECKPOINT=$LATENT_CONTACT_CKPT
DIFFUSION_CONTACT_CHECKPOINT=$DIFFUSION_CONTACT_CKPT
CONTACT_COMPARISON_OUTPUT=$CONTACT_ROOT/contact_comparison_metrics.csv
CONTACT_ONLINE_OUTPUT=$CONTACT_ROOT/online_evaluation
EOF

cat "$CONTACT_ROOT/submission_ids.txt"
