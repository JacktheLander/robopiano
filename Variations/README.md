# Variations: RP1M Press Pairs and Conditional Diffusion

Variations builds a pointwise successful-press dataset from RP1M and trains a conditional diffusion model for:

```text
target_keys[88] -> joint_state[46]
```

`target_keys` is the binary piano-key goal at an onset frame. The model target is only `hand_joints[46]`. Extracted NPZ files may still contain the raw RP1M `hand_state` field for compatibility, but the dataset, normalization stats, losses, training, inference, and evaluation slice it to the first 46 joint variables.

## Extraction Policy

A successful press row is emitted only when all of the following are true at timestep `t`:

- `piano_states[t, :88] == goals[t, :88]` after thresholding.
- At least one piano key transitions from off to on at `t`.
- At least one goal key is active.

Extraction visits trajectories in round-robin rank order across songs:

1. Select songs with deterministic stride sampling.
2. Score a fixed random sample of trajectories per song.
3. Keep each song's best `top_k_trajectories`.
4. Process rank 0 for every song before rank 1 for any song.

Duplicates are removed globally by exact `target_keys` fingerprint. The first accepted hand pose for a key pattern wins; later rows with the same 88-bit goal pattern are skipped, even if they come from another song or trajectory.

## Local Commands

Run from the repo root:

```bash
python Variations/scripts/inspect_rp1m.py --rp1m-root /WAVE/users/unix/jlanders/rp1m_300/rp1m_repertoire.zarr
python Variations/scripts/extract_press_pairs.py --config Variations/configs/extraction/debug.yaml
python Variations/scripts/summarize_dataset.py --extraction-root Variations/outputs/extraction/debug
python Variations/scripts/build_splits.py --extraction-root Variations/outputs/extraction/debug --val-fraction 0.1 --seed 42 --min-pairs-per-split 1
python Variations/scripts/compute_norm_stats.py --extraction-root Variations/outputs/extraction/debug
python Variations/scripts/train_diffusion.py --config Variations/configs/diffusion/debug.yaml --no-wandb
python Variations/scripts/evaluate_diffusion.py --config Variations/configs/diffusion/debug.yaml --checkpoint Variations/outputs/diffusion/debug/checkpoints/best.pt
```

`train_diffusion.py` creates missing song-level splits and train-only normalization stats before training.

## MAESTRO simulation (RoboPianist video)

Run a checkpoint on an unseen MAESTRO MIDI (quantized keysets per control step) and record a simulator video using the same **state injection** strategy as Partita recorded playback:

```bash
python Variations/simulate/simulate_maestro.py \
  --model-type diffusion \
  --checkpoint Variations/outputs/diffusion/debug/checkpoints/best.pt \
  --maestro-root /WAVE/datasets/ccoelho_lab-jlanders/MAESTRO \
  --piece-index 0 \
  --output-root /WAVE/datasets/ccoelho_lab-jlanders/Variations/simulation
```

See [Variations/simulate/README.md](simulate/README.md) for flags and output layout.

## Supervised MLP Baselines

The diffusion model remains available as the conditional model:

```text
p(joint_state[46] | target_keys[88])
```

The supervised baseline is available for direct comparison:

```bash
python Variations/scripts/train_mlp_baseline.py --config Variations/configs/mlp_baseline_debug.yaml --no-wandb
```

The MLP learns `target_keys[88] -> joint_state[46]`. It reuses the same extracted NPZs, song-level splits, and train-only `norm_stats.npz` as diffusion.

Compare trained checkpoints with (paths assume **debug** extraction and training outputs):

```bash
python Variations/scripts/evaluate_press_pose_models.py \
  --config Variations/configs/eval_press_pose.yaml \
  --mlp-checkpoint Variations/outputs/mlp_baseline/joints_debug/checkpoints/best.pt \
  --diffusion-checkpoint Variations/outputs/diffusion/debug/checkpoints/best.pt
```

If extraction lives under `VARIATIONS_OUTPUT_ROOT` or another folder, pass `--extraction-root /path/to/extraction/debug`.

For **full** RP1M extraction, set `extraction_root` in `eval_press_pose.yaml` to `Variations/outputs/extraction/full` (or your output tree) and raise `min_pairs_per_split` only after that extraction has finished.

The comparison CSV reports normalized joint MSE, denormalized joint MSE, parameter count, and inference time per sample.

Run the three Variations model families through the same headless online RoboPianist rollout scorer used by Intermezzo:

```bash
python Variations/scripts/evaluate_online_rollout.py \
  --mlp-checkpoint /WAVE/datasets/ccoelho_lab-jlanders/Variations/<run>/variations/mlp_baseline/joints/checkpoints/best.pt \
  --latent-mdn-checkpoint /WAVE/datasets/ccoelho_lab-jlanders/Variations/<run>/variations/latent_mdn/mdn/checkpoints/best.pt \
  --diffusion-checkpoint /WAVE/datasets/ccoelho_lab-jlanders/Variations/<run>/variations/diffusion/checkpoints/best.pt \
  --midi-selection shortest \
  --max-duration-s 10
```

If checkpoint paths are omitted, the evaluator searches `/WAVE/datasets/ccoelho_lab-jlanders/Variations` and uses the most recent `best.pt` for each model type. Outputs are written under `/WAVE/datasets/ccoelho_lab-jlanders/Variations/online_evaluation` with per-model rollout JSON/NPZ files plus a combined `summary.json`.

Merge evaluation CSVs from different suites and plot grouped bars:

```bash
python Variations/scripts/plot_press_pose_comparison.py \
  --csv-suite Variations/outputs/comparisons/mlp_vs_diffusion_metrics.csv "debug_extract_local" \
  --csv-suite /WAVE/datasets/ccoelho_lab-jlanders/Variations/<your_run>/variations/latent_mdn/comparisons/rp1m_val_mlp_vs_latent_mdn.csv "RP1M_full_val" \
  --output-png Variations/outputs/comparisons/press_pose_comparison.png \
  --merged-csv Variations/outputs/comparisons/press_pose_comparison_merged.csv
```

The latent MDN is a probabilistic baseline over a learned press-pose manifold:

```text
target_keys[88] -> mixture over z -> joint decoder -> joint_state[46]
```

Train the two-phase pipeline with:

```bash
python Variations/scripts/train_latent_mdn.py --config Variations/configs/latent_mdn.yaml
```

This first trains `PoseAutoencoder(joint_state[46] -> z -> joint_state[46])`, computes train-split latent normalization stats, then freezes the autoencoder and trains `LatentMDN(target_keys[88] -> p(z))`.

Evaluate against direct MLP and diffusion with:

```bash
python Variations/scripts/evaluate_latent_mdn.py \
  --config Variations/configs/eval_press_pose.yaml \
  --latent-mdn-checkpoint Variations/outputs/latent_mdn/mdn/checkpoints/best.pt \
  --mlp-checkpoint Variations/outputs/mlp_baseline/joints/checkpoints/best.pt \
  --diffusion-checkpoint Variations/outputs/diffusion/full/checkpoints/best.pt
```

## FingerPred Active-Fingertip Model

FingerPred is an additive Variations model family. It does not replace the joint-pose MLP, diffusion, or latent MDN runs. It uses the latent-MDN two-stage architecture for:

```text
target_keys[88] -> active hand_fingertips[30]
```

RP1M does not expose explicit fingering labels, so FingerPred derives an `active_tip_mask[10]` by assigning each pressed key to the nearest observed fingertip at the accepted press frame. The model outputs all ten xyz fingertips, but training and primary evaluation metrics are masked to active fingertip coordinates only. FingerPred checkpoints predict fingertip positions, not 46D hand joints, so they are intentionally rejected by the MAESTRO simulation and online rollout tools.

Train with:

```bash
python Variations/scripts/train_fingerpred.py --config Variations/configs/fingerpred.yaml
```

Evaluate with:

```bash
python Variations/scripts/evaluate_fingerpred.py \
  --config Variations/configs/fingerpred.yaml \
  --checkpoint Variations/outputs/fingerpred/fingerpred_active_tips_latent16_k3/mdn/checkpoints/best.pt
```

On Slurm:

```bash
sbatch Variations/slurm/train_fingerpred.slurm
```

## Outputs

By default extraction writes to:

```text
Variations/outputs/extraction/<profile>/
```

Set `VARIATIONS_OUTPUT_ROOT` to move large outputs outside the repo:

```bash
export VARIATIONS_OUTPUT_ROOT=/WAVE/datasets/ccoelho_lab-jlanders/Variations/$RUN_NAME/variations
```

The extraction output contains:

- `song_<safe_song_id>.npz` with accepted rows for one originating song.
- `manifest.csv` with candidate, accepted, and duplicate-skip counts.
- `summary.json` with aggregate counts and policy metadata.
- `extraction_state/seen_goal_fp.pkl` and `resume.json` for crash-safe resume.

Diffusion runs write to:

```text
Variations/outputs/diffusion/<run_name>/
```

with `checkpoints/best.pt`, `checkpoints/last.pt`, metrics, config artifacts, and generated samples.

## HPC Commands

Use the standard WAVE workflow from `HowToRun.md`.

HPC outputs default under the lab Variations dataset tree (create the Slurm log dir once if needed):

```bash
mkdir -p /WAVE/datasets/ccoelho_lab-jlanders/Variations/slurm_logs
```

CPU extraction:

```bash
cd /WAVE/projects/ECEN-524-Wi26/robopiano
conda activate sonata
export RUN_NAME=variations_$(date +%Y%m%d_%H%M%S)
export RP1M_ROOT=/WAVE/users/unix/jlanders/rp1m_300/rp1m_repertoire.zarr
export VARIATIONS_OUTPUT_ROOT=/WAVE/datasets/ccoelho_lab-jlanders/Variations/$RUN_NAME/variations
sbatch Variations/slurm/extract_press_pairs.slurm
```

GPU training:

```bash
cd /WAVE/projects/ECEN-524-Wi26/robopiano
conda activate sonata
export RUN_NAME=<same-run-name>
export VARIATIONS_OUTPUT_ROOT=/WAVE/datasets/ccoelho_lab-jlanders/Variations/$RUN_NAME/variations
sbatch Variations/slurm/train_diffusion.slurm
```

## Notes

- Splits are song-level, deterministic, and generated from `manifest.csv`.
- Normalization stats are computed from train-split NPZs only.
- `target_keys` are not normalized.
- This is pointwise diffusion, not sequence diffusion or simulator rollout evaluation.
