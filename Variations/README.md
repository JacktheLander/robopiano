# Variations: RP1M Press Pairs and Conditional Diffusion

Variations builds a pointwise successful-press dataset from RP1M and trains a conditional diffusion model for:

```text
target_keys[88] -> hand_state[76]
```

`target_keys` is the binary piano-key goal at an onset frame. `hand_state` is `hand_joints[46]` concatenated with `hand_fingertips[30]`. The sustain pedal channel is intentionally dropped.

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

## Outputs

By default extraction writes to:

```text
Variations/outputs/extraction/<profile>/
```

Set `VARIATIONS_OUTPUT_ROOT` to move large outputs outside the repo:

```bash
export VARIATIONS_OUTPUT_ROOT=/WAVE/datasets/ccoelho_lab-jlanders/$RUN_NAME/variations
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

CPU extraction:

```bash
cd /WAVE/projects/ECEN-524-Wi26/robopiano
conda activate sonata
export RUN_NAME=variations_$(date +%Y%m%d_%H%M%S)
export RP1M_ROOT=/WAVE/users/unix/jlanders/rp1m_300/rp1m_repertoire.zarr
export VARIATIONS_OUTPUT_ROOT=/WAVE/datasets/ccoelho_lab-jlanders/$RUN_NAME/variations
sbatch Variations/slurm/extract_press_pairs.slurm
```

GPU training:

```bash
cd /WAVE/projects/ECEN-524-Wi26/robopiano
conda activate sonata
export RUN_NAME=<same-run-name>
export VARIATIONS_OUTPUT_ROOT=/WAVE/datasets/ccoelho_lab-jlanders/$RUN_NAME/variations
sbatch Variations/slurm/train_diffusion.slurm
```

## Notes

- Splits are song-level, deterministic, and generated from `manifest.csv`.
- Normalization stats are computed from train-split NPZs only.
- `target_keys` are not normalized.
- This is pointwise diffusion, not sequence diffusion or simulator rollout evaluation.

