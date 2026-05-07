# Partita

Partita is a small offline primitive-learning pipeline for RP1M. It is intentionally narrower than Sonata: instead of trying to discover primitives across the full RP1M repertoire, Partita learns an unsupervised primitive library from many successful trajectories of one song, then reconstructs a held-out or selected trajectory from that same song.

The first version is an oracle/nearest-primitive same-song primitive autoencoder. It does not train a transformer, diffusion model, or policy. The goal is to check whether repeated successful performances of one RP1M piece contain reusable motion/key-event chunks that can reconstruct another successful trajectory.

## Why One Song

RP1M is large, and cross-song primitive learning is hard to debug. Same-song primitive learning keeps the musical structure fixed while varying the demonstrations. If primitive reuse is meaningful here, it gives a clearer starting point for later Sonata-scale experiments.

French Suite No. 5 Sarabande is the preferred first target because the RP1M paper uses it as a clean exemplar trajectory. The default selector searches for related group names before falling back to other known RP1M examples or a small scan of available songs.

## First Experiment

The intended debug run is:

1. Inspect the RP1M Zarr root.
2. Select one preferred song.
3. Score all trajectories from that song.
4. Use the top successful trajectories for primitive learning.
5. Hold out the best trajectory by default as the reconstruction target.
6. Segment selected trajectories into short chunks.
7. Cluster segment features into a shared primitive library.
8. Reconstruct the target trajectory from nearest primitive centers.
9. Evaluate action and piano-state similarity.
10. Save compact reports and plots.

## Commands

From the repo root on WAVE, use the existing `sonata` conda environment:

```bash
cd /WAVE/projects/ECEN-524-Wi26/robopiano
conda activate sonata

python partita/scripts/inspect_rp1m.py \
  --rp1m-root /WAVE/users/unix/jlanders/rp1m_300/rp1m_repertoire.zarr \
  --max-songs 5

python partita/scripts/run_partita_debug.py \
  --config partita/configs/debug.yaml
```

Every stage is also runnable independently with `--config partita/configs/debug.yaml`.


## Online Rollout Video

After running the debug pipeline, render RoboPianist rollout videos with:

```bash
cd /WAVE/projects/ECEN-524-Wi26/robopiano
conda activate sonata
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export MUJOCO_GL=egl

python partita/scripts/simulate_rollout.py   --config partita/configs/debug.yaml   --which both   --width 640   --height 480
```

This writes videos and rollout JSON reports under `partita/outputs/rollout/<experiment_name>/`.
The rollout script synthesizes a RoboPianist MIDI/proto target from the RP1M `goals` pianoroll, because this checkout may not include the full PIG MIDI asset library. If the local environment exposes more action dimensions than RP1M, the replay pads the missing controls with zeros.

## Outputs

Outputs are written under `partita/outputs`:

- `inspection/rp1m_inspection.json`
- `data/<experiment_name>/trajectory_scores.csv`
- `data/<experiment_name>/selection.json`
- `data/<experiment_name>/selected_trajectories.npz`
- `primitives/<experiment_name>/segments.csv`
- `primitives/<experiment_name>/primitive_library.pkl`
- `primitives/<experiment_name>/primitive_summary.csv`
- `reconstruction/<experiment_name>/reconstructed_actions.npy`
- `evaluation/<experiment_name>/metrics.json`
- `evaluation/<experiment_name>/pianoroll_comparison.png`

## What To Look At

Useful first diagnostics:

- `key_f1`, `mispress_rate`, and `action_smoothness` in `trajectory_scores.csv`.
- Segment durations in `segment_duration_histogram.png`.
- Primitive counts and trajectory coverage in `primitive_summary.csv`.
- Primitive reuse in `primitive_usage_by_trajectory.csv` and `primitive_timeline_by_trajectory.png`.
- Reconstruction action MSE/L1 and key F1 in `metrics.json`.

A good first sign is that multiple primitives appear across many trajectories and no single primitive dominates the timeline. A bad sign is collapse into one primitive, mostly single-trajectory primitives, or extremely short segments.

## Current Limitations

- Offline only; no RoboPianist rollout yet.
- One song only.
- KMeans primitives only.
- Nearest-center reconstruction only.
- Piano-state reconstruction is an approximate primitive mean profile for diagnostics, not a real environment rollout.
- Metrics depend on RP1M arrays exposing `actions`; key metrics require `goals` and `piano_states`.
