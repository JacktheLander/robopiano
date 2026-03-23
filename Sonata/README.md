# Sonata-3
Segmented GMR -> Factored Planner -> Diffusion for RoboPianist

Sonata-3 is a research prototype for robot piano imitation learning on RoboPianist and RP1M. The policy is explicitly hierarchical:

1. Primitive learning with segmented GMM/GMR.
2. Goal-conditioned primitive planning with a factored causal transformer.
3. Low-level action refinement with a conditional diffusion model.

The implementation is designed for incremental use on HPC: Stage 1 writes compact cached datasets on CPU-oriented storage, then Stage 2 and Stage 3 reuse those cached outputs on GPU jobs without recomputing segmentation or primitive discovery.

## Repository Layout

```text
Sonata/
  README.md
  TODO_HPC.md
  pyproject.toml
  requirements.txt
  configs/
    data/
    primitive/
    transformer/
    diffusion/
    pipeline/
  scripts/
    prepare_rp1m.py
    train_primitives.py
    train_transformer.py
    train_diffusion.py
    train_joint_refine.py
    evaluate.py
    visualize_primitives.py
    visualize_rollout.py
    run_pipeline.py
  src/sonata/
    data/
    primitives/
    transformer/
    diffusion/
    models/
    training/
    evaluation/
    utils/
  outputs/
```

## Setup

Use the existing RoboPianist environment, then install Sonata-3 in editable mode:

```bash
cd /home/jackthelander/robopianist/Sonata
pip install -e .
```

If you need the optional score parsing and MJX tooling:

```bash
pip install -e ".[full]"
```

## Data Flow

Stage 0 writes a cached RP1M manifest and song-level splits:

- `outputs/data/<profile>/dataset_manifest.csv`
- `outputs/data/<profile>/dataset_splits.csv`
- `outputs/data/<profile>/dataset_summary.json`

Stage 1 consumes the manifest and writes:

- segment metadata tables in `outputs/primitives/<profile>/segments/`
- chunked slim cache artifacts in `outputs/primitives/<profile>/slim/`
- feature bundles in `outputs/primitives/<profile>/features/`
- GMM assignments in `outputs/primitives/<profile>/clustering/`
- GMR priors and primitive summaries in `outputs/primitives/<profile>/library/`
- token tables in `outputs/primitives/<profile>/tokens/`
- stage metrics in `outputs/primitives/<profile>/metrics/`
- stage plots in `outputs/primitives/<profile>/plots/`

The slim cache keeps only:

- segment index / metadata rows
- clustering feature vectors
- one fixed-horizon GMR target per segment

The legacy raw `segment_chunk_*.npz` bundles are optional debug artifacts. By default Sonata migrates existing raw chunks into the slim cache and continues new segmentation directly in slim mode.

Stage 2 consumes primitive tokens and primitive-library summaries, derives planner families online, and writes run directories under `outputs/transformer/`.

Stage 3 consumes primitive tokens, GMR priors, and a Stage 2 checkpoint, then writes run directories under `outputs/diffusion/`.

Evaluation writes offline metrics into `<output-root>/offline/` and optional rollout outputs into `<output-root>/rollout/`.

## Architecture

### Stage 1: Segmented GMR primitives

- Segmenters are pluggable: `fixed_window`, `changepoint`, `note_aligned`, `dtw_assisted`.
- Features combine joint summaries, action summaries, score context, contact summaries, and resampled trajectory traces.
- Primitive discovery uses a GMM sweep over candidate `K`, chosen by BIC or AIC.
- Each primitive fits a phase-conditioned GMR prior and exports a reusable coarse trajectory template.
- Tokens are explicit and inspectable: primitive id, primitive index, duration bucket, dynamics bucket, and score context JSON.
- Segment metadata keeps `chunk_path` / `chunk_index` pointed at the slim cache. If `write_full_segment_cache: true`, the optional raw debug chunk location is preserved in `raw_chunk_path` / `raw_chunk_index`.

### Stage 2: Factored goal-conditioned planner

The old Stage 2 baseline treated primitive planning as plain next-token prediction over primitive ids. That was too sensitive to primitive imbalance: dominant press-like primitives could win the cross-entropy objective without learning useful planning structure.

The current Stage 2 planner keeps causal autoregression over primitive histories, but predicts the next primitive in a factored way:

- `primitive_family`: a deterministic coarse family derived online from Stage 1 metadata and primitive-library statistics
- `primitive_id`: predicted within the family using a family-conditioned classifier and family-aware masking
- `duration_bucket`
- `dynamics_bucket`
- optional continuous primitive parameters from cached Stage 1 metadata
- `plan_embedding`: a fused embedding for Stage 3 diffusion conditioning

Planner inputs:

- primitive history
- primitive-family history
- duration and dynamics histories
- history context from prior segments
- explicit current goal context from score histogram, future density, active ratio, chord proxy, key center, and state summary

Planner outputs:

- family logits
- primitive logits
- duration logits
- dynamics logits
- optional continuous parameter regression
- structured plan embedding

The planner uses a shared causal transformer backbone, then separate heads for each prediction target. The plan embedding is not just the last hidden state: it fuses planner state, goal context, and predicted family / primitive / duration / dynamics intents before projection.

### Imbalance robustness

The planner includes configurable controls for highly imbalanced primitive vocabularies:

- focal loss on selected heads
- class-balanced weighting
- family-aware balanced sampling
- label smoothing
- weighted factored losses
- temperature-scaled evaluation
- top-k metrics per head
- family macro-F1 and balanced accuracy

### Stage 3: Diffusion refiner

- 1D conditional denoiser over short action chunks.
- Conditions: state context, goal context, primitive token scalars, and either the factored planner embedding or a primitive embedding fallback.
- Variants:
  - `full`
  - `planner_no_prior`
  - `diffusion_only`
  - `gmr_only`

### Stage 4: Integration

`sonata.models.pipeline.Sonata3Pipeline` loads primitive priors, the factored planner, and the diffusion model together for offline inference and rollout evaluation.

## Stage 2 training and evaluation

Recommended planner config:

- `configs/transformer/debug.yaml` for local smoke tests
- `configs/transformer/medium.yaml` for medium-scale GPU runs
- `configs/transformer/full.yaml` for full RP1M/HPC runs

Train the planner:

```bash
python scripts/train_transformer.py --profile debug --no-wandb
```

Train diffusion against the planner checkpoint:

```bash
python scripts/train_diffusion.py \
  --profile debug \
  --planner-checkpoint /path/to/outputs/transformer/<run>/checkpoints/best.pt \
  --no-wandb
```

Evaluate the full planner + diffusion stack offline:

```bash
python scripts/evaluate.py \
  --primitive-root outputs/primitives/debug \
  --diffusion-checkpoint /path/to/outputs/diffusion/<run>/checkpoints/best.pt \
  --output-root outputs/eval/debug
```

Planner validation metrics are written during training into:

- `outputs/transformer/<run>/metrics/metrics.csv`
- `outputs/transformer/<run>/metrics/metrics.jsonl`
- `outputs/transformer/<run>/artifacts/generated_sequences.csv`
- `outputs/transformer/<run>/artifacts/family_confusion_best.csv`
- `outputs/transformer/<run>/artifacts/primitive_family_mapping.csv`
- `outputs/transformer/<run>/artifacts/planner_metadata.json`

## Key Stage 2 config fields

Planner architecture:

- `model_variant`
- `plan_embedding_dim`
- `context_length`
- `family_mapping_mode`
- `continuous_param_names`

Factored loss weights:

- `family_loss_weight`
- `primitive_loss_weight`
- `duration_loss_weight`
- `dynamics_loss_weight`
- `param_loss_weight`
- `normalize_loss_by_active_weights`

Imbalance controls:

- `use_focal_loss`
- `focal_heads`
- `focal_gamma`
- `use_class_balanced_loss`
- `class_balance_strategy`
- `class_balance_beta`
- `class_weight_power`
- `class_weight_max`
- `use_balanced_sampler`
- `balanced_sampler_target`
- `label_smoothing`
- `eval_temperature`
- `topk`

## Migration notes

- The shipped transformer configs now default to `model_variant: factored_goal_conditioned`.
- Old Stage 2 checkpoints trained with the plain `token_prediction` planner are intentionally treated as incompatible. Retrain Stage 2 with the new configs before using those checkpoints in Stage 3.
- Diffusion checkpoints now store the resolved planner config and planner metadata so offline inference does not rely on silently re-inferring the old planner shape.
- Stage 1 outputs do not need to be regenerated for the planner redesign. Primitive families are derived deterministically inside Stage 2 / Stage 3 loading from cached Stage 1 token and library metadata.

## RP1M Assumptions

The current implementation assumes the RP1M dataset follows the layout already present in this workspace:

- Main storage is a Zarr archive with per-song groups named like `RoboPianist-...-v0_0`.
- Each group may contain arrays such as:
  - `actions`
  - `goals`
  - `piano_states`
  - `hand_joints`
  - `hand_fingertips`
  - optionally `joint_velocities`, `wrist_pose`, `hand_pose`
- The first array dimension is trajectories, the second is time.
- Flat `.npy` trajectory directories are also supported.
- Score files are matched by environment stem from:
  - `extracted_midis/`
  - `robopianist/music/data/`
- If `.proto` or MIDI files are missing, score events fall back to `goals`, then `piano_states`.

If score files are missing, segmentation falls back to `goals` and then `piano_states`; if neither is present, score-conditioned segmentation metadata becomes empty rather than forcing a crash.

## Recommended Usage

Debug run on the toy dataset:

```bash
python scripts/prepare_rp1m.py --profile debug
python scripts/train_primitives.py --profile debug
python scripts/train_transformer.py --profile debug --no-wandb
python scripts/train_diffusion.py --profile debug --planner-checkpoint /path/to/best.pt --no-wandb
```

If you already have legacy `segment_chunk_*.npz` files and want to materialize the slim cache before rerunning Stage 1:

```bash
python scripts/migrate_segment_chunks.py --profile full
```

To migrate existing raw chunks, continue the remaining segmentation directly in slim mode, and finish Stage 1 end-to-end:

```bash
python scripts/train_primitives.py --profile full
```

W&B is enabled by default in the shipped Sonata configs. Disable sync per run with `--no-wandb`, or switch to local buffering with `--wandb-mode offline`.

End-to-end debug pipeline:

```bash
python scripts/run_pipeline.py --profile debug --no-wandb
```

## Config Profiles

- `configs/data/debug.yaml`, `medium.yaml`, `full.yaml`
- `configs/primitive/debug.yaml`, `medium.yaml`, `full.yaml`
- `configs/transformer/debug.yaml`, `medium.yaml`, `full.yaml`
- `configs/diffusion/debug.yaml`, `medium.yaml`, `full.yaml`
- `configs/pipeline/debug.yaml`, `medium.yaml`, `full.yaml`

`medium` and `full` resolve `dataset_root` from the `RP1M_300_ROOT` environment variable. On the cluster, export it before launching:

```bash
export RP1M_300_ROOT=/project/$USER/rp1m_300.zarr
```

If cluster score files live outside the repo checkout, you can point `note_search_roots` at environment-expanded paths because Sonata resolves `$VARS` in config paths before use.

Primitive configs also expose the slim-cache migration knobs documented in `STAGE1_ONLINE_PRIMITIVES.md`.

## Evaluation Outputs

Stage 1:

- cluster count sweep
- primitive frequency plots
- GMR reconstruction metrics
- primitive reuse / entropy metrics
- token tables and primitive vocabulary metadata

Stage 2:

- overall loss plus per-head losses
- family / primitive / duration / dynamics accuracy
- primitive and family top-k accuracy
- family macro-F1 and balanced accuracy
- family / primitive entropy and confidence diagnostics
- primitive-family mapping artifacts and confusion matrix
- `best.pt` plus periodic `epoch_XXXX.pt` checkpoints

Stage 3 and full pipeline:

- action MSE / L1
- smoothness
- improvement over GMR prior
- optional DM Control rollout reward and note metrics
- optional MJX physics rollout metadata
- `best.pt` plus periodic `epoch_XXXX.pt` checkpoints

## Logging And Checkpoints

- Primitive discovery writes stage outputs directly under `outputs/primitives/<profile>/`.
- Transformer and diffusion use timestamped run directories.
- Each run contains:
  - `checkpoints/`
  - `metrics/metrics.csv`
  - `metrics/metrics.jsonl`
  - `artifacts/`
  - `plots/`
  - `logs/`
- `resume: true` reuses the latest matching timestamped run directory for transformer/diffusion and restores the most recent `*.pt` checkpoint found there.
- W&B logging mirrors run config, scalar metrics, summaries, checkpoint artifacts, and run-output directories when enabled in config or via CLI.

## Slim Cache Migration

Use this flow when a profile already contains legacy `segment_chunk_*.npz` outputs:

1. Run `python scripts/migrate_segment_chunks.py --profile <profile>` to scan each raw chunk, rebuild the exact Stage 1 features, store the pre-resampled GMR target, and write resumable per-chunk slim manifests under `outputs/primitives/<profile>/slim/`.
2. Run `python scripts/train_primitives.py --profile <profile>` to continue any remaining segmentation directly in slim mode and then finish clustering, GMR fitting, and tokenization.
3. Add `--delete-raw` to the migration script, or set `delete_raw_chunks_after_migration: true`, when you want Sonata to remove each raw chunk after the slim outputs have been written and verified.

Migration is idempotent and resumable: completed slim chunks are skipped on rerun, new segmentation appends new slim chunks, and the consolidated `segments/segment_index.csv` is rebuilt from the migrated slim records plus any remaining legacy rows.

## Target Environment

- `prepare_rp1m.py` and `train_primitives.py` are the CPU-heavy stages.
- `train_transformer.py` and `train_diffusion.py` consume cached Stage 1 outputs rather than recomputing segmentation.
- `run_pipeline.py` is convenient for local debugging and end-to-end smoke runs, but on the cluster the more robust pattern is: run Stage 0 and Stage 1 on shared CPU storage first, then launch Stage 2 and Stage 3 jobs against those cached outputs.
- The code does not assume Slurm locally. Cluster-specific scheduling remains outside Sonata itself; Sonata focuses on explicit filesystem outputs so CPU and GPU jobs can hand off work through shared storage cleanly.

## Notes On MJX

The MJX module in this repo currently accelerates MuJoCo physics stepping, not the full RoboPianist task stack. Observation/reward parity with dm_control still needs task-specific glue. The remaining work is listed in `TODO_HPC.md`.
