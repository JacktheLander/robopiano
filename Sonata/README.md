# Sonata-3
Segmented GMR -> Transformer -> Diffusion for RoboPianist

Sonata-3 is a research prototype for robot piano imitation learning on RoboPianist and RP1M. The policy is explicitly hierarchical:

1. Primitive learning with segmented GMM/GMR.
2. High-level primitive sequencing with an autoregressive transformer.
3. Low-level action refinement with a conditional diffusion model.

The implementation is designed for incremental use on an HPC cluster: cached dataset manifests, chunked segment caches, stage-specific checkpoints, explicit artifact directories, and a workflow that keeps raw RP1M scanning/segmentation on CPU-oriented storage passes while reusing cached primitive outputs for later GPU training.

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

- chunked segment caches in `outputs/primitives/<profile>/segments/`
- feature bundles in `outputs/primitives/<profile>/features/`
- GMM assignments in `outputs/primitives/<profile>/clustering/`
- GMR priors and primitive summaries in `outputs/primitives/<profile>/library/`
- token tables in `outputs/primitives/<profile>/tokens/`
- stage metrics in `outputs/primitives/<profile>/metrics/`
- stage plots in `outputs/primitives/<profile>/plots/`

Stage 2 consumes primitive tokens and writes run directories under `outputs/transformer/`.

Stage 3 consumes primitive tokens, GMR priors, and a transformer checkpoint, then writes run directories under `outputs/diffusion/`.

Evaluation writes offline metrics into `<output-root>/offline/` and optional rollout outputs into `<output-root>/rollout/`.

## Architecture

### Stage 1: Segmented GMR primitives

- Segmenters are pluggable: `fixed_window`, `changepoint`, `note_aligned`, `dtw_assisted`.
- Features combine joint summaries, action summaries, score context, contact summaries, and resampled trajectory traces.
- Primitive discovery uses a GMM sweep over candidate `K`, chosen by BIC or AIC.
- Each primitive fits a phase-conditioned GMR prior and exports a reusable coarse trajectory template.
- Tokens are explicit and inspectable: primitive id, primitive index, duration bucket, dynamics bucket, and score context JSON.

### Stage 2: Transformer planner

- Autoregressive transformer over primitive histories.
- Inputs: primitive id, duration bucket, dynamics bucket, positional embedding, and score context projection.
- Outputs: next primitive, duration, and dynamics logits, plus a plan embedding for diffusion conditioning.
- Baseline A is included through `model_variant: direct_transformer_action`.

### Stage 3: Diffusion refiner

- 1D conditional denoiser over short action chunks.
- Conditions: state context, score context, primitive token identity or transformer embedding, and GMR prior.
- Variants:
  - `full`
  - `planner_no_prior`
  - `diffusion_only`
  - `gmr_only`

### Stage 4: Integration

`sonata.models.pipeline.Sonata3Pipeline` loads primitive priors, the planner, and the diffusion model together for offline inference and rollout evaluation.

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

These assumptions are encoded in [indexer.py](/home/jackthelander/robopianist/Sonata/src/sonata/data/indexer.py) and [score.py](/home/jackthelander/robopianist/Sonata/src/sonata/data/score.py). If score files are missing, segmentation falls back to `goals` and then `piano_states`; if neither is present, score-conditioned segmentation metadata becomes empty rather than forcing a crash.

## Recommended Usage

Debug run on the toy dataset:

```bash
python scripts/prepare_rp1m.py --profile debug
python scripts/train_primitives.py --profile debug
python scripts/train_transformer.py --profile debug
python scripts/train_diffusion.py --profile debug --planner-checkpoint /path/to/best.pt
```

W&B is enabled by default in the shipped Sonata configs. If you want uploads to the Santa Clara workspace, authenticate first:

```bash
wandb login
```

You can disable sync per run with `--no-wandb`, or switch to local buffering with `--wandb-mode offline`.

End-to-end debug pipeline:

```bash
python scripts/run_pipeline.py --profile debug
```

Offline evaluation:

```bash
python scripts/evaluate.py \
  --primitive-root outputs/primitives/debug \
  --diffusion-checkpoint /path/to/best.pt \
  --output-root outputs/eval/debug
```

DM Control rollout evaluation:

```bash
python scripts/evaluate.py \
  --primitive-root outputs/primitives/debug \
  --diffusion-checkpoint /path/to/best.pt \
  --output-root outputs/eval/debug \
  --backend dm_control
```

MJX physics smoke test:

```bash
python scripts/evaluate.py \
  --primitive-root outputs/primitives/debug \
  --diffusion-checkpoint /path/to/best.pt \
  --output-root outputs/eval/debug \
  --backend mjx_physics \
  --xml-path /path/to/model.xml
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

If cluster score files live outside the repo checkout, you can also point `note_search_roots` at environment-expanded paths because Sonata resolves `$VARS` in config paths before use. If no score files are available, segmentation falls back to `goals` and then `piano_states`.

## Evaluation Outputs

Stage 1:

- cluster count sweep
- primitive frequency plots
- GMR reconstruction metrics
- primitive reuse / entropy metrics
- token tables and primitive vocabulary metadata

Stage 2:

- loss, accuracy, top-k accuracy, perplexity
- generated primitive prediction tables
- `best.pt` plus periodic `epoch_XXXX.pt` checkpoints in the run directory

Stage 3 and full pipeline:

- action MSE / L1
- smoothness
- improvement over GMR prior
- optional DM Control rollout reward and note metrics
- optional MJX physics rollout metadata
- `best.pt` plus periodic `epoch_XXXX.pt` checkpoints in the run directory

## Logging And Checkpoints

- Primitive discovery writes stage outputs directly under `outputs/primitives/<profile>/`.
- Transformer and diffusion use timestamped run directories similar to `tin/train.py`.
- Each run contains:
  - `checkpoints/`
  - `metrics/metrics.csv`
  - `metrics/metrics.jsonl`
  - `artifacts/`
  - `plots/`
  - `logs/`
- `resume: true` reuses the latest matching timestamped run directory for transformer/diffusion and restores the most recent `*.pt` checkpoint found there.
- W&B logging mirrors run config, scalar metrics, summaries, checkpoint artifacts, and run-output directories when enabled in config or via CLI.
- The `logs/` directory is reserved as part of the run structure, but the current scripts primarily emit structured logs to stdout/stderr rather than automatically writing a persistent per-run log file there.

## Target Environment

- `prepare_rp1m.py` and `train_primitives.py` are the CPU-heavy stages. They scan RP1M, load episodes one at a time, segment trajectories, and save chunked `.npz` segment bundles so later stages avoid reopening the raw dataset repeatedly.
- `train_transformer.py` and `train_diffusion.py` consume the cached primitive outputs rather than rescanning RP1M. This is the intended GPU-node path.
- `run_pipeline.py` is convenient for local debugging and end-to-end smoke runs, but on the cluster the more robust pattern is: run Stage 0 and Stage 1 on shared CPU storage first, then launch Stage 2 and Stage 3 jobs against those cached outputs.
- The code does not assume Slurm locally. Cluster-specific scheduling remains outside Sonata itself; Sonata focuses on explicit filesystem outputs so CPU and GPU jobs can hand off work through shared storage cleanly.

## Notes On MJX

The MJX module in this repo currently accelerates MuJoCo physics stepping, not the full RoboPianist task stack. Observation/reward parity with dm_control still needs task-specific glue. The exact remaining work is listed in [TODO_HPC.md](/home/jackthelander/robopianist/Sonata/TODO_HPC.md).
