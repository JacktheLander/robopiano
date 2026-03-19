# AGENTS.md

## Environment

- Runs on SLURM cluster
- CPU nodes: up to 2TB RAM
- GPU nodes: NVIDIA V100
- Storage:
  - /project/$USER for persistent data
  - /scratch/$USER for cache/temp

## Pipeline Strategy

### Stage 1: Preprocessing (CPU node)
- Load RP1M dataset
- Generate:
  - segments (16x39)
  - primitive assignments (GMM)
  - token sequences
  - diffusion windows
- Save outputs as compact datasets (numpy / zarr / torch)

### Stage 2: Training (GPU node)
- Transformer + diffusion trained from cached datasets
- Avoid recomputing segmentation
- Use PyTorch with CUDA

## Constraints

- You are operating on a local machine preparing the repo for hpc
- Do NOT load full expanded dataset into memory unless explicitly requested
- Prefer streaming or chunked datasets
- Avoid recomputing primitives each run
- Minimize disk I/O bottlenecks

## Commands

### Local development
- Install deps: `pip install -r requirements.txt`
- Run unit tests: `pytest -q`
- Run small smoke training only: `python scripts/train.py --config configs/smoke.yaml`
- Never run full-dataset training locally.

## Environment rules
- Local machine is for editing, debugging, and smoke tests.
- HPC is for full RP1M preprocessing and full training.
- Do not assume Slurm exists locally.
- Do not assume local paths exist on HPC.