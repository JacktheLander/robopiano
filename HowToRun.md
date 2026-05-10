# Running Sonata on the WAVE HPC Environment

This guide outlines the standard workflow for running Sonata experiments on the WAVE HPC cluster. It covers working from the correct project directory, activating the correct Conda environment, launching Slurm jobs, using `tmux` to protect long-running jobs from SSH disconnects, and writing outputs to the shared dataset/output location.

---

## Project Directory

Work from the main `robopiano` project directory:

```bash
cd /WAVE/projects/ECEN-524-Wi26/robopiano
```

This should be the starting point for running Sonata scripts, accessing the local `Sonata/` directory, and referencing the included RoboPianist environment.

---

## Conda Environment

Before running any Sonata scripts, activate the correct Conda environment:

```bash
conda activate sonata
```

Use this environment for preprocessing, training, evaluation, and rollout scripts.

---

## Training Dataset

The RP1M dataset used for training should be read from:

```bash
/WAVE/users/unix/jlanders/rp1m_300/rp1m_repertoire.zarr
```

Recommended setup:

```bash
export RP1M_300_ROOT=/WAVE/users/unix/jlanders/rp1m_300/rp1m_repertoire.zarr
```

Many scripts expect the dataset root to be available through this environment variable.

---

## Output Directory

Large outputs should not be written into the project directory or home directory.

New experiment outputs should be written to a new folder under:

```bash
/WAVE/datasets/ccoelho_lab-jlanders
```

This location should also be used to inspect prior run data.

Example setup:

```bash
export RUN_NAME=sonata_run_$(date +%Y%m%d_%H%M%S)
export OUTPUT_ROOT=/WAVE/datasets/ccoelho_lab-jlanders/$RUN_NAME
mkdir -p $OUTPUT_ROOT
```

Suggested output subdirectories:

```bash
$OUTPUT_ROOT/primitives
$OUTPUT_ROOT/transformer
$OUTPUT_ROOT/diffusion
$OUTPUT_ROOT/evaluation
$OUTPUT_ROOT/logs
```

---

# Using `tmux` for Safe Long Runs

Long HPC jobs should be started inside a `tmux` session. This protects the run if the SSH connection drops or the local terminal closes.

Start a named `tmux` session:

```bash
tmux new -s sonata_run
```

Inside the session, move to the project directory and activate the environment:

```bash
cd /WAVE/projects/ECEN-524-Wi26/robopiano
conda activate sonata
export RP1M_300_ROOT=/WAVE/users/unix/jlanders/rp1m_300/rp1m_repertoire.zarr
```

After the Slurm allocation is active and the run has started, detach from `tmux` with:

```text
Ctrl+b, then d
```

This leaves the session running in the background. You can safely disconnect from SSH after detaching.

To reconnect later:

```bash
tmux attach -t sonata_run
```

To list active sessions:

```bash
tmux ls
```

To kill a finished session:

```bash
tmux kill-session -t sonata_run
```

---

# Slurm Run Types

There are two main Slurm partitions to use for Sonata work: `cmp` and `gpu`.

Choose the partition based on whether the job needs CUDA acceleration.

---

## 1. `cmp` Partition

Use `cmp` for CPU-heavy or non-GPU jobs.

Good for:

- Dataset indexing
- Preprocessing
- Stage 1 primitive extraction if CUDA is not required
- Offline evaluation
- Lightweight debugging
- File migration or cleanup scripts

Example interactive allocation:

```bash
srun -p cmp \
  --cpus-per-task=16 \
  --mem=128G \
  --time=0-08:00:00 \
  --pty bash
```

Adjust memory and time based on the expected job size.

For larger preprocessing jobs:

```bash
srun -p cmp \
  --cpus-per-task=24 \
  --mem=256G \
  --time=1-00:00:00 \
  --pty bash
```

---

## 2. `gpu` Partition

Use `gpu` for CUDA training runs.

Good for:

- Transformer training
- Diffusion training
- GPU-accelerated evaluation
- PyTorch training with CUDA
- Rollouts or model inference that benefit from GPU acceleration

Example interactive allocation:

```bash
srun -p gpu \
  --gres=gpu:1 \
  --cpus-per-task=16 \
  --mem=128G \
  --time=1-00:00:00 \
  --pty bash
```

For longer or heavier GPU runs:

```bash
srun -p gpu \
  --gres=gpu:1 \
  --cpus-per-task=24 \
  --mem=256G \
  --time=2-00:00:00 \
  --pty bash
```

Request enough time up front. If the Slurm time limit expires, the job will be killed even if the code is running correctly.

---

# Recommended Interactive Workflow

A safe standard workflow is:

1. Start a `tmux` session.
2. Request a Slurm allocation.
3. Move into the project directory.
4. Activate the `sonata` Conda environment.
5. Export the RP1M dataset path.
6. Create a new output folder under `/WAVE/datasets/ccoelho_lab-jlanders`.
7. Start the run, always enabling wandb where possible.
8. Detach from `tmux` once the run is confirmed to be active.

Example CPU workflow:

```bash
tmux new -s sonata_cpu

srun -p cmp \
  --cpus-per-task=16 \
  --mem=128G \
  --time=0-08:00:00 \
  --pty bash

cd /WAVE/projects/ECEN-524-Wi26/robopiano
conda activate sonata

export RP1M_300_ROOT=/WAVE/users/unix/jlanders/rp1m_300/rp1m_repertoire.zarr
export RUN_NAME=sonata_cpu_$(date +%Y%m%d_%H%M%S)
export OUTPUT_ROOT=/WAVE/datasets/ccoelho_lab-jlanders/$RUN_NAME
mkdir -p $OUTPUT_ROOT
```

Example GPU workflow:

```bash
tmux new -s sonata_gpu

srun -p gpu \
  --gres=gpu:1 \
  --cpus-per-task=16 \
  --mem=128G \
  --time=1-00:00:00 \
  --pty bash

cd /WAVE/projects/ECEN-524-Wi26/robopiano
conda activate sonata

export RP1M_300_ROOT=/WAVE/users/unix/jlanders/rp1m_300/rp1m_repertoire.zarr
export RUN_NAME=sonata_gpu_$(date +%Y%m%d_%H%M%S)
export OUTPUT_ROOT=/WAVE/datasets/ccoelho_lab-jlanders/$RUN_NAME
mkdir -p $OUTPUT_ROOT
```

Once the run is active, detach safely:

```text
Ctrl+b, then d
```

---

# Example Sonata Run Pattern

Example pattern for launching a Sonata preprocessing or training script:

```bash
cd /WAVE/projects/ECEN-524-Wi26/robopiano
conda activate sonata

export RP1M_300_ROOT=/WAVE/users/unix/jlanders/rp1m_300/rp1m_repertoire.zarr
export RUN_NAME=example_run_$(date +%Y%m%d_%H%M%S)
export OUTPUT_ROOT=/WAVE/datasets/ccoelho_lab-jlanders/$RUN_NAME
mkdir -p $OUTPUT_ROOT

python Sonata/scripts/prepare_rp1m.py \
  --profile medium
```

If the script or config supports an explicit output path, point it to `$OUTPUT_ROOT` or a subfolder under it.

Example:

```bash
python Sonata/scripts/train_primitives.py \
  --profile medium
```

If using YAML configs, update the config so output roots point to a folder under:

```bash
/WAVE/datasets/ccoelho_lab-jlanders
```

not inside the project directory.

---

# Storage Guidelines

Use this location for large outputs:

```bash
/WAVE/datasets/ccoelho_lab-jlanders
```

Store the following there:

- Training outputs
- Checkpoints
- Logs
- Tokenized datasets
- Cached features
- Evaluation results
- Rendered videos
- WandB artifacts if manually exported

Avoid writing large outputs to:

```bash
/WAVE/users/unix/jlanders
/WAVE/projects/ECEN-524-Wi26/robopiano
```

The project directory should mainly contain code, configs, and lightweight metadata.

---

# Checking Active Runs

Check active Slurm jobs:

```bash
squeue -u $USER
```

Check CPU and memory usage:

```bash
top
```

or:

```bash
htop
```

Check GPU usage on a GPU node:

```bash
nvidia-smi
```

Check output folder size:

```bash
du -sh /WAVE/datasets/ccoelho_lab-jlanders/*
```

Check the current run output size:

```bash
du -sh $OUTPUT_ROOT
```

---

# Reattaching to a Run

If you detached from `tmux`, reconnect to the HPC over SSH and run:

```bash
tmux ls
```

Then attach to the session:

```bash
tmux attach -t sonata_run
```

If you named the session differently, replace `sonata_run` with the correct session name.

---

# Safe Minimal Command Template

Use this as a basic template for launching a protected GPU run:

```bash
tmux new -s sonata_run

srun -p gpu \
  --gres=gpu:1 \
  --cpus-per-task=16 \
  --mem=128G \
  --time=1-00:00:00 \
  --pty bash

cd /WAVE/projects/ECEN-524-Wi26/robopiano
conda activate sonata

export RP1M_300_ROOT=/WAVE/users/unix/jlanders/rp1m_300/rp1m_repertoire.zarr
export OUTPUT_ROOT=/WAVE/datasets/ccoelho_lab-jlanders/my_new_run
mkdir -p $OUTPUT_ROOT

# Start the desired Sonata command here.
# After confirming the run has started, detach with:
# Ctrl+b, then d
```

---

# Summary

The safest pattern for long Sonata runs is:

```bash
tmux new -s sonata_run
srun -p gpu --gres=gpu:1 --cpus-per-task=16 --mem=128G --time=1-00:00:00 --pty bash
cd /WAVE/projects/ECEN-524-Wi26/robopiano
conda activate sonata
export RP1M_300_ROOT=/WAVE/users/unix/jlanders/rp1m_300/rp1m_repertoire.zarr
export OUTPUT_ROOT=/WAVE/datasets/ccoelho_lab-jlanders/my_new_run
mkdir -p $OUTPUT_ROOT
# run Sonata command
# detach with Ctrl+b, then d
```

Detach from `tmux`, not from the running process itself. Once detached, the Slurm job and shell session should continue running even if the SSH connection drops.
