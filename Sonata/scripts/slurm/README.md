# Stage 1 Eventful HPC Launch

This directory contains the checked-in cluster workflow for the event-faithful Stage 1 primitive run plus automatic MAESTRO primitive online evaluation.

## Required Environment Variables

- `RP1M_300_ROOT`: shared RP1M zarr path
- `MAESTRO_MIDI_ROOT`: root directory containing MAESTRO `.mid` / `.midi` files

Optional overrides:

- `PROJECT_ROOT`
- `SONATA_ROOT`
- `CONDA_ENV_NAME`
- `PRIMITIVE_PROFILE`
- `PRIMITIVE_CONFIG`
- `PRIMITIVE_OUTPUT_ROOT`
- `POST_EVAL_CONFIG`
- `POST_EVAL_OUTPUT_ROOT`
- `ROBOPIANIST_ROOT`

## Recommended tmux Launch

From a login node:

```bash
export RP1M_300_ROOT=/WAVE/users2/unix/$USER/rp1m_300/rp1m_repertoire.zarr
export MAESTRO_MIDI_ROOT=/WAVE/datasets/ccoelho_lab-jlanders/maestro-v3.0.0/maestro-v3.0.0
export PERSIST_ROOT=/WAVE/datasets/ccoelho_lab-jlanders
tmux new-session -d -s sonata-stage1-eventful \
  "bash -lc 'sbatch Sonata/scripts/slurm/train_primitives_eventful.slurm; exec bash'"
tmux attach -t sonata-stage1-eventful
```

Or use the helper:

```bash
bash Sonata/scripts/slurm/launch_stage1_eventful_tmux.sh
```

## Monitoring

```bash
squeue -u $USER
tail -f /WAVE/datasets/ccoelho_lab-jlanders/sonata_logs/sonata-s1-eventful-*.out
```

The Slurm job activates the `sonata` conda environment, runs `train_primitives.py` with the `medium_eventful` config, and then launches the post-training MAESTRO primitive online evaluation automatically. By default, Stage 1 data outputs, primitive outputs, evaluation outputs, and Slurm logs are written under `/WAVE/datasets/ccoelho_lab-jlanders`.
