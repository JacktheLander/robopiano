# TIN MAESTRO Evaluation on SLURM

Use a long-lived `tmux` session so the job survives disconnects, then request one V100 GPU for up to 24 hours.

```bash
tmux new -s tin-maestro-eval
```

Inside the tmux session:

```bash
cd /home/jackthelander/robopianist
srun --gres=gpu:v100:1 --time=24:00:00 --pty bash
conda activate sonata
export MUJOCO_GL=egl
python tin/evaluate_maestro.py \
  --dataset-root /WAVE/datasets/ccoelho_lab-jlanders/MAESTRO \
  --output-root /WAVE/datasets/ccoelho_lab-jlanders/tin_eval \
  --max-steps-per-song 100000 \
  --device auto \
  --resume
```

Useful `tmux` commands:

```bash
tmux detach
tmux attach -t tin-maestro-eval
tmux kill-session -t tin-maestro-eval
```

Outputs are updated after each completed song in `/WAVE/datasets/ccoelho_lab-jlanders/tin_eval`:

- `piece_metrics.jsonl`
- `piece_metrics.csv`
- `summary.json`
- `f1_histogram.png`
