# Stage 3 Residual Control

## What Changed

Stage 3 is now trained as a residual refiner over the Stage 1 GMR prior instead of a full action regressor:

`final_action = gmr_prior + residual`

For `full`, the diffusion model denoises residual corrections and the pipeline composes them back onto the prior at inference time. `planner_no_prior` and `diffusion_only` still work by zeroing the prior. `gmr_only` remains a prior-only baseline.

The trainer now derives note-state targets directly from cached episode `goals` or `piano_states`:

- `note_target`: active piano keys over the Stage 3 horizon
- `hold_target`: keys that should continue to be held
- `sustain_target`: sustain pedal state when available

Those targets supervise Stage 3 through a lightweight fitted action-to-note surrogate rather than pure action reconstruction alone.

## New Loss

Stage 3 now optimizes:

`L_total = w_note * L_note + w_hold * L_hold + w_imitation * L_imitation + w_smooth * L_smooth + w_residual * L_residual + w_diffusion * L_diffusion`

Default weights are recall-first:

- `note_loss_weight: 1.0`
- `hold_loss_weight: 0.5`
- `imitation_weight: 0.25`
- `smoothness_weight: 0.02`
- `residual_reg_weight: 0.05`
- `diffusion_weight: 0.25`
- `false_negative_weight: 4.0`
- `false_positive_weight: 1.0`

`L_note` and `L_hold` use asymmetric binary losses so missing a required note costs more than adding an extra note. Imitation and smoothness remain secondary regularizers to keep outputs in-distribution without suppressing decisive key strikes.

## Why This Matches Piano Control Better

The old Stage 3 objective mainly rewarded denoising and action similarity, which made conservative outputs acceptable even when they missed notes. The new objective shifts the primary training signal to note correctness:

- first hit the target keys
- then keep them held or sustained correctly
- then clean up extra presses
- then improve smoothness

That is closer to online piano control, where a slightly aggressive but correct strike is preferable to a smooth miss.

## Training

Recommended first run:

```bash
cd /home/jackthelander/robopianist/Sonata
python scripts/train_diffusion.py \
  --profile medium \
  --variant full \
  --planner-checkpoint /home/jackthelander/robopianist/Sonata/outputs/transformer/Sonata-3-transformer-Sonata-3-medium-planner-tuned-v1-seed7-20260322-191906/checkpoints/best.pt
```

Validation artifacts now include predicted actions, priors, residuals, targets, predicted note activations, hold activations, and sustain activations in `artifacts/val_samples.npz`.

## Ablation Plan

1. Old Stage 3: `predict_residual: false`, `note_loss_weight: 0.0`, `hold_loss_weight: 0.0`, `diffusion_weight: 1.0`, `imitation_weight: 0.5`, `smoothness_weight: 0.05`.
2. Residual-only with imitation: `predict_residual: true`, `note_loss_weight: 0.0`, `hold_loss_weight: 0.0`, keep `imitation_weight` and `residual_reg_weight`.
3. Residual + note loss: `predict_residual: true`, `note_loss_weight: 1.0`, `hold_loss_weight: 0.0`.
4. Residual + note + hold loss: full new default config.

Primary comparison metrics:

- `val/note_recall`
- `val/note_precision`
- `val/note_f1`
- `val/sustain_recall`
- `val/sustain_precision`
- `val/sustain_f1`
- `val/imitation_l1`
- `val/smoothness`
