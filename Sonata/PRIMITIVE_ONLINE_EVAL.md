# Primitive Online Evaluation

This utility evaluates Stage 1 primitives as executable behaviors in RoboPianist, not just as compact reconstructions in feature space.

It works by:

1. Loading Stage 1 primitive assignments and primitive priors.
2. Rebuilding real primitive instances from the source dataset episodes.
3. Recovering intended key targets from `goals` and realized piano behavior from `piano_states`.
4. Restoring the segment start state as faithfully as the current RoboPianist API allows.
5. Replaying the primitive prior in simulation.
6. Scoring the resulting piano behavior with key-event metrics and optional trajectory errors.

## Run

From the `Sonata/` directory:

```bash
python scripts/evaluate_primitives_online.py \
  --profile debug \
  --primitive-root outputs/primitives/debug \
  --output-root outputs/evaluation/primitives_online/debug \
  --instances-per-primitive 2 \
  --save-debug
```

For an external RoboPianist clone, point the evaluator at the clone root or set `ROBOPIANIST_ROOT`:

```bash
python scripts/evaluate_primitives_online.py \
  --profile medium \
  --primitive-root /project/$USER/sonata/outputs/primitives/medium \
  --output-root /project/$USER/sonata/outputs/evaluation/primitives_online/medium \
  --robopianist-root /project/$USER/robopianist \
  --instances-per-primitive 8 \
  --save-debug
```

## Main Outputs

The evaluator writes under `outputs/evaluation/primitives_online/<profile>/` by default:

- `primitive_instances_enriched.csv`
- `primitive_instance_metrics.csv`
- `primitive_summary_metrics.csv`
- `aggregate_metrics.json`
- `failure_counts.json`
- `runtime_status.json`
- `plots/`
- `debug/` when `--save-debug` is enabled

## Current Limitations

- The current implementation restores hand joint positions and piano key states directly when the RoboPianist task exposes those internals. It does not reconstruct every hidden MuJoCo state perfectly.
- When Stage 1 priors were fit on non-action targets such as `hand_joints`, those primitives are marked as unsupported for online replay instead of silently substituting ground-truth actions.
- If the upstream RoboPianist task API changes its hand joint ordering or internal state fields, the direct restore path may need a small adapter update.
