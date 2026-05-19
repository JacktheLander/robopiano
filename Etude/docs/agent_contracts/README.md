# Etude Agent Contracts

This document defines the shared scaffolding that feature agents may depend on without editing each other's files.

## PlanBundle contract

`etude.core.plan_bundle.PlanBundle` is the canonical lightweight container for closed-loop playback plans.

- Required fields: `q_ref`, `dt`
- Optional aligned arrays: `qdot_ref`, `target_keys`, `fingertip_ref`, `fingertip_weights`, `phase`
- Free-form fields: `assignments`, `metadata`
- Validation is simulator-free and limited to array shape checks plus positive `dt`

## Parallelization rule

Agents must not edit the same file. If a new shared contract is needed, add a new owned module and expose it by import path instead of modifying a central registry.

## Feature module exposure

Feature blocks should be referenced by module path strings, for example:

```yaml
features:
  blocks:
    - etude.features.key_blocks:build_key_features
    - etude.features.fingertip_phase_blocks:build_phase_features
```

Use `etude.utils.import_utils.load_symbol(...)` or `etude.features.registry.resolve_feature_block(...)` to load those hooks.

## Allowed validation

Allowed lightweight validation commands:

- `python -m compileall src/etude/core src/etude/data src/etude/features src/etude/utils`
- `python -m py_compile <owned python files>`
- `python -c "..."`
- small synthetic-array unit tests already scoped to owned files

## Explicit local bans

Do not run heavy local jobs:

- full `pytest`
- training scripts
- rollout scripts
- dataset extraction
- RP1M or MAESTRO scans
- RoboPianist simulator rollouts
- WandB runs
