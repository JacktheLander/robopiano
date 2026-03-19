# Sonata Deployment Snapshot

This repository snapshot is focused on the `Sonata/` pipeline and related local training references.

It intentionally does **not** vendor the upstream RoboPianist source tree. On HPC, clone upstream RoboPianist alongside this repository before running rollout evaluation or any environment-dependent tooling.

## Contents

- `Sonata/`: the Sonata preprocessing, primitive discovery, transformer, diffusion, evaluation, and W&B-integrated training pipeline.
- `tin/`: local reference training code used while wiring W&B behavior.
- `AGENTS.md`: cluster-oriented constraints and workflow notes.

## HPC bootstrap

```bash
bash scripts/bootstrap_upstream_robopianist.sh /project/$USER
```

This creates:

- `/project/$USER/robopiano` for this repository
- `/project/$USER/robopianist-upstream` for the upstream dependency

The detailed Sonata workflow remains in `Sonata/README.md`.
