# Testing And Local Batch Plan

## Login-node smoke test

Run:

```bash
bash scripts/smoke_check.sh
```

This only performs compile checks, a package import check, optional lightweight pytest selection, and dry-runs for every experiment config.

## Dry-run experiment configs

Run:

```bash
python scripts/run_experiment.py --config configs/experiments/00_smoke_pd.yaml --dry-run
```

Dry-run parses config includes, validates import paths, resolves the output directory, and prints the planned command without loading datasets, training, or rollout.

## Submit one CPU eval

Run:

```text
cmd /c local_batch\run_etude_cpu_eval.echo D:\Codex\Robopiano\robopiano\Etude\configs\experiments\01_pd_gain_sweep.yaml
```

## Submit one GPU training run

Run:

```text
cmd /c local_batch\run_etude_gpu_train.echo D:\Codex\Robopiano\robopiano\Etude\configs\experiments\03_key_aware_clean.yaml
```

## Submit rollout eval

Run:

```text
cmd /c local_batch\run_etude_rollout_eval.echo D:\Codex\Robopiano\robopiano\Etude\configs\experiments\10_rollout_eval.yaml
```

## Submit robustness sweep

Run:

```bash
bash scripts/queue_etude_experiments.sh --submit
```

Or run the array helper directly:

```text
cmd /c local_batch\run_etude_robustness_array.echo
```

## Outputs and logs

- Experiment outputs default to `D:\Codex\Robopiano\robopiano\Etude\runs`.
- Each run writes a `resolved_config.yaml` and `experiment_manifest.json` into its output directory before execution.
- Batch helper logs should be redirected under `D:\Codex\Robopiano\robopiano\Etude\runs\logs`.

## What not to run on the local machine

- Full training loops interactively from the Windows workspace shell unless you intentionally want a local run.
- Rollout or MuJoCo evaluation from the login/smoke path.
- Large dataset scans or ad hoc experiments outside the queued wrappers.

## Summarize results

Run:

```bash
python scripts/summarize_results.py --root D:\Codex\Robopiano\robopiano\Etude\runs --out D:\Codex\Robopiano\robopiano\Etude\runs\summary.csv
```

The summarizer scans for JSON and CSV metric files, joins nearby manifest metadata when present, and writes a single CSV table.

## Add a new experiment config

1. Copy one of the files in `configs/experiments`.
2. Update `experiment.name`, `experiment.controller_family`, `experiment.evaluation_mode`, and `experiment.resource_tier`.
3. Point `includes` at the controller, feature, loss, corruption, and evaluation presets you want to compose.
4. Set `execution.kind` and `execution.local_batch_script`.
5. Replace any `CHANGE_ME` dataset, checkpoint, or trajectory placeholders.
6. Verify the config with `python scripts/run_experiment.py --config <path> --dry-run`.
