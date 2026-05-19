# Local Batch Helpers

These batch helpers stay inside `Etude` and write outputs under `D:\Codex\Robopiano\robopiano\Etude\runs`.

Usage:

```text
cmd /c local_batch\run_etude_cpu_eval.echo D:\Codex\Robopiano\robopiano\Etude\configs\experiments\01_pd_gain_sweep.yaml
cmd /c local_batch\run_etude_gpu_train.echo D:\Codex\Robopiano\robopiano\Etude\configs\experiments\03_key_aware_clean.yaml
cmd /c local_batch\run_etude_rollout_eval.echo D:\Codex\Robopiano\robopiano\Etude\configs\experiments\10_rollout_eval.yaml
cmd /c local_batch\run_etude_robustness_array.echo
```

Notes:

- Replace `CHANGE_ME` placeholders in experiment configs before submission.
- `scripts/queue_etude_experiments.sh --dry-run` prints the planned commands.
- `scripts/queue_etude_experiments.sh --submit` executes the `.echo` scripts through `cmd.exe`.
- Logs land under `D:\Codex\Robopiano\robopiano\Etude\runs\logs`.
