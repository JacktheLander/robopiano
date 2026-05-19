# Etude Integration Debug Report

## 1. What Was Broken

- The public controller package only exported the original PD and hybrid controllers, while the newer controller families were spread across standalone modules with no Etude-local factory or normalized family resolution.
- Cross-feature plan data had drifted: `PlanBundle` did not carry `key_state` or `planner_confidence`, and it did not expose compatibility aliases for `desired_keys` / `goal_keys`.
- Phase naming had diverged across modules and configs. Newer code used canonical safety phases such as `pre_contact` and `contact`, while existing tests and configs still used legacy names such as `attack`, `press`, `sustain`, and `idle`.
- Frame-level piano metric naming was inconsistent. Event metrics used the expected `piano/event_*` namespace, but frame metrics were still exposed only as `piano/note_*`.
- Experiment config validation still required only `controller.type`, even after newer configs started to benefit from a normalized `controller.family`.

## 2. What Was Fixed

- Added [`src/etude/controllers/factory.py`](/D:/Codex/Robopiano/robopiano/Etude/src/etude/controllers/factory.py) as the Etude-local controller factory surface.
- Expanded [`src/etude/controllers/__init__.py`](/D:/Codex/Robopiano/robopiano/Etude/src/etude/controllers/__init__.py) to export the newer controller families and factory helpers.
- Expanded [`src/etude/core/plan_bundle.py`](/D:/Codex/Robopiano/robopiano/Etude/src/etude/core/plan_bundle.py) to carry `key_state`, `planner_confidence`, and compatibility aliases while preserving the existing reset/act controller contract.
- Normalized phase handling across [`src/etude/features/fingertip_phase_blocks.py`](/D:/Codex/Robopiano/robopiano/Etude/src/etude/features/fingertip_phase_blocks.py), [`src/etude/controllers/residual_safety.py`](/D:/Codex/Robopiano/robopiano/Etude/src/etude/controllers/residual_safety.py), [`src/etude/controllers/pd_gain_utils.py`](/D:/Codex/Robopiano/robopiano/Etude/src/etude/controllers/pd_gain_utils.py), and [`src/etude/controllers/pd_scheduled.py`](/D:/Codex/Robopiano/robopiano/Etude/src/etude/controllers/pd_scheduled.py).
- Added frame-metric aliases in [`src/etude/evaluation/metrics.py`](/D:/Codex/Robopiano/robopiano/Etude/src/etude/evaluation/metrics.py) so both `piano/frame_*` and legacy `piano/note_*` are available.
- Updated [`src/etude/experiments.py`](/D:/Codex/Robopiano/robopiano/Etude/src/etude/experiments.py) so config validation accepts `controller.family` or `controller.type`, and summary extraction recognizes `piano/frame_f1`.
- Added controller-factory and compatibility coverage in:
  - [`tests/test_controller_factory.py`](/D:/Codex/Robopiano/robopiano/Etude/tests/test_controller_factory.py)
  - [`tests/test_fingertip_phase_features.py`](/D:/Codex/Robopiano/robopiano/Etude/tests/test_fingertip_phase_features.py)
  - [`tests/test_event_metrics.py`](/D:/Codex/Robopiano/robopiano/Etude/tests/test_event_metrics.py)
  - [`tests/test_pd_scheduled.py`](/D:/Codex/Robopiano/robopiano/Etude/tests/test_pd_scheduled.py)
  - [`tests/test_universal_scaffold.py`](/D:/Codex/Robopiano/robopiano/Etude/tests/test_universal_scaffold.py)
- Updated controller config examples to include normalized `controller.family` keys and canonical PD phase names:
  - [`configs/controller/pd_phase_scheduled.yaml`](/D:/Codex/Robopiano/robopiano/Etude/configs/controller/pd_phase_scheduled.yaml)
  - [`configs/controller/pd_grouped.yaml`](/D:/Codex/Robopiano/robopiano/Etude/configs/controller/pd_grouped.yaml)
  - [`configs/controller/key_aware_mlp.yaml`](/D:/Codex/Robopiano/robopiano/Etude/configs/controller/key_aware_mlp.yaml)
  - [`configs/controller/fingertip_phase_residual.yaml`](/D:/Codex/Robopiano/robopiano/Etude/configs/controller/fingertip_phase_residual.yaml)
  - [`configs/controller/temporal_residual_gru.yaml`](/D:/Codex/Robopiano/robopiano/Etude/configs/controller/temporal_residual_gru.yaml)
  - [`configs/controller/temporal_residual_tcn.yaml`](/D:/Codex/Robopiano/robopiano/Etude/configs/controller/temporal_residual_tcn.yaml)
  - [`configs/controller/inverse_dynamics.yaml`](/D:/Codex/Robopiano/robopiano/Etude/configs/controller/inverse_dynamics.yaml)
  - [`configs/controller/hierarchical_contact.yaml`](/D:/Codex/Robopiano/robopiano/Etude/configs/controller/hierarchical_contact.yaml)
  - [`configs/controller/residual_safe_mlp.yaml`](/D:/Codex/Robopiano/robopiano/Etude/configs/controller/residual_safe_mlp.yaml)
  - [`configs/controller/residual_safe_gru.yaml`](/D:/Codex/Robopiano/robopiano/Etude/configs/controller/residual_safe_gru.yaml)

## 3. What Remains Untested

- Dataset-backed training via `RP1MTrackingDataset`
- RoboPianist rollout execution
- MuJoCo / DM Control integration
- GPU-backed model loading and checkpoints
- Local batch scripts and Slurm submission paths

## 4. Commands Run

- `Get-Location | Select-Object -ExpandProperty Path`
- `git status --short .`
- `Get-ChildItem src,configs,scripts,tests,README.md -Force | Format-List FullName,Mode,Length`
- `rg --files src/etude configs tests`
- `Get-ChildItem -Recurse -Filter '__init__.py' src\etude | Select-Object -ExpandProperty FullName`
- `python -m compileall -q src scripts tests`
- Inline Python import checks for `etude`, `etude.controllers`, `etude.evaluation`, `etude.training`, `etude.utils`, `etude.features`, and `etude.core`
- Inline YAML parse check for every `configs/**/*.yaml`
- `python -m pytest tests -q -m "not heavy and not rollout and not gpu and not dataset"`

## 5. Test Results

- `compileall`: pass
- Import checks: pass
- Config parse checks: pass
- `pytest` lightweight: pass (`70 passed`)

## 6. Known Risks

- The new controller factory is in place, but any future runtime entrypoint that instantiates controllers directly still needs to route through it to get the full normalization benefit.
- Phase inference from `target_keys` is intentionally heuristic and should be treated as a compatibility fallback, not a substitute for authoritative planner-provided phase labels.
- I did not modify existing user changes in `src/etude/training/replay_buffer.py`.

## 7. Recommended Next Agent / Run

- Run a local-safe experiment-config validation sweep using the new experiment configs and factory normalization.
- After that, run an explicit heavy-environment agent or local session for dataset-backed training and rollout validation against RoboPianist.
