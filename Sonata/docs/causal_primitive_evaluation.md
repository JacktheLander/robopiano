# Causal Primitive Evaluation

Primitive online evaluation is causal only when success is measured from robot-caused physical key activation during rollout. Restored dataset piano state, goals, target MIDI, highlighted keys, and reference audio are target/context signals, not observations.

## Restore Modes

`hands_only` is the default. It resets the simulator, restores available robot hand joint positions and velocities, and does not restore piano qpos, piano activation, goals, MIDI timestep, or target note state.

`neutral` resets the simulator normally and restores neither hands nor piano.

`unsafe_legacy` preserves the old behavior for comparison only. It is explicit opt-in, emits warnings, and writes `causal_validated=false` with `causal_failure_reason="unsafe_legacy_eval"`.

## Neutral Start

Causal evaluation validates frame 0 before any primitive action is applied:

```yaml
causal_eval:
  require_neutral_piano_start: true
  fail_if_initial_keys_active: true
  initial_key_activation_threshold: 0.1
```

If any physical key activation is above threshold at frame 0, the rollout fails. This prevents an already-pressed key from being counted as primitive success.

## Zero-Action Ablation

For each primitive instance, the evaluator can run a paired rollout with the same reset/restore state and duration but all controls set to zero:

```yaml
causal_eval:
  run_zero_action_ablation: true
```

If zero actions press the same target key, the primitive rollout is marked `status="contaminated_zero_action"`, `success=false`, and `causal_validated=false`.

## Contact-Gated Success

A target keypress is counted only when:

1. The physical key activation crosses threshold from inactive to active during rollout.
2. The key was inactive at rollout start.
3. A fingertip contacts, or is within the configured distance of, that same key.
4. Contact occurs before or at activation within `contact_tolerance_frames`.

The evaluator prefers MuJoCo contact pairs. If contact pairs are unavailable, it uses a fingertip/key distance proxy when key positions are exposed. If neither is available and contact is required, the rollout fails with `status="contact_unavailable"`.

## Target vs Observation Columns

Outputs keep target and observation sources separate:

`target_key_indices`, `reference_midi_key_indices`: target/context.

`physical_pressed_key_indices`, `activation_key_indices`: simulator piano activation.

`contact_key_indices`: same-key fingertip contact candidates.

`robot_midi_key_indices`: robot-generated MIDI, when available.

Legacy state metrics are reported under `legacy_state_metrics`. Causal contact metrics are reported under `causal_contact_metrics` and exclude rows with `causal_validated=false`.

## Audio and Video

Causal primitive videos default to:

```yaml
rollout:
  video_audio_source: none
```

Reference MIDI is not muxed into primitive causality videos. If robot MIDI audio is enabled, it must be labeled as robot-generated and must not be used for success scoring.

Rendered debug frames include primitive id, segment id, restore mode, target keys, and causal warnings when available.

## Prepress Causal Segmentation

`prepress_causal` learns the hand motion leading into key activation:

```yaml
segmentation_strategy: prepress_causal
prepress_steps: 12
post_onset_steps: 3
min_inactive_pre_steps: 4
min_hold_steps: 2
activation_threshold: 0.5
```

For each activation onset `t_on`, the segment is:

```text
start = t_on - prepress_steps
end   = t_on + post_onset_steps
```

Segments are rejected if the target key is active at segment start, active in the prepress window, lacks sufficient history, lacks action data, is outside length bounds, or does not hold after onset. Dataset piano/goals may identify event times, but GMR targets are always action trajectories.

## Commands

Debug Stage 1:

```bash
python Sonata/scripts/train_primitives.py --profile prepress_debug
```

Validate Stage 1 contract:

```bash
python Sonata/scripts/validate_primitives_contract.py \
  --primitive-root Sonata/outputs/primitives/prepress_debug
```

Causal primitive evaluation:

```bash
python Sonata/scripts/evaluate_primitives_online.py \
  --primitive-root Sonata/outputs/primitives/prepress_debug \
  --output-root Sonata/outputs/evaluation/primitives_online/prepress_debug_causal \
  --render-video \
  --max-render-instances 12 \
  --save-debug \
  --force \
  --video-audio-source none
```

Leakage audit:

```bash
python Sonata/scripts/audit_no_piano_state_leakage.py
```
