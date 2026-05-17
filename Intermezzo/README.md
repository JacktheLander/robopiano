# Intermezzo: Variations Waypoint Trajectory Planner

Intermezzo converts MIDI-derived `target_keys[T, 88]` into a full hand-state guidance trajectory `planned_hand_joints[T, 46]`.

It uses a trained Variations diffusion checkpoint for sparse press waypoints:

```text
target_keys[88] -> hand_joints[46]
```

Then it plans the states between waypoints. Intermezzo does not emit actions, step RoboPianist, or implement a controller. The output is intended to guide a downstream action controller.

## Planning Policy

- Waypoints are non-empty keyset onset/change frames from the input `target_keys` roll.
- Variations predicts one 46-D hand pose for each waypoint.
- The planner preserves waypoint endpoint poses exactly after clipping the hand vertical forearm DOFs into the RoboPianist reduced-hand range.
- Between consecutive waypoints, Intermezzo first interpolates smooth joint motion on an internal dense buffer, then applies waypoint-centered press and optional magnetic key-alignment windows, then downsamples back to the original control-rate shape.
- Hand side is inferred from key index: keys `<44` use left hand, keys `>=44` use right hand, and cross-hand chords lift both.

The reduced 46-D hand layout used by Intermezzo is:

```text
right forearm_ty index = 22
left  forearm_ty index = 45
forearm_ty range       = [0.0, 0.06]
```

## Interpolation and Force Pipeline

The public output contract stays `planned_hand_joints[T, 46]` and `planned_hand_velocities[T, 46]`, but planning runs internally at `interpolation_substeps` times the input rate. The default is `10` substeps for `control_timestep=0.05`, giving a dense timestep of `0.005s` to match MuJoCo's 200 Hz physics step.

1. Allocate a dense hand-state buffer of length `T * interpolation_substeps`, placing each Variations waypoint at `waypoint_frame * interpolation_substeps`.
2. Fill each dense segment with force-free smoothstep interpolation.
3. Apply per-waypoint vertical press windows using `press_approach_s`, `press_hold_s`, `press_release_s`, `press_envelope_power`, and optional `press_depth`; optional magnetic XY correction uses the same waypoint-centered approach/hold window.
4. Keep the 200 Hz dense stream as `planned_hand_joints_dense` / `planned_hand_velocities_dense`, then downsample every `interpolation_substeps` frame for the legacy `planned_hand_joints[T, 46]` and `planned_hand_velocities[T, 46]` arrays.

`clearance_height`, `lift_fraction`, and `descent_fraction` remain available; `clearance_height` is used to clamp the released forearm height between press windows.

## WAVE Usage

Follow the repository run guide first:

```bash
cd /WAVE/projects/ECEN-524-Wi26/robopiano
conda activate sonata
```

Use the shared dataset/output location for runtime outputs:

```bash
export INTERMEZZO_OUTPUT_ROOT=/WAVE/datasets/ccoelho_lab-jlanders/Intermezzo/runs
export VARIATIONS_DIFFUSION_CHECKPOINT=/WAVE/datasets/ccoelho_lab-jlanders/Variations/diffusion/full/checkpoints/best.pt
```

Plan from a MIDI file:

```bash
python Intermezzo/scripts/plan_trajectory.py \
  --checkpoint "$VARIATIONS_DIFFUSION_CHECKPOINT" \
  --midi-path /WAVE/datasets/ccoelho_lab-jlanders/MAESTRO/path/to/piece.mid \
  --output-root "$INTERMEZZO_OUTPUT_ROOT" \
  --device auto
```

Plan from an existing NPZ containing `target_keys`:

```bash
python Intermezzo/scripts/plan_trajectory.py \
  --checkpoint "$VARIATIONS_DIFFUSION_CHECKPOINT" \
  --target-keys-npz /path/to/target_keys.npz \
  --output-root "$INTERMEZZO_OUTPUT_ROOT" \
  --device auto
```

Each invocation creates a new timestamped run directory and refuses to overwrite an existing run directory.

## Output Files

Each run directory contains:

| File | Contents |
| ---- | -------- |
| `trajectory.npz` | `target_keys`, `waypoint_frames`, `waypoint_target_keys`, `waypoint_hand_joints`, `planned_hand_joints`, `planned_hand_velocities`, `planned_hand_joints_dense`, `planned_hand_velocities_dense`, `segment_ids`, `segment_ids_dense` |
| `metadata.json` | Source paths, model settings, planner settings, shapes, and run directory |

## Smoke Checks

```bash
cd /WAVE/projects/ECEN-524-Wi26/robopiano
conda activate sonata

python Intermezzo/scripts/plan_trajectory.py --help
pytest Intermezzo/tests
```
