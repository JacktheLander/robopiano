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
- Between consecutive waypoints, the active hand(s) use a `Lift Then Land` clearance profile:
  - keep the current press pose at the segment start,
  - lift the active hand forearm vertical DOF,
  - move through the segment while cleared,
  - descend into the next Variations press pose at the next waypoint.
- Hand side is inferred from key index: keys `<44` use left hand, keys `>=44` use right hand, and cross-hand chords lift both.

The reduced 46-D hand layout used by Intermezzo is:

```text
right forearm_ty index = 22
left  forearm_ty index = 45
forearm_ty range       = [0.0, 0.06]
```

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
| `trajectory.npz` | `target_keys`, `waypoint_frames`, `waypoint_target_keys`, `waypoint_hand_joints`, `planned_hand_joints`, `planned_hand_velocities`, `segment_ids` |
| `metadata.json` | Source paths, model settings, planner settings, shapes, and run directory |

## Smoke Checks

```bash
cd /WAVE/projects/ECEN-524-Wi26/robopiano
conda activate sonata

python Intermezzo/scripts/plan_trajectory.py --help
pytest Intermezzo/tests
```
