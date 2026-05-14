# MAESTRO online-style simulation

This folder drives RoboPianist **state-injection** playback (same style as Partita recorded-state rollout): each control step sets **hand joint positions** from a Variations checkpoint and **piano key poses** from MAESTRO-derived `target_keys[88]`, then renders offscreen frames to video.

## Requirements

- Repo layout on `PYTHONPATH`: the script prepends `Variations/src`, `Variations/`, `partita/src`, and the repo root.
- Conda env with **PyTorch**, **robopianist**, **MuJoCo**, **imageio**, and (for audio mux) **ffmpeg** + a soundfont, matching `HowToRun.md`.
- MAESTRO MIDI files (`.mid` / `.midi`) and `pretty_midi` (pulled in with robopianist).

Headless nodes often need e.g. `export MUJOCO_GL=egl` (the Partita rollout path sets a default).

## Command

From the robopiano repo root:

```bash
python Variations/simulate/simulate_maestro.py \
  --model-type diffusion \
  --checkpoint /path/to/checkpoints/best.pt \
  --midi-path /WAVE/datasets/ccoelho_lab-jlanders/MAESTRO/.../piece.mid \
  --output-root /WAVE/datasets/ccoelho_lab-jlanders/Variations/simulation
```

Pick an unseen MAESTRO piece by index (sorted relative paths under `--maestro-root`):

```bash
python Variations/simulate/simulate_maestro.py \
  --model-type mlp_baseline \
  --checkpoint /path/to/mlp_best.pt \
  --maestro-root /WAVE/datasets/ccoelho_lab-jlanders/MAESTRO \
  --piece-index 42
```

## Outputs (per run directory)

Under `--output-root` (default: `/WAVE/datasets/ccoelho_lab-jlanders/Variations/simulation/`):

| File | Description |
|------|--------------|
| `simulation.mp4` or `simulation.gif` | Copy of rendered media |
| `simulation.json` | Run metadata + rollout stats |
| `inputs_and_predictions.npz` | `target_keys`, `hand_joints` |
| `target_goals.proto` | Copy of goals proto used to load the task |
| `simulation_playback.json` | Detailed rollout payload from Partita-style recording |

## Note

Checkpoints implement **pointwise** `target_keys → joint_state[46]`; this script visualizes independent predictions along a **quantized MIDI timeline**, not a learned temporal policy.
