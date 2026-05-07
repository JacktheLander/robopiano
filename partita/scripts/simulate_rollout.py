from __future__ import annotations

import argparse
import sys
from pathlib import Path

PARTITA_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PARTITA_ROOT.parent
SRC_ROOT = PARTITA_ROOT / "src"
for import_root in [REPO_ROOT, SRC_ROOT]:
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import numpy as np

from partita.evaluation.rollout import (
    rollout_recorded_rp1m_episode_with_robopianist,
    rollout_reconstructed_actions_with_robopianist,
)
from partita.utils.config import experiment_name, load_config, output_root
from partita.utils.io import ensure_dir, load_json, save_json


def _load_target_npz(data_dir: Path) -> dict[str, np.ndarray]:
    target = np.load(data_dir / "target_trajectory.npz")
    if "goals" not in target.files:
        raise RuntimeError("target_trajectory.npz does not contain goals; cannot synthesize rollout MIDI.")
    return {name: target[name] for name in target.files}


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay Partita target/reconstructed trajectories in RoboPianist and render video.")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--which",
        choices=["original", "reconstructed", "reconstructed-state", "both", "all"],
        default="both",
        help=(
            "original/reconstructed replay normalized actions; reconstructed-state restores "
            "reconstructed hand joints and piano states frame-by-frame; all runs every mode."
        ),
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--render-every", type=int, default=1)
    args = parser.parse_args()

    config = load_config(args.config)
    exp = experiment_name(config)
    root = output_root(config)
    data_dir = root / "data" / exp
    recon_dir = root / "reconstruction" / exp
    rollout_dir = ensure_dir(root / "rollout" / exp)
    selection = load_json(data_dir / "selection.json")
    target = _load_target_npz(data_dir)
    goals = np.asarray(target["goals"], dtype=np.float32)
    threshold = float(selection.get("key_threshold", config.get("selection", {}).get("key_threshold", 0.5)))
    control_timestep = float(config.get("control_timestep", 0.05))

    action_jobs = []
    if args.which in {"original", "both", "all"}:
        action_jobs.append(("original_target", np.load(recon_dir / "original_actions.npy")))
    if args.which in {"reconstructed", "both", "all"}:
        action_jobs.append(("reconstructed", np.load(recon_dir / "reconstructed_actions.npy")))

    state_jobs = []
    if args.which in {"reconstructed-state", "all"}:
        hand_joints_path = recon_dir / "reconstructed_hand_joints.npy"
        piano_states_path = recon_dir / "reconstructed_piano_states.npy"
        if not hand_joints_path.exists():
            raise RuntimeError(f"Missing {hand_joints_path}; rerun reconstruct_trajectory.py after training primitives.")
        if not piano_states_path.exists():
            raise RuntimeError(f"Missing {piano_states_path}; rerun reconstruct_trajectory.py after training primitives.")
        state_jobs.append((
            "reconstructed_state",
            np.load(hand_joints_path),
            np.load(piano_states_path),
        ))

    results = []
    for label, actions in action_jobs:
        print(f"Rendering {label} rollout ({actions.shape[0]} steps)...", flush=True)
        result = rollout_reconstructed_actions_with_robopianist(
            actions=actions,
            goals=goals,
            song_name=selection["song_name"],
            output_dir=rollout_dir,
            label=label,
            control_timestep=control_timestep,
            fps=args.fps,
            width=args.width,
            height=args.height,
            max_steps=args.max_steps,
            render_every=args.render_every,
            seed=0,
        )
        results.append(result)
        print(f"  video: {result.get('video_path')}")
        print(f"  reward: {result.get('total_reward')} executed={result.get('actions_executed')} terminated={result.get('terminated')}")
        if "rollout_key_f1" in result:
            print(
                "  rollout_key_f1: "
                f"{result.get('rollout_key_f1')} "
                f"precision={result.get('rollout_key_precision')} "
                f"recall={result.get('rollout_key_recall')}"
            )
        if result.get("audio_warning"):
            print(f"  audio_warning: {result.get('audio_warning')}")
    for label, hand_joints, piano_states in state_jobs:
        print(f"Rendering {label} playback ({min(hand_joints.shape[0], piano_states.shape[0])} steps)...", flush=True)
        result = rollout_recorded_rp1m_episode_with_robopianist(
            hand_joints=hand_joints,
            piano_states=piano_states,
            goals=goals,
            song_name=selection["song_name"],
            output_dir=rollout_dir,
            label=label,
            control_timestep=control_timestep,
            fps=args.fps,
            width=args.width,
            height=args.height,
            max_steps=args.max_steps,
            render_every=args.render_every,
            seed=0,
            threshold=threshold,
        )
        results.append(result)
        print(f"  video: {result.get('video_path')}")
        goals_metrics = result.get("against_goals", {})
        states_metrics = result.get("against_rp1m_piano_states", {})
        print(
            "  goals_key_f1: "
            f"{goals_metrics.get('key_f1')} "
            f"states_key_f1={states_metrics.get('key_f1')}"
        )
        if result.get("audio_warning"):
            print(f"  audio_warning: {result.get('audio_warning')}")
    existing_results = []
    summary_path = rollout_dir / "rollout_summary.json"
    if summary_path.exists():
        try:
            existing = load_json(summary_path)
            existing_results = list(existing.get("results", []))
        except Exception:
            existing_results = []
    by_label = {str(item.get("label")): item for item in existing_results if isinstance(item, dict)}
    for item in results:
        by_label[str(item.get("label"))] = item
    combined = list(by_label.values())
    save_json(summary_path, {"experiment_name": exp, "results": combined})
    print(f"Saved rollout summary: {summary_path}")


if __name__ == "__main__":
    main()
