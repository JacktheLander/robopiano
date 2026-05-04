"""Aggregate diagnostics from evaluate_stage1_rollout / evaluate_stage2_rollout outputs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _first_episode(rollout: dict[str, Any]) -> dict[str, Any]:
    eps = rollout.get("episodes") or []
    return eps[0] if eps else {}


def _action_summary_flat(rollout_dir: Path) -> dict[str, Any]:
    p = rollout_dir / "action_magnitude_summary.json"
    if p.exists():
        return _read_json(p)
    return {}


def extract_mode_row(mode_label: str, run_root: Path) -> dict[str, Any]:
    run_root = run_root.resolve()
    s1 = run_root / "stage1_rollout_summary.json"
    s2 = run_root / "stage2_rollout_summary.json"
    summary = _read_json(s1 if s1.exists() else s2)
    rollout_dir = run_root / "rollout"
    dm = _read_json(rollout_dir / "dm_control_rollout.json")
    ep = _first_episode(dm)
    mag = _action_summary_flat(rollout_dir)

    video_rel = ep.get("video_path")
    video_path = str((rollout_dir / video_rel).resolve()) if video_rel else None

    return {
        "mode_label": mode_label,
        "suite_load_succeeded": bool(ep.get("suite_load_succeeded", not ep.get("error"))),
        "error": ep.get("error"),
        "action_mean_abs": mag.get("action_mean_abs"),
        "action_max_abs": mag.get("action_max_abs"),
        "nonzero_fraction": mag.get("nonzero_fraction"),
        "env_action_dim": mag.get("env_action_dim"),
        "action_dim": mag.get("action_dim"),
        "reward": ep.get("reward"),
        "precision": ep.get("precision"),
        "recall": ep.get("recall"),
        "f1": ep.get("f1"),
        "keys_pressed_total": ep.get("keys_pressed_total"),
        "fraction_steps_with_key_press": ep.get("fraction_steps_with_key_press"),
        "video_path": video_path,
        "episode_id": ep.get("episode_id") or summary.get("selected_episode_id"),
        "song_id": ep.get("song_id") or summary.get("selected_song_id"),
        "midi_file": ep.get("midi_file"),
        "environment_name": ep.get("environment_name"),
    }


def build_comparison(
    *,
    dataset_actions_root: Path,
    oracle_gmr_root: Path,
    stage2_root: Path,
    out_json: Path,
    out_csv: Path,
) -> dict[str, Any]:
    ds = extract_mode_row("oracle_dataset_actions", dataset_actions_root)
    og = extract_mode_row("oracle_gmr_primitives", oracle_gmr_root)
    s2 = extract_mode_row("stage2_gmr", stage2_root)

    episode_id = ds.get("episode_id") or og.get("episode_id") or s2.get("episode_id")
    song_id = ds.get("song_id") or og.get("song_id") or s2.get("song_id")
    midi_file = ds.get("midi_file") or og.get("midi_file") or s2.get("midi_file")
    environment_name = ds.get("environment_name") or og.get("environment_name") or s2.get("environment_name")

    payload = {
        "selected_episode_id": episode_id,
        "selected_song_id": song_id,
        "midi_file_used": midi_file,
        "environment_name": environment_name,
        "modes": {
            "oracle_dataset_actions": ds,
            "oracle_gmr_primitives": og,
            "stage2_gmr": s2,
        },
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rows = [ds, og, s2]
    if rows:
        fieldnames = list(rows[0].keys())
        with out_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge rollout debug outputs into comparison JSON/CSV.")
    parser.add_argument(
        "--dataset-actions-root",
        type=Path,
        default=Path("Sonata/outputs/evaluation/debug_dataset_actions"),
    )
    parser.add_argument(
        "--oracle-gmr-root",
        type=Path,
        default=Path("Sonata/outputs/evaluation/debug_oracle_gmr"),
    )
    parser.add_argument(
        "--stage2-root",
        type=Path,
        default=Path("Sonata/outputs/evaluation/debug_stage2_gmr"),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("Sonata/outputs/evaluation/debug_rollout_comparison.json"),
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("Sonata/outputs/evaluation/debug_rollout_comparison.csv"),
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    build_comparison(
        dataset_actions_root=repo / args.dataset_actions_root,
        oracle_gmr_root=repo / args.oracle_gmr_root,
        stage2_root=repo / args.stage2_root,
        out_json=repo / args.out_json,
        out_csv=repo / args.out_csv,
    )


if __name__ == "__main__":
    main()
