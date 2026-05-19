from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from etude.data.trajectory_io import finite_difference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Etude tracking episodes from RP1M-like arrays.")
    parser.add_argument("--rp1m-root", required=True, help="RP1M zarr root or directory of .npz files")
    parser.add_argument("--profile", default="debug")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-songs", type=int, default=None)
    parser.add_argument("--max-episodes-per-song", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.rp1m_root)
    output_root = Path(args.output_root)
    if output_root.exists() and any(output_root.iterdir()) and not args.force:
        raise FileExistsError(f"{output_root} exists; pass --force to overwrite/add files")
    episodes_dir = output_root / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    candidates = sorted(source.rglob("*.npz")) if source.is_dir() else []
    if not candidates:
        raise RuntimeError(
            "No .npz episodes found. The RP1M zarr adapter is intentionally not guessed; "
            "add a source-specific reader once the exact array schema is known."
        )

    rows = []
    max_episodes = args.max_songs if args.max_songs is not None else len(candidates)
    if args.max_episodes_per_song is not None:
        max_episodes *= args.max_episodes_per_song

    for episode_id, path in enumerate(candidates[:max_episodes]):
        with np.load(path, allow_pickle=False) as npz:
            q = np.asarray(npz["q"] if "q" in npz else npz["q_ref"], dtype=np.float32)
            qdot = np.asarray(npz["qdot"] if "qdot" in npz else finite_difference(q), dtype=np.float32)
            actions = np.asarray(npz["actions"], dtype=np.float32)
            target_keys = np.asarray(npz["target_keys"], dtype=np.float32) if "target_keys" in npz else np.zeros((q.shape[0], 88), dtype=np.float32)
            fingertips = np.asarray(npz["fingertips"], dtype=np.float32) if "fingertips" in npz else np.zeros((q.shape[0], 30), dtype=np.float32)
        q_ref = q.astype(np.float32)
        qdot_ref = finite_difference(q_ref)
        out_name = f"episode_{episode_id:06d}.npz"
        np.savez_compressed(
            episodes_dir / out_name,
            q=q,
            qdot=qdot,
            q_ref=q_ref,
            qdot_ref=qdot_ref,
            actions=actions,
            target_keys=target_keys,
            fingertips=fingertips,
            dt=np.asarray(0.005, dtype=np.float32),
        )
        rows.append({"episode_id": episode_id, "path": f"episodes/{out_name}", "source": str(path), "timesteps": q.shape[0]})

    pd.DataFrame(rows).to_csv(output_root / "manifest.csv", index=False)
    (output_root / "normalization.json").write_text(json.dumps({}, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} Etude episodes to {output_root}")


if __name__ == "__main__":
    main()
