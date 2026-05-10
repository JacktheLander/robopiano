from __future__ import annotations

import argparse
import sys
from pathlib import Path

VARIATIONS_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = VARIATIONS_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from variations.data.rp1m_loader import array_keys, group_keys, inspect_song, list_songs, open_rp1m_root
from variations.utils.io import save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect RP1M Zarr structure for Variations.")
    parser.add_argument("--rp1m-root", required=True)
    parser.add_argument("--max-songs", type=int, default=10)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = open_rp1m_root(args.rp1m_root)
    songs = list_songs(root)
    print(f"RP1M root: {args.rp1m_root}")
    print(f"Found {len(songs)} song groups")
    inspected = []
    for song in songs[: args.max_songs]:
        info = inspect_song(root, song)
        inspected.append(info)
        print(f"\n{song}")
        for name, meta in info["arrays"].items():
            print(f"  {name}: shape={meta['shape']} dtype={meta['dtype']}")
    out = {
        "rp1m_root": args.rp1m_root,
        "num_song_groups": len(songs),
        "first_song_groups": songs[: args.max_songs],
        "root_group_keys": group_keys(root)[: args.max_songs],
        "root_array_keys": array_keys(root),
        "inspected_songs": inspected,
    }
    output = Path(args.output) if args.output else VARIATIONS_ROOT / "outputs" / "inspection" / "rp1m_inspection.json"
    save_json(output, out)
    print(f"\nSaved inspection JSON: {output}")


if __name__ == "__main__":
    main()

