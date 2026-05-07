from __future__ import annotations

import sys
from pathlib import Path

PARTITA_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PARTITA_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse

from partita.data.rp1m_loader import array_keys, group_keys, inspect_song, list_songs, open_rp1m_root
from partita.data.song_selector import find_preferred_songs
from partita.utils.io import save_json

DEFAULT_TERMS = [
    "FrenchSuiteNo5Sarabande",
    "FrenchSuiteNo5",
    "Sarabande",
    "PianoSonataNo232NdMov",
    "FrenchSuiteNo1Allemande",
    "PianoSonataD8451StMov",
    "PartitaNo26",
    "GolliwoggsCakewalk",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect RP1M Zarr structure for Partita.")
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

    matches = find_preferred_songs(songs, DEFAULT_TERMS)
    print("\nPreferred matches:")
    for match in matches[:20]:
        print(f"  priority={match['priority']} term={match['matched_term']} song={match['song_name']}")

    out = {
        "rp1m_root": args.rp1m_root,
        "num_song_groups": len(songs),
        "first_song_groups": songs[: args.max_songs],
        "root_group_keys": group_keys(root)[: args.max_songs],
        "root_array_keys": array_keys(root),
        "inspected_songs": inspected,
        "preferred_matches": matches,
    }
    output = Path(args.output) if args.output else PARTITA_ROOT / "outputs" / "inspection" / "rp1m_inspection.json"
    save_json(output, out)
    print(f"\nSaved inspection JSON: {output}")


if __name__ == "__main__":
    main()
