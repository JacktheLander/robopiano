from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sonata.primitives.visualization import plot_gmr_reconstruction, plot_primitive_frequency, plot_usage_entropy
from sonata.utils.io import read_table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rebuild primitive plots for Sonata-3.")
    parser.add_argument("--primitive-root", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    primitive_root = Path(args.primitive_root).resolve()
    assignments_df = read_table(primitive_root / "clustering" / "segment_assignments")
    library_df = read_table(primitive_root / "library" / "primitive_library")
    plot_dir = primitive_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_primitive_frequency(assignments_df, plot_dir / "primitive_frequency.png")
    plot_gmr_reconstruction(library_df, plot_dir / "primitive_gmr_reconstruction.png")
    plot_usage_entropy(assignments_df, plot_dir / "primitive_usage_entropy.png")


if __name__ == "__main__":
    main()
