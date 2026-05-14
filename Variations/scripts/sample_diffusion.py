from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

VARIATIONS_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = VARIATIONS_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

def _load_targets(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=False)
    if "target_keys" in data:
        return np.asarray(data["target_keys"], dtype=np.float32)
    first = data.files[0]
    return np.asarray(data[first], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample 46D joint states from a trained Variations diffusion checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--target-keys-file", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--ddim-steps", type=int, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    from variations.diffusion.trainer import sample_hand_states

    target_keys = _load_targets(Path(args.target_keys_file))
    samples = sample_hand_states(
        args.checkpoint,
        target_keys,
        num_samples=args.num_samples,
        ddim_steps=args.ddim_steps,
        device=args.device,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, target_keys=target_keys, joint_state_samples=samples)
    print(f"Saved samples: {out}")


if __name__ == "__main__":
    main()
