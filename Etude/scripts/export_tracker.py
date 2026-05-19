from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Export an Etude checkpoint to TorchScript.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if "scripted" not in ckpt:
        raise ValueError("Checkpoint export requires a materialized scripted model in future training runs")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(ckpt["scripted"], args.output)


if __name__ == "__main__":
    main()
