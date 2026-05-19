from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from etude.controllers.residual_mlp import ResidualMLP
from etude.data.rp1m_tracking_dataset import RP1MTrackingDataset
from etude.training.bc_trainer import train_bc_epoch
from etude.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an Etude MLP action/residual controller.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--wandb", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed_everything(int(config.get("seed", 7)))
    dataset = RP1MTrackingDataset(
        config["data"]["dataset_root"],
        sequence_length=int(config["data"].get("sequence_length", 1)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["data"].get("batch_size", 256)),
        shuffle=True,
        num_workers=int(config["data"].get("num_workers", 0)),
    )
    sample = dataset[0]
    input_dim = int(sample["features"].shape[-1])
    action_dim = int(sample["actions"].shape[-1])
    model = ResidualMLP(
        input_dim=input_dim,
        action_dim=action_dim,
        hidden_dim=int(config["controller"].get("hidden_dim", 256)),
        num_layers=int(config["controller"].get("num_layers", 4)),
        dropout=float(config["controller"].get("dropout", 0.05)),
    )
    device = torch.device(config["training"].get("device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"].get("lr", 3e-4)),
        weight_decay=float(config["training"].get("weight_decay", 1e-5)),
    )
    output_root = Path(args.output_root)
    checkpoint_dir = output_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    for epoch in range(1, int(config["training"].get("epochs", 30)) + 1):
        result = train_bc_epoch(model, loader, optimizer, device)
        print(f"epoch={epoch} train_loss={result.train_loss:.6f}")
        if result.train_loss < best_loss:
            best_loss = result.train_loss
            torch.save({"model": model.state_dict(), "config": config, "input_dim": input_dim, "action_dim": action_dim}, checkpoint_dir / "best.pt")


if __name__ == "__main__":
    main()
