#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

VARIATIONS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = VARIATIONS_ROOT.parent
for path in (VARIATIONS_ROOT / "src", VARIATIONS_ROOT, REPO_ROOT):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from variations.diffusion.trainer import build_diffusion_from_config, build_model_from_config, load_model_for_inference, resolve_device  # noqa: E402
from variations.inference.predict_press_pose import load_latent_mdn_checkpoint, load_mlp_baseline_checkpoint  # noqa: E402
from variations.losses.latent_mdn_loss import latent_mdn_total_loss  # noqa: E402


def load_refined_labels(path: str | Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    data = np.load(path, allow_pickle=True)
    target_keys = torch.as_tensor(np.asarray(data["target_keys"], dtype=np.float32), device=device)
    refined = torch.as_tensor(np.asarray(data["refined_hand_state"], dtype=np.float32), device=device)
    meta = {key: data[key].item() if np.asarray(data[key]).shape == () else data[key] for key in data.files if key not in {"target_keys", "refined_hand_state"}}
    return target_keys, refined, meta


def split_train_val(x: torch.Tensor, y: torch.Tensor, val_fraction: float) -> tuple[TensorDataset, TensorDataset]:
    count = int(x.shape[0])
    val_count = max(1, int(round(count * float(val_fraction)))) if count > 1 else 0
    train_count = count - val_count
    return TensorDataset(x[:train_count], y[:train_count]), TensorDataset(x[train_count:], y[train_count:])


def normalize(refined: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (refined - mean.to(refined.device, refined.dtype)) / std.to(refined.device, refined.dtype)


def fine_tune_mlp(args: argparse.Namespace, device: torch.device) -> Path:
    keys, refined, _meta = load_refined_labels(args.labels, device)
    model, normalizer, config, payload = load_mlp_baseline_checkpoint(args.checkpoint, device=device)
    target_norm = normalize(refined, normalizer.mean, normalizer.std)
    train_ds, val_ds = split_train_val(keys, target_norm, args.val_fraction)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_state = None
    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for xb, yb in tqdm(loader, desc=f"mlp contact epoch {epoch}", leave=False):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        vals = []
        with torch.no_grad():
            for xb, yb in val_loader:
                vals.append(float(torch.nn.functional.mse_loss(model(xb), yb).detach().cpu()))
        val = float(np.mean(vals)) if vals else float(np.mean(losses))
        print({"epoch": epoch, "train_contact_mse": float(np.mean(losses)), "val_contact_mse": val})
        if val < best_val:
            best_val = val
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload.update(
        {
            "model_state_dict": model.state_dict(),
            "contact_refinement_source_checkpoint": str(args.checkpoint),
            "contact_refinement_labels": str(args.labels),
            "contact_refinement_best_val_mse": best_val,
        }
    )
    torch.save(payload, out)
    return out


def fine_tune_latent_mdn(args: argparse.Namespace, device: torch.device) -> Path:
    keys, refined, _meta = load_refined_labels(args.labels, device)
    mdn_model, autoencoder, latent_stats, normalizer, config, payload = load_latent_mdn_checkpoint(args.checkpoint, device=device)
    refined_norm = normalize(refined, normalizer.mean, normalizer.std)
    with torch.no_grad():
        z = autoencoder.encode(refined_norm)
        z_mean = latent_stats["z_mean"].to(device)
        z_std = latent_stats["z_std"].to(device)
        z_true = (z - z_mean) / z_std
    train_ds, val_ds = split_train_val(keys, refined_norm, args.val_fraction)
    train_z, val_z = split_train_val(keys, z_true, args.val_fraction)
    loader = DataLoader(TensorDataset(train_ds.tensors[0], train_ds.tensors[1], train_z.tensors[1]), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_ds.tensors[0], val_ds.tensors[1], val_z.tensors[1]), batch_size=args.batch_size, shuffle=False)
    for parameter in autoencoder.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.AdamW(mdn_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_state = None
    best_val = float("inf")

    def decoder(z_standardized: torch.Tensor) -> torch.Tensor:
        return autoencoder.decode(z_standardized * z_std.to(z_standardized.device, z_standardized.dtype) + z_mean.to(z_standardized.device, z_standardized.dtype))

    for epoch in range(args.epochs):
        mdn_model.train()
        losses = []
        for xb, pose_b, z_b in tqdm(loader, desc=f"mdn contact epoch {epoch}", leave=False):
            optimizer.zero_grad(set_to_none=True)
            logits, means, log_stds = mdn_model(xb)
            loss_dict = latent_mdn_total_loss(
                logits,
                means,
                log_stds,
                z_b,
                decoder,
                pose_b,
                mdn_nll_weight=args.mdn_nll_weight,
                pose_aux_weight=args.pose_aux_weight,
            )
            loss_dict["loss"].backward()
            torch.nn.utils.clip_grad_norm_(mdn_model.parameters(), args.grad_clip_norm)
            optimizer.step()
            losses.append(float(loss_dict["loss"].detach().cpu()))
        mdn_model.eval()
        vals = []
        with torch.no_grad():
            for xb, pose_b, z_b in val_loader:
                logits, means, log_stds = mdn_model(xb)
                vals.append(
                    float(
                        latent_mdn_total_loss(
                            logits,
                            means,
                            log_stds,
                            z_b,
                            decoder,
                            pose_b,
                            mdn_nll_weight=args.mdn_nll_weight,
                            pose_aux_weight=args.pose_aux_weight,
                        )["loss"].detach().cpu()
                    )
                )
        val = float(np.mean(vals)) if vals else float(np.mean(losses))
        print({"epoch": epoch, "train_contact_loss": float(np.mean(losses)), "val_contact_loss": val})
        if val < best_val:
            best_val = val
            best_state = {key: value.detach().cpu().clone() for key, value in mdn_model.state_dict().items()}
    if best_state is not None:
        mdn_model.load_state_dict(best_state)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload.update(
        {
            "model_state_dict": mdn_model.state_dict(),
            "contact_refinement_source_checkpoint": str(args.checkpoint),
            "contact_refinement_labels": str(args.labels),
            "contact_refinement_best_val_loss": best_val,
        }
    )
    torch.save(payload, out)
    return out


def fine_tune_diffusion(args: argparse.Namespace, device: torch.device) -> Path:
    keys, refined, _meta = load_refined_labels(args.labels, device)
    checkpoint_payload = torch.load(args.checkpoint, map_location=device)
    model, diffusion, config, mean_np, std_np = load_model_for_inference(args.checkpoint, device)
    mean = torch.as_tensor(mean_np, dtype=torch.float32, device=device)
    std = torch.as_tensor(std_np, dtype=torch.float32, device=device)
    target_norm = normalize(refined, mean, std)
    train_ds, val_ds = split_train_val(keys, target_norm, args.val_fraction)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_state = None
    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for xb, x0 in tqdm(loader, desc=f"diffusion contact epoch {epoch}", leave=False):
            timestep = diffusion.sample_timesteps(x0.shape[0])
            noise = torch.randn_like(x0)
            noisy = diffusion.q_sample(x0, timestep, noise)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(noisy, timestep, xb), noise)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        vals = []
        with torch.no_grad():
            for xb, x0 in val_loader:
                timestep = diffusion.sample_timesteps(x0.shape[0])
                noise = torch.randn_like(x0)
                noisy = diffusion.q_sample(x0, timestep, noise)
                vals.append(float(torch.nn.functional.mse_loss(model(noisy, timestep, xb), noise).detach().cpu()))
        val = float(np.mean(vals)) if vals else float(np.mean(losses))
        print({"epoch": epoch, "train_contact_eps_mse": float(np.mean(losses)), "val_contact_eps_mse": val})
        if val < best_val:
            best_val = val
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_payload.update(
        {
            "model": model.state_dict(),
            "ema": {},
            "contact_refinement_source_checkpoint": str(args.checkpoint),
            "contact_refinement_labels": str(args.labels),
            "contact_refinement_best_val_eps_mse": best_val,
        }
    )
    torch.save(checkpoint_payload, out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Variations checkpoints on contact-refined joint labels.")
    parser.add_argument("--model-type", required=True, choices=["mlp_baseline", "latent_mdn", "diffusion"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--mdn-nll-weight", type=float, default=1.0)
    parser.add_argument("--pose-aux-weight", type=float, default=1.0)
    args = parser.parse_args()

    device = resolve_device(str(args.device).strip())
    if args.model_type == "mlp_baseline":
        out = fine_tune_mlp(args, device)
    elif args.model_type == "latent_mdn":
        out = fine_tune_latent_mdn(args, device)
    else:
        out = fine_tune_diffusion(args, device)
    print(f"Wrote contact fine-tuned checkpoint: {out}")


if __name__ == "__main__":
    main()
