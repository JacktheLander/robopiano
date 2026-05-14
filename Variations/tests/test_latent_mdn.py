from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from variations.inference.predict_press_pose import HandStateNormalizer, predict_with_latent_mdn
from variations.losses.mdn_loss import mdn_nll_loss
from variations.models.latent_mdn import LatentMDN
from variations.models.pose_autoencoder import PoseAutoencoder


def test_pose_autoencoder_shapes():
    batch = 4
    x = torch.randn(batch, 46)
    model = PoseAutoencoder(input_dim=46, latent_dim=12)
    recon, z = model(x)
    assert recon.shape == (batch, 46)
    assert z.shape == (batch, 12)
    assert torch.isfinite(recon).all()
    assert torch.isfinite(z).all()


def test_latent_mdn_shapes():
    batch = 4
    x = torch.randn(batch, 88)
    model = LatentMDN(input_dim=88, latent_dim=12, num_components=5)
    logits, means, log_stds = model(x)
    assert logits.shape == (batch, 5)
    assert means.shape == (batch, 5, 12)
    assert log_stds.shape == (batch, 5, 12)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(means).all()
    assert torch.isfinite(log_stds).all()


def test_mdn_loss_backward():
    batch = 4
    components = 5
    dim = 12
    logits = torch.randn(batch, components, requires_grad=True)
    means = torch.randn(batch, components, dim, requires_grad=True)
    raw_log_stds = torch.randn(batch, components, dim, requires_grad=True)
    log_stds = raw_log_stds.clamp(-6.0, 2.0)
    z_true = torch.randn(batch, dim)
    loss = mdn_nll_loss(logits, means, log_stds, z_true)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert means.grad is not None


def test_latent_mdn_best_component_inference():
    batch = 4
    mdn = LatentMDN(input_dim=88, latent_dim=12, num_components=3)
    autoencoder = PoseAutoencoder(input_dim=46, latent_dim=12)
    normalizer = HandStateNormalizer(mean=torch.zeros(46), std=torch.ones(46))
    latent_stats = {"z_mean": torch.zeros(12), "z_std": torch.ones(12)}
    out = predict_with_latent_mdn(torch.randn(batch, 88), mdn, autoencoder, latent_stats, normalizer)
    assert out.shape == (batch, 46)
    assert torch.isfinite(out).all()
