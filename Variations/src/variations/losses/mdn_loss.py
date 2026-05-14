from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def mdn_nll_loss(
    mixture_logits: torch.Tensor,
    means: torch.Tensor,
    log_stds: torch.Tensor,
    z_true: torch.Tensor,
) -> torch.Tensor:
    """
    Gaussian mixture negative log likelihood.

    Args:
        mixture_logits: (B, K)
        means:          (B, K, D)
        log_stds:       (B, K, D)
        z_true:         (B, D)
    """
    z = z_true.unsqueeze(1)
    inv_stds = torch.exp(-log_stds)
    normalized = (z - means) * inv_stds
    log_prob_per_dim = -0.5 * normalized.pow(2) - log_stds - 0.5 * math.log(2.0 * math.pi)
    log_prob = log_prob_per_dim.sum(dim=-1)
    log_mix = F.log_softmax(mixture_logits, dim=-1)
    log_prob_mixture = torch.logsumexp(log_mix + log_prob, dim=-1)
    return -log_prob_mixture.mean()

