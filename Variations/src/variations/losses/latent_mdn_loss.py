from __future__ import annotations

import torch
import torch.nn.functional as F

from variations.losses.mdn_loss import mdn_nll_loss


def select_best_component_mean(mixture_logits: torch.Tensor, means: torch.Tensor) -> torch.Tensor:
    best_idx = torch.argmax(mixture_logits, dim=-1)
    batch_idx = torch.arange(means.shape[0], device=means.device)
    return means[batch_idx, best_idx]


def latent_mdn_total_loss(
    mixture_logits: torch.Tensor,
    means: torch.Tensor,
    log_stds: torch.Tensor,
    z_true: torch.Tensor,
    decoder,
    joint_state_true: torch.Tensor,
    *,
    mdn_nll_weight: float = 1.0,
    pose_aux_weight: float = 0.5,
) -> dict[str, torch.Tensor]:
    nll = mdn_nll_loss(mixture_logits, means, log_stds, z_true)
    z_best = select_best_component_mean(mixture_logits, means)
    joint_pred = decoder(z_best)
    joint_mse = F.mse_loss(joint_pred, joint_state_true)
    total = float(mdn_nll_weight) * nll + float(pose_aux_weight) * joint_mse
    return {
        "loss": total,
        "nll": nll.detach(),
        "joint_mse": joint_mse.detach(),
    }
