from __future__ import annotations

import torch
import torch.nn.functional as F


def supervised_pose_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    joints_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    joint_loss = F.mse_loss(pred, target)
    total = float(joints_weight) * joint_loss
    return {
        "loss": total,
        "joint_mse": joint_loss.detach(),
    }
