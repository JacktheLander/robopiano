"""Losses for supervised Variations baselines."""

from variations.losses.latent_mdn_loss import latent_mdn_total_loss, select_best_component_mean
from variations.losses.mdn_loss import mdn_nll_loss
from variations.losses.supervised_pose_loss import supervised_pose_loss

__all__ = ["latent_mdn_total_loss", "mdn_nll_loss", "select_best_component_mean", "supervised_pose_loss"]
