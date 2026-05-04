"""Planner-focused validation metrics, checkpoint scoring, and collapse heuristics."""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

_LOGGER = logging.getLogger(__name__)


def compute_stage2_planner_score(val_metrics: dict[str, float], config: dict[str, Any]) -> float:
    _ = config
    w_f1 = float(val_metrics.get("primitive_weighted_f1", np.nan))
    rem_acc = float(val_metrics.get("remapped_primitive_accuracy", val_metrics.get("primitive_accuracy", np.nan)))
    d_acc = float(val_metrics.get("duration_accuracy", np.nan))
    y_acc = float(val_metrics.get("dynamics_accuracy", np.nan))
    p_mae = float(val_metrics.get("param_mae", 0.0))
    avg_conf = float(val_metrics.get("primitive_avg_confidence", np.nan))
    p_acc_for_cal = float(val_metrics.get("primitive_accuracy", np.nan))
    calibration_penalty = (
        float(np.clip(avg_conf - p_acc_for_cal, 0.0, 1.0))
        if np.isfinite(avg_conf) and np.isfinite(p_acc_for_cal)
        else 0.0
    )
    scaled_param = float(min(3.0, p_mae) / 3.0) if np.isfinite(p_mae) else 0.0

    dur = d_acc if np.isfinite(d_acc) else 0.0
    dyn = y_acc if np.isfinite(y_acc) else 0.0

    if np.isfinite(w_f1) and np.isfinite(rem_acc):
        score = 0.45 * w_f1 + 0.20 * rem_acc + 0.15 * dur + 0.10 * dyn - 0.05 * scaled_param - 0.05 * calibration_penalty
    else:
        score = 0.60 * (w_f1 if np.isfinite(w_f1) else 0.0) + 0.25 * dur + 0.15 * dyn
    return float(score)




def resolve_checkpoint_monitor(val_metrics: dict[str, float], config: dict[str, Any]) -> tuple[str, float, str]:
    ck = dict(config.get("checkpoint_selection") or {})
    metric = str(ck.get("metric", "loss")).strip()
    mode = str(ck.get("mode", "min")).lower()

    vm = dict(val_metrics)
    if metric in {"stage2_planner_score", "planner_score"}:
        computed = compute_stage2_planner_score(vm, config)
        vm["stage2_planner_score"] = computed
        return "stage2_planner_score", float(computed)

    lookup = metric[4:] if metric.startswith("val/") else metric
    if lookup not in vm:
        _LOGGER.warning("checkpoint_selection.metric=%r missing in validation metrics — using loss/min.", lookup)
        return "loss", float(vm.get("loss", float("inf")))

    return lookup, float(vm[lookup])


def checkpoint_improved(best_value: float, monitor_value: float, min_delta: float, mode: str) -> bool:
    mode_norm = mode.lower().strip()
    if mode_norm == "max":
        return (not np.isfinite(best_value)) or (monitor_value > best_value + min_delta)
    return (not np.isfinite(best_value)) or (monitor_value < best_value - min_delta)


def augment_factored_validation_metrics(
    *,
    primitive_truth: list[int],
    primitive_pred: list[int],
    duration_truth: list[int],
    duration_pred: list[int],
    dynamics_truth: list[int],
    dynamics_pred: list[int],
    family_truth: list[int],
    family_pred: list[int],
    primitive_original_truth: list[int] | None,
    remap_diag: dict[str, Any],
    batches_top1_topk_gap: list[float],
    batches_wrong_conf: list[float],
    transition_snapshots: list[tuple[str, int, int, int]],
    baseline_metrics: dict[str, float],
    config: dict[str, Any],
) -> dict[str, float]:
    """transition_snapshots: (episode_id, timestep, tgt_prim, pred_prim)."""
    out: dict[str, float] = dict(baseline_metrics)
    yt = np.asarray(primitive_truth, dtype=np.int64)
    yp = np.asarray(primitive_pred, dtype=np.int64)

    weak_keep = frozenset(int(x) for x in (remap_diag.get("weak_keep_vocab_indices") or []))

    if yt.size > 0:
        out["primitive_weighted_f1"] = float(f1_score(yt, yp, average="weighted", zero_division=0))
        out["primitive_macro_f1"] = float(f1_score(yt, yp, average="macro", zero_division=0))
        out["primitive_balanced_accuracy"] = float(balanced_accuracy_score(yt, yp))
        out["primitive_accuracy"] = float(accuracy_score(yt, yp))
        out["remapped_primitive_accuracy"] = float(out["primitive_accuracy"])
        out["family_accuracy"] = float(accuracy_score(np.asarray(family_truth, dtype=np.int64), np.asarray(family_pred, dtype=np.int64)))

        if primitive_original_truth and len(primitive_original_truth) == len(primitive_truth):
            yto = np.asarray(primitive_original_truth, dtype=np.int64)
            out["original_primitive_accuracy"] = float(accuracy_score(yto, yp))
            out["remap_affected_fraction"] = float(np.mean(yto != yt))
            out["remapped_target_fraction"] = float(np.mean(yto != yt))
            if weak_keep:
                mask = np.array([int(orig) in weak_keep for orig in yto.flatten()], dtype=np.float32)
                out["weak_keep_fraction"] = float(np.mean(mask)) if mask.size else 0.0
            else:
                out["weak_keep_fraction"] = 0.0
        else:
            out["original_primitive_accuracy"] = float(out["primitive_accuracy"])
            out.setdefault("remap_affected_fraction", 0.0)
            out.setdefault("remapped_target_fraction", 0.0)
            out.setdefault("weak_keep_fraction", 0.0)

        bigram_hits = bigram_total = trans_hits = trans_total = 0
        if len(transition_snapshots) >= 2:
            rows = sorted(transition_snapshots, key=lambda r: (r[0], r[1]))
            for prev, curr in zip(rows, rows[1:]):
                ep0, t0, g0, p0 = prev
                ep1, t1, g1, p1 = curr
                if ep0 == ep1 and t1 == t0 + 1:
                    bigram_total += 1
                    bigram_hits += int((g0, g1) == (p0, p1))
                    tgt_transition = int(g1 != g0)
                    pred_transition = int(p1 != p0)
                    trans_total += 1
                    trans_hits += int(tgt_transition == pred_transition)
        out["primitive_bigram_accuracy"] = float(bigram_hits / max(bigram_total, 1))
        out["transition_accuracy"] = float(trans_hits / max(trans_total, 1))

    if batches_top1_topk_gap:
        out["primitive_top1_topk_gap"] = float(np.mean(batches_top1_topk_gap))

    if batches_wrong_conf:
        out["primitive_wrong_avg_confidence"] = float(np.mean(batches_wrong_conf))
    elif "primitive_wrong_avg_confidence" not in out:
        out["primitive_wrong_avg_confidence"] = float(out.get("primitive_avg_confidence", 0.0))

    out["stage2_planner_score"] = compute_stage2_planner_score(out, config)
    return out


def maybe_warn_primitive_collapsing(history: deque[dict[str, float]], logger: logging.Logger) -> None:
    if len(history) < 3:
        return
    latest = history[-1]
    oldest = history[0]

    def pick(row: dict[str, float], key: str) -> float:
        return float(row.get(key, float("nan")))

    loss_old, loss_new = pick(oldest, "primitive_loss"), pick(latest, "primitive_loss")
    acc_old, acc_new = pick(oldest, "primitive_accuracy"), pick(latest, "primitive_accuracy")
    conf_old, conf_new = pick(oldest, "primitive_avg_confidence"), pick(latest, "primitive_avg_confidence")
    ppl_old, ppl_new = pick(oldest, "primitive_perplexity"), pick(latest, "primitive_perplexity")

    loss_spike = np.isfinite(loss_old) and np.isfinite(loss_new) and loss_new > loss_old * 1.25
    acc_flat = np.isfinite(acc_old) and np.isfinite(acc_new) and (acc_new - acc_old) < 0.01
    conf_up = np.isfinite(conf_old) and np.isfinite(conf_new) and conf_new > conf_old
    ppl_up = np.isfinite(ppl_old) and np.isfinite(ppl_new) and ppl_new > ppl_old

    if loss_spike and acc_flat and conf_up and ppl_up:
        logger.warning(
            "Possible overconfident primitive collapse: confidence rising while primitive validation loss/perplexity worsen."
            " primitive_loss %.4f→%.4f perplexity %.3f→%.3f primitive_acc %.4f→%.4f avg_conf %.4f→%.4f.",
            loss_old,
            loss_new,
            ppl_old,
            ppl_new,
            acc_old,
            acc_new,
            conf_old,
            conf_new,
        )


def batch_primitive_top_gap_and_wrong_conf(
    masked_primitive_logits: torch.Tensor,
    target_primitive: torch.Tensor,
    predicted_primitive: torch.Tensor,
    *,
    topk: int,
) -> tuple[float, float]:
    probs = torch.softmax(masked_primitive_logits, dim=-1)
    kdim = max(2, min(int(topk), probs.shape[-1]))
    vals, _ = probs.topk(kdim, dim=-1)
    gap_mean = float((vals[:, 0] - vals[:, 1:].mean(dim=-1)).mean().item())
    mism = ~predicted_primitive.eq(target_primitive)
    if mism.any():
        wc = float(probs.max(dim=-1).values[mism].mean().item())
    else:
        wc = float(probs.max(dim=-1).values.mean().item())
    return gap_mean, wc
