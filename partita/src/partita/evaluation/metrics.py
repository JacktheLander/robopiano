from __future__ import annotations

import numpy as np


def key_metrics(goals, piano_states, threshold: float = 0.5) -> dict[str, float]:
    if goals is None or piano_states is None:
        return {
            "key_precision": float("nan"),
            "key_recall": float("nan"),
            "key_f1": float("nan"),
            "mispress_rate": float("nan"),
        }
    goal = np.asarray(goals) > threshold
    played = np.asarray(piano_states) > threshold
    if goal.shape != played.shape:
        n = min(goal.shape[-1], played.shape[-1])
        goal = goal[..., :n]
        played = played[..., :n]
    tp = np.logical_and(goal, played).sum(dtype=np.float64)
    fp = np.logical_and(~goal, played).sum(dtype=np.float64)
    fn = np.logical_and(goal, ~played).sum(dtype=np.float64)
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    mispress = fp / max(played.sum(dtype=np.float64), 1.0)
    return {
        "key_precision": float(precision),
        "key_recall": float(recall),
        "key_f1": float(f1),
        "mispress_rate": float(mispress),
    }


def action_smoothness(actions) -> float:
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim < 2 or arr.shape[0] < 2:
        return 0.0
    diff = np.diff(arr, axis=0)
    return float(np.mean(diff * diff))


def action_metrics(original, reconstructed) -> dict[str, float]:
    orig = np.asarray(original, dtype=np.float32)
    recon = np.asarray(reconstructed, dtype=np.float32)
    t = min(orig.shape[0], recon.shape[0])
    d = min(orig.shape[-1], recon.shape[-1])
    orig = orig[:t, :d]
    recon = recon[:t, :d]
    mse = float(np.mean((orig - recon) ** 2))
    l1 = float(np.mean(np.abs(orig - recon)))
    smooth_orig = action_smoothness(orig)
    smooth_recon = action_smoothness(recon)
    ratio = float(smooth_recon / max(smooth_orig, 1e-12))
    return {
        "action_mse": mse,
        "action_l1": l1,
        "action_smoothness_original": smooth_orig,
        "action_smoothness_reconstructed": smooth_recon,
        "action_smoothness_ratio": ratio,
    }


def trajectory_score(actions, goals, piano_states, threshold: float = 0.5) -> dict[str, float]:
    km = key_metrics(goals, piano_states, threshold=threshold)
    smooth = action_smoothness(actions)
    key_f1 = km.get("key_f1", float("nan"))
    mispress = km.get("mispress_rate", float("nan"))
    if np.isfinite(key_f1) and np.isfinite(mispress):
        score = 2.0 * key_f1 - mispress - 0.05 * smooth
    else:
        score = -0.05 * smooth
    km["action_smoothness"] = float(smooth)
    km["score"] = float(score)
    return km
