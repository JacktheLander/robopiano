from __future__ import annotations

import numpy as np


def joint_metrics(q: np.ndarray, q_ref: np.ndarray, qdot: np.ndarray | None = None, qdot_ref: np.ndarray | None = None) -> dict[str, float]:
    q = np.asarray(q, dtype=np.float32)
    q_ref = np.asarray(q_ref, dtype=np.float32)
    metrics = {
        "tracking/joint_mse": float(np.mean((q - q_ref) ** 2)),
        "tracking/joint_mae": float(np.mean(np.abs(q - q_ref))),
    }
    if qdot is not None and qdot_ref is not None:
        metrics["tracking/qvel_mse"] = float(
            np.mean((np.asarray(qdot, dtype=np.float32) - np.asarray(qdot_ref, dtype=np.float32)) ** 2)
        )
    return metrics


def fingertip_metrics(fingertips: np.ndarray, fingertips_ref: np.ndarray) -> dict[str, float]:
    err = np.asarray(fingertips, dtype=np.float32) - np.asarray(fingertips_ref, dtype=np.float32)
    mse = float(np.mean(err**2))
    return {
        "tracking/fingertip_mse": mse,
        "tracking/fingertip_mean_cm": float(np.mean(np.linalg.norm(err.reshape(err.shape[0], -1, 3), axis=-1)) * 100.0),
    }


def action_metrics(actions: np.ndarray, low: np.ndarray | None = None, high: np.ndarray | None = None) -> dict[str, float]:
    actions = np.asarray(actions, dtype=np.float32)
    metrics = {
        "control/action_l2": float(np.mean(np.linalg.norm(actions, axis=-1))),
        "control/action_smoothness": float(np.mean(np.diff(actions, axis=0) ** 2)) if actions.shape[0] > 1 else 0.0,
    }
    if low is not None and high is not None:
        low = np.asarray(low, dtype=np.float32)
        high = np.asarray(high, dtype=np.float32)
        clipped = np.isclose(actions, low) | np.isclose(actions, high)
        metrics["control/action_clip_rate"] = float(np.mean(clipped))
    return metrics


def note_metrics(predicted_keys: np.ndarray, target_keys: np.ndarray) -> dict[str, float]:
    pred = np.asarray(predicted_keys, dtype=bool)
    target = np.asarray(target_keys, dtype=bool)
    tp = float(np.logical_and(pred, target).sum())
    fp = float(np.logical_and(pred, ~target).sum())
    fn = float(np.logical_and(~pred, target).sum())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "piano/note_precision": precision,
        "piano/note_recall": recall,
        "piano/note_f1": f1,
        "piano/missed_notes": fn,
        "piano/extra_notes": fp,
    }
