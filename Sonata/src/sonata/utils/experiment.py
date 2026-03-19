from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sonata.utils.io import ensure_dir


@dataclass
class RunPaths:
    root: Path
    checkpoints: Path
    metrics: Path
    plots: Path
    logs: Path
    artifacts: Path


def make_run_paths(output_root: str | Path, stage: str, experiment_name: str, seed: int, resume: bool = False) -> RunPaths:
    base = ensure_dir(output_root)
    prefix = f"Sonata-3-{stage}-{experiment_name}-seed{seed}"
    if resume:
        candidates = sorted(base.glob(f"{prefix}*"))
        if candidates:
            run_root = candidates[-1]
        else:
            run_root = base / f"{prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    else:
        run_root = base / f"{prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    checkpoints = ensure_dir(run_root / "checkpoints")
    metrics = ensure_dir(run_root / "metrics")
    plots = ensure_dir(run_root / "plots")
    logs = ensure_dir(run_root / "logs")
    artifacts = ensure_dir(run_root / "artifacts")
    return RunPaths(root=run_root, checkpoints=checkpoints, metrics=metrics, plots=plots, logs=logs, artifacts=artifacts)
