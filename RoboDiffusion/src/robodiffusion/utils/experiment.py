from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from robodiffusion.utils.io import ensure_dir


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
    prefix = f"RoboDiffusion-{stage}-{experiment_name}-seed{seed}"
    if resume:
        candidates = sorted(base.glob(f"{prefix}*"))
        run_root = candidates[-1] if candidates else base / f"{prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    else:
        run_root = base / f"{prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    return RunPaths(
        root=run_root,
        checkpoints=ensure_dir(run_root / "checkpoints"),
        metrics=ensure_dir(run_root / "metrics"),
        plots=ensure_dir(run_root / "plots"),
        logs=ensure_dir(run_root / "logs"),
        artifacts=ensure_dir(run_root / "artifacts"),
    )
