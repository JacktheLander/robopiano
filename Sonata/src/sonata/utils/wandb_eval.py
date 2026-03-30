from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from sonata.utils.wandb import WandbRun


def safe_init_eval_wandb_run(
    *,
    config: dict[str, Any] | None,
    run_name: str,
    config_payload: dict[str, Any],
    logger: logging.Logger | None = None,
    group: str | None = None,
    tags: list[str] | None = None,
) -> WandbRun | None:
    effective_config = dict(config or {})
    if group and not effective_config.get("group"):
        effective_config["group"] = group
    try:
        return WandbRun(
            effective_config,
            run_name=run_name,
            config_payload=config_payload,
            logger=logger,
            job_type="evaluation",
            tags=tags,
        )
    except Exception as exc:  # pragma: no cover
        if logger is not None:
            logger.warning("Evaluation W&B logging disabled after initialization failure: %s", exc)
        return None


def log_prefixed_metrics(
    run: WandbRun | None,
    payload: dict[str, Any],
    *,
    prefix: str | None = None,
    summary: bool = False,
    step: int | None = None,
) -> None:
    if run is None or not run.active:
        return
    flat = _flatten_payload(payload, prefix=prefix)
    if not flat:
        return
    run.log(flat, step=step)
    if summary:
        run.summary(flat)


def log_rollout_table(
    run: WandbRun | None,
    *,
    key: str,
    dataframe: pd.DataFrame,
    logger: logging.Logger | None = None,
) -> None:
    if run is None or not run.active or dataframe.empty:
        return
    try:
        table = run.make_table(dataframe=_sanitize_dataframe(dataframe))
        if table is not None:
            run.log_raw({key: table})
    except Exception as exc:  # pragma: no cover
        if logger is not None:
            logger.warning("Failed to log rollout table to W&B: %s", exc)


def log_rollout_video(
    run: WandbRun | None,
    *,
    key: str,
    video_path: str | Path,
    caption: str,
    fps: int,
    logger: logging.Logger | None = None,
) -> None:
    if run is None or not run.active:
        return
    path = Path(video_path).resolve()
    if not path.exists():
        return
    try:
        video = run.make_video(path, caption=caption, fps=fps, format=path.suffix.lstrip("."))
        if video is not None:
            run.log_raw({key: video})
    except Exception as exc:  # pragma: no cover
        if logger is not None:
            logger.warning("Failed to log rollout video `%s` to W&B: %s", path, exc)


def finish_eval_wandb_run(run: WandbRun | None) -> None:
    if run is not None:
        run.finish()


def _flatten_payload(payload: dict[str, Any], *, prefix: str | None = None) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in payload.items():
        scoped_key = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_payload(value, prefix=scoped_key))
            continue
        scalar = _coerce_scalar(value)
        if scalar is not None:
            flat[scoped_key] = scalar
    return flat


def _coerce_scalar(value: Any) -> Any | None:
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    return None


def _sanitize_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    cleaned = dataframe.copy()
    for column in cleaned.columns:
        if cleaned[column].dtype == object:
            cleaned[column] = cleaned[column].map(_sanitize_table_value)
    return cleaned


def _sanitize_table_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (bool, int, float, str)):
        return value
    return str(value)
