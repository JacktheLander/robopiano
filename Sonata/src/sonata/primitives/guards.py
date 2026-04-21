from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


def normalize_primitive_storage_aliases(config: dict[str, Any]) -> dict[str, Any]:
    """Map alternate YAML keys to canonical Sonata primitive config keys."""
    if "enable_raw_segment_cache" in config:
        config["save_raw_segment_chunks"] = bool(config["enable_raw_segment_cache"])
    if "write_full_segment_cache" in config and bool(config["write_full_segment_cache"]):
        config["save_raw_segment_chunks"] = True
    config.setdefault("primitive_discovery_method", "gmm_sweep")
    return config


def primitive_count_guard(*, count: int, max_clusters_total: int, context: str = "") -> None:
    if int(count) > int(max_clusters_total):
        msg = (
            f"Primitive count guard: {count} > max_clusters_total={max_clusters_total}. {context}".strip()
        )
        raise RuntimeError(msg)


def fit_row_guard(*, num_rows: int, max_fit_rows: int, context: str = "") -> None:
    if int(num_rows) > int(max_fit_rows):
        raise RuntimeError(
            f"fit_row_guard: {num_rows} training rows exceed max_fit_rows={max_fit_rows}. {context}".strip()
        )


def storage_guard(
    *,
    bytes_written: int,
    soft_limit: int,
    hard_limit: int,
    logger: logging.Logger | None = None,
) -> None:
    log = logger or LOGGER
    if int(bytes_written) >= int(hard_limit):
        raise RuntimeError(
            f"storage_guard (hard): estimated bytes {bytes_written} >= max_storage_bytes_hard={hard_limit}."
        )
    if int(bytes_written) >= int(soft_limit):
        log.warning(
            "storage_guard (soft): estimated bytes %s >= max_storage_bytes_soft=%s.",
            bytes_written,
            soft_limit,
        )


def estimate_dir_bytes(root: Path) -> int:
    total = 0
    if not root.exists():
        return 0
    for path in root.rglob("*"):
        if path.is_file():
            try:
                total += path.stat().st_size
            except OSError:
                continue
    return int(total)


def runtime_guard_projected_walltime(
    *,
    elapsed_seconds: float,
    episodes_done: int,
    episodes_total: int,
    max_walltime_seconds: float,
    logger: logging.Logger | None = None,
) -> None:
    """Fail early if linear projection exceeds allocation."""
    log = logger or LOGGER
    if episodes_done <= 0 or episodes_total <= 0:
        return
    rate = float(elapsed_seconds) / float(episodes_done)
    projected = rate * float(episodes_total)
    remaining = episodes_total - episodes_done
    eta = rate * float(remaining)
    log.info(
        "Runtime estimate: elapsed=%.1fs episodes=%d/%d projected_total=%.1fs eta_remaining=%.1fs",
        elapsed_seconds,
        episodes_done,
        episodes_total,
        projected,
        eta,
    )
    min_samples = min(200, max(50, int(episodes_total * 0.01)))
    if episodes_done < min_samples:
        return
    if projected > float(max_walltime_seconds) * 0.92:
        raise RuntimeError(
            f"runtime_guard: projected walltime {projected:.0f}s exceeds "
            f"0.92 * max_walltime_seconds={max_walltime_seconds:.0f}s after {episodes_done}/{episodes_total} episodes."
        )


def slurm_remaining_seconds() -> float | None:
    end = os.environ.get("SLURM_END_TIME")
    if not end:
        return None
    try:
        end_ts = time.mktime(time.strptime(end, "%Y%m%d%H%M%S"))
        return float(end_ts - time.time())
    except (OSError, ValueError):
        return None
