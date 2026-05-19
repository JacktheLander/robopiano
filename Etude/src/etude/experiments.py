from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from etude.config import deep_update, load_config
from etude.utils.import_utils import load_symbol


@dataclass
class ExecutionPlan:
    config_path: str
    config_name: str
    experiment_name: str
    controller_family: str
    evaluation_mode: str
    resource_tier: str
    execution_kind: str
    output_root: str
    output_dir: str
    local_batch_script: str | None
    requires_dataset: bool
    requires_gpu: bool
    requires_rollout: bool
    dry_run_support: bool
    allow_execute: bool
    local_safe: bool
    command: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def runs_root() -> Path:
    return repo_root() / "runs"


def resolve_config_reference(reference: str | Path, relative_to: Path) -> Path:
    candidate = Path(reference)
    if candidate.is_absolute():
        return candidate.resolve()
    return (relative_to / candidate).resolve()


def load_experiment_config(path: str | Path, *, _seen: set[Path] | None = None) -> dict[str, Any]:
    config_path = Path(path).resolve()
    seen = set() if _seen is None else set(_seen)
    if config_path in seen:
        raise ValueError(f"Experiment config include cycle detected at {config_path}")
    seen.add(config_path)

    raw = load_config(config_path)
    includes = raw.pop("includes", [])
    if includes is None:
        includes = []
    if not isinstance(includes, list):
        raise ValueError(f"'includes' must be a list in {config_path}")

    merged: dict[str, Any] = {}
    for include in includes:
        include_path = resolve_config_reference(include, config_path.parent)
        merged = deep_update(merged, load_experiment_config(include_path, _seen=seen))

    return deep_update(merged, raw)


def build_execution_plan(
    config: dict[str, Any],
    *,
    config_path: Path,
    output_root_override: str | None = None,
) -> ExecutionPlan:
    experiment = _as_dict(config.get("experiment"))
    execution = _as_dict(config.get("execution"))
    output = _as_dict(config.get("output"))
    controller = _as_dict(config.get("controller"))

    experiment_name = str(experiment.get("name") or config_path.stem)
    controller_family = str(
        experiment.get("controller_family")
        or controller.get("family")
        or controller.get("type")
        or "unknown"
    )
    evaluation_mode = str(experiment.get("evaluation_mode") or execution.get("mode") or "unknown")
    resource_tier = str(experiment.get("resource_tier") or execution.get("resource_tier") or "unspecified")
    execution_kind = str(execution.get("kind") or "dry_run_only")
    batch_script = execution.get("local_batch_script")

    root_value = (
        output_root_override
        or os.environ.get("ETUDE_OUTPUT_ROOT")
        or output.get("root")
        or str(runs_root())
    )
    output_root = Path(root_value)
    if not output_root.is_absolute():
        output_root = (repo_root() / output_root).resolve()
    output_subdir = str(output.get("subdir") or experiment_name)
    output_dir = (output_root / output_subdir).resolve()

    command = build_command(config, execution_kind=execution_kind, output_dir=output_dir)

    return ExecutionPlan(
        config_path=str(config_path),
        config_name=config_path.name,
        experiment_name=experiment_name,
        controller_family=controller_family,
        evaluation_mode=evaluation_mode,
        resource_tier=resource_tier,
        execution_kind=execution_kind,
        output_root=str(output_root),
        output_dir=str(output_dir),
        local_batch_script=str(batch_script) if batch_script else None,
        requires_dataset=bool(execution.get("requires_dataset", False)),
        requires_gpu=bool(execution.get("requires_gpu", False)),
        requires_rollout=bool(execution.get("requires_rollout", False)),
        dry_run_support=bool(execution.get("dry_run_support", True)),
        allow_execute=bool(execution.get("allow_execute", False)),
        local_safe=bool(execution.get("local_safe", False)),
        command=command,
    )


def validate_experiment_config(config: dict[str, Any], *, config_path: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    controller = _as_dict(config.get("controller"))
    execution = _as_dict(config.get("execution"))
    output = _as_dict(config.get("output"))

    if not (controller.get("family") or controller.get("type")):
        errors.append("controller.family or controller.type is required")
    if not _as_dict(config.get("experiment")).get("name"):
        warnings.append("experiment.name not set; falling back to config file stem")
    if not execution.get("kind"):
        warnings.append("execution.kind not set; defaulting to dry_run_only")
    if not output.get("subdir"):
        warnings.append("output.subdir not set; falling back to experiment name")

    model_module = controller.get("model_module")
    if model_module:
        try:
            load_symbol(str(model_module))
        except Exception as exc:  # pragma: no cover - defensive
            errors.append(f"failed to import controller.model_module '{model_module}': {exc}")

    feature_blocks = _as_dict(config.get("features")).get("blocks", [])
    if feature_blocks and not isinstance(feature_blocks, list):
        errors.append("features.blocks must be a list when present")
    for block in feature_blocks if isinstance(feature_blocks, list) else []:
        try:
            load_symbol(str(block))
        except Exception as exc:  # pragma: no cover - defensive
            errors.append(f"failed to import features.blocks entry '{block}': {exc}")

    if execution.get("allow_execute") and execution.get("kind") in {"dry_run_only", "smoke"}:
        warnings.append("execution.allow_execute is true for a dry-run-oriented config")

    plan = build_execution_plan(config, config_path=config_path)
    if not Path(plan.output_root).is_absolute():
        errors.append("resolved output root must be absolute")

    return errors, warnings


def build_command(config: dict[str, Any], *, execution_kind: str, output_dir: Path) -> list[str]:
    execution = _as_dict(config.get("execution"))
    output_dir_str = str(output_dir)
    config_placeholder = "{resolved_config}"
    python_exe = os.environ.get("PYTHON", "python")

    if execution_kind == "train_supervised":
        command = [
            python_exe,
            "scripts/train_controller.py",
            "--config",
            config_placeholder,
            "--output-root",
            output_dir_str,
        ]
        if bool(_as_dict(config.get("wandb")).get("enabled", False)):
            command.append("--wandb")
        return command

    if execution_kind == "offline_pd_eval":
        dataset_root = _as_dict(config.get("data")).get("dataset_root")
        command = [
            python_exe,
            "scripts/tune_pd.py",
            "--config",
            config_placeholder,
            "--dataset-root",
            str(dataset_root or ""),
            "--output-root",
            output_dir_str,
        ]
        return command

    if execution_kind == "rollout_eval":
        args = _as_dict(execution.get("args"))
        command = [
            python_exe,
            "scripts/evaluate_tracker.py",
            "--config",
            config_placeholder,
            "--output-root",
            output_dir_str,
        ]
        checkpoint = args.get("checkpoint")
        trajectory = args.get("trajectory")
        backend = args.get("backend")
        if checkpoint:
            command.extend(["--checkpoint", str(checkpoint)])
        if trajectory:
            command.extend(["--trajectory", str(trajectory)])
        if backend:
            command.extend(["--backend", str(backend)])
        if bool(args.get("render_video", False)):
            command.append("--render-video")
        return command

    return []


def materialize_runtime_config(config: dict[str, Any], destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return destination


def write_manifest(plan: ExecutionPlan, destination: Path, *, source_config: dict[str, Any]) -> Path:
    payload = plan.to_dict()
    payload["source_config"] = source_config
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination


def find_metric_files(root: str | Path) -> list[Path]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    matches: list[Path] = []
    for pattern in ("metrics.json", "*metrics.json", "*summary.json", "*metrics.csv", "*summary.csv"):
        matches.extend(root_path.rglob(pattern))
    deduped: dict[str, Path] = {str(path.resolve()): path for path in matches if path.is_file()}
    return sorted(deduped.values())


def summarize_result_tree(root: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric_file in find_metric_files(root):
        if metric_file.suffix == ".json":
            payload = _load_json_metrics(metric_file)
            if payload is None:
                continue
            rows.append(_metric_payload_to_row(metric_file, payload))
        elif metric_file.suffix == ".csv":
            rows.extend(_load_csv_rows(metric_file))
    return rows


def write_summary_csv(rows: list[dict[str, Any]], out_path: str | Path) -> Path:
    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "experiment_name",
        "config_name",
        "controller_family",
        "evaluation_mode",
        "resource_tier",
        "metric_source",
        "event_f1",
        "frame_f1",
        "missed_events",
        "false_events",
        "timing_error_s",
    ]

    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return destination


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _load_json_metrics(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _load_csv_rows(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            source_rows = list(reader)
    except Exception:
        return []

    if not source_rows:
        return []

    manifest = _load_manifest(path.parent)
    rows: list[dict[str, Any]] = []
    for source_row in source_rows:
        rows.append(
            {
                "experiment_name": manifest.get("experiment_name", path.parent.name),
                "config_name": manifest.get("config_name", ""),
                "controller_family": manifest.get("controller_family", ""),
                "evaluation_mode": manifest.get("evaluation_mode", ""),
                "resource_tier": manifest.get("resource_tier", ""),
                "metric_source": str(path),
                "event_f1": source_row.get("event_f1", source_row.get("piano/event_f1", "")),
                "frame_f1": source_row.get(
                    "frame_f1",
                    source_row.get("piano/frame_f1", source_row.get("frame_f1_score", "")),
                ),
                "missed_events": source_row.get("missed_events", source_row.get("piano/missed_events", "")),
                "false_events": source_row.get("false_events", source_row.get("piano/false_events", "")),
                "timing_error_s": source_row.get(
                    "timing_error_s",
                    source_row.get("piano/timing_abs_error_mean_s", ""),
                ),
            }
        )
    return rows


def _metric_payload_to_row(metric_file: Path, payload: dict[str, Any]) -> dict[str, Any]:
    manifest = _load_manifest(metric_file.parent)
    return {
        "experiment_name": manifest.get("experiment_name", metric_file.parent.name),
        "config_name": manifest.get("config_name", ""),
        "controller_family": manifest.get("controller_family", ""),
        "evaluation_mode": manifest.get("evaluation_mode", ""),
        "resource_tier": manifest.get("resource_tier", ""),
        "metric_source": str(metric_file),
        "event_f1": payload.get("event_f1", payload.get("piano/event_f1", "")),
        "frame_f1": payload.get(
            "frame_f1",
            payload.get("piano/frame_f1", payload.get("tracking/frame_f1", "")),
        ),
        "missed_events": payload.get("missed_events", payload.get("piano/missed_events", "")),
        "false_events": payload.get("false_events", payload.get("piano/false_events", "")),
        "timing_error_s": payload.get(
            "timing_error_s",
            payload.get("piano/timing_abs_error_mean_s", ""),
        ),
    }


def _load_manifest(directory: Path) -> dict[str, Any]:
    manifest_path = directory / "experiment_manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}
