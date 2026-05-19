from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from etude.experiments import (  # noqa: E402
    build_execution_plan,
    load_experiment_config,
    materialize_runtime_config,
    validate_experiment_config,
    write_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or dry-run an Etude experiment config.")
    parser.add_argument("--config", required=True, help="Experiment config YAML under configs/experiments.")
    parser.add_argument("--output-root", default=None, help="Override ETUDE_OUTPUT_ROOT for this invocation.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config structure and print the planned action without loading datasets or running training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_experiment_config(config_path)
    errors, warnings = validate_experiment_config(config, config_path=config_path)
    plan = build_execution_plan(config, config_path=config_path, output_root_override=args.output_root)

    status = "ok"
    if args.dry_run:
        status = "dry_run_ready" if plan.command else "dry_run_only"

    payload = {
        "config": str(config_path),
        "experiment_name": plan.experiment_name,
        "controller_family": plan.controller_family,
        "evaluation_mode": plan.evaluation_mode,
        "resource_tier": plan.resource_tier,
        "output_dir": plan.output_dir,
        "dry_run": args.dry_run,
        "execution_status": status,
        "planned_command": plan.command,
        "warnings": warnings,
        "errors": errors,
    }

    if errors:
        print(json.dumps(payload, indent=2))
        raise SystemExit(1)

    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return

    if not plan.allow_execute:
        payload["execution_status"] = "dry_run_only"
        print(json.dumps(payload, indent=2))
        raise SystemExit("This config is marked dry-run-only. Use --dry-run or update execution.allow_execute.")

    if not plan.command:
        payload["execution_status"] = "unsupported_execute"
        print(json.dumps(payload, indent=2))
        raise SystemExit("No executable command is configured for this experiment.")

    output_dir = Path(plan.output_dir)
    resolved_config_path = output_dir / "resolved_config.yaml"
    materialize_runtime_config(config, resolved_config_path)
    write_manifest(plan, output_dir / "experiment_manifest.json", source_config=config)

    command = [part if part != "{resolved_config}" else str(resolved_config_path) for part in plan.command]
    payload["execution_status"] = "running"
    payload["planned_command"] = command
    print(json.dumps(payload, indent=2))
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
