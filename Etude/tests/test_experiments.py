from __future__ import annotations

import json
from pathlib import Path

from etude.experiments import (
    build_execution_plan,
    load_experiment_config,
    summarize_result_tree,
    validate_experiment_config,
    write_manifest,
    write_summary_csv,
)


def test_load_experiment_config_merges_includes() -> None:
    config_path = Path("configs/experiments/03_key_aware_clean.yaml").resolve()
    config = load_experiment_config(config_path)

    assert config["experiment"]["name"] == "key_aware_clean"
    assert config["controller"]["type"] == "key_aware_residual"
    assert config["loss"]["key_press_weight"] == 2.0
    assert config["features"]["blocks"] == ["etude.features.key_blocks:build_key_features"]


def test_build_execution_plan_for_rollout_config() -> None:
    config_path = Path("configs/experiments/10_rollout_eval.yaml").resolve()
    config = load_experiment_config(config_path)
    errors, warnings = validate_experiment_config(config, config_path=config_path)
    plan = build_execution_plan(config, config_path=config_path)

    assert errors == []
    assert plan.execution_kind == "rollout_eval"
    assert plan.requires_rollout is True
    assert plan.local_batch_script == "local_batch/run_etude_rollout_eval.echo"
    assert any("--trajectory" in part or "CHANGE_ME" in part for part in plan.command)
    assert warnings == []


def test_summarize_result_tree_uses_manifest_metadata(tmp_path: Path) -> None:
    output_dir = tmp_path / "runs" / "train" / "demo"
    output_dir.mkdir(parents=True)
    write_manifest(
        build_execution_plan(
            {
                "experiment": {
                    "name": "demo",
                    "controller_family": "pd_grouped",
                    "evaluation_mode": "offline_eval",
                    "resource_tier": "tier1_cpu_offline",
                },
                "controller": {"type": "scheduled_pd"},
                "execution": {"kind": "offline_pd_eval"},
                "output": {"root": str(tmp_path / "runs"), "subdir": "train/demo"},
            },
            config_path=(tmp_path / "demo.yaml"),
        ),
        output_dir / "experiment_manifest.json",
        source_config={"demo": True},
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "piano/event_f1": 0.9,
                "piano/missed_events": 1,
                "piano/false_events": 2,
                "piano/timing_abs_error_mean_s": 0.03,
            }
        ),
        encoding="utf-8",
    )

    rows = summarize_result_tree(tmp_path / "runs")
    out_path = write_summary_csv(rows, tmp_path / "summary.csv")

    assert len(rows) == 1
    assert rows[0]["experiment_name"] == "demo"
    assert rows[0]["event_f1"] == 0.9
    assert "demo" in out_path.read_text(encoding="utf-8")
