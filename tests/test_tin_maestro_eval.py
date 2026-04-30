from __future__ import annotations

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tin.maestro_eval import (
    MaestroEvalConfig,
    append_jsonl_row,
    build_summary,
    discover_midi_files,
    evaluate_piece_batch,
    load_jsonl_rows,
    piece_id_from_path,
)
from tin.evaluate_maestro import build_parser


def test_discover_midi_files_and_piece_ids(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    (dataset_root / "b").mkdir(parents=True)
    (dataset_root / "z.mid").write_text("a", encoding="utf-8")
    (dataset_root / "b" / "x.midi").write_text("b", encoding="utf-8")
    (dataset_root / "ignore.txt").write_text("c", encoding="utf-8")

    midi_files = discover_midi_files(dataset_root)

    assert [piece_id_from_path(path, dataset_root) for path in midi_files] == ["b/x.midi", "z.mid"]


def test_load_jsonl_rows_skips_invalid_lines(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text('{"piece_id":"ok","status":"completed"}\n{"piece_id"', encoding="utf-8")

    rows = load_jsonl_rows(path)

    assert rows == [{"piece_id": "ok", "status": "completed"}]


def test_evaluate_piece_batch_resumes_and_continues_after_errors(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "out"
    dataset_root.mkdir()
    pieces = []
    for name in ("a.mid", "b.mid", "c.mid"):
        path = dataset_root / name
        path.write_text(name, encoding="utf-8")
        pieces.append(path)

    append_jsonl_row(
        output_root / "piece_metrics.jsonl",
        {
            "piece_id": "a.mid",
            "midi_path": str((dataset_root / "a.mid").resolve()),
            "status": "completed",
            "f1": 0.25,
        },
    )

    seen: list[str] = []

    def runner(piece_path: Path, piece_id: str) -> dict[str, object]:
        seen.append(piece_id)
        if piece_path.name == "b.mid":
            raise RuntimeError("boom")
        return {"status": "completed", "f1": 0.75, "precision": 0.7, "recall": 0.8}

    payload = evaluate_piece_batch(
        dataset_root=dataset_root,
        piece_paths=pieces,
        output_root=output_root,
        runner=runner,
        resume=True,
    )

    assert seen == ["b.mid", "c.mid"]
    rows = load_jsonl_rows(output_root / "piece_metrics.jsonl")
    assert len(rows) == 3
    assert rows[1]["status"] == "error"
    assert rows[2]["status"] == "completed"
    assert payload["processed_now"] == 2
    assert payload["skipped_existing"] == 1
    assert (output_root / "piece_metrics.csv").exists()
    assert (output_root / "summary.json").exists()
    assert (output_root / "f1_histogram.png").exists()

    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    assert summary["completed_rows"] == 2
    assert summary["failed_rows"] == 1
    assert summary["f1_count"] == 2


def test_build_summary_uses_completed_rows_only() -> None:
    summary = build_summary(
        [
            {"status": "completed", "f1": 0.2},
            {"status": "error", "f1": 0.9},
            {"status": "completed", "f1": 0.6},
        ]
    )

    assert summary["completed_rows"] == 2
    assert summary["failed_rows"] == 1
    assert summary["f1_count"] == 2
    assert summary["f1_mean"] == 0.4


def test_evaluate_maestro_parser_accepts_droq_backend_flags() -> None:
    args = build_parser().parse_args(
        [
            "--max-steps-per-song",
            "1000",
            "--agent-backend",
            "droq",
            "--utd-ratio",
            "12",
            "--compile-models",
            "--normalize-observations",
            "--bc-checkpoint",
            "/tmp/bc.pt",
            "--use-mjx",
            "--n-mjx-envs",
            "4",
        ]
    )

    assert args.agent_backend == "droq"
    assert args.utd_ratio == 12
    assert args.compile_models is True
    assert args.normalize_observations is True
    assert args.bc_checkpoint == "/tmp/bc.pt"
    assert args.use_mjx is True
    assert args.n_mjx_envs == 4


def test_maestro_eval_config_disables_intermediate_evals_by_default() -> None:
    config = MaestroEvalConfig(max_steps_per_song=1000)

    assert config.run_intermediate_evals is False
