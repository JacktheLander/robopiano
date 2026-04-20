from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sonata.data.indexer import index_external_midi_dataset
from sonata.data.loading import load_episode_record
from sonata.data.schema import ScoreEvent
from sonata.evaluation.external_midi import (
    ExternalReferenceStats,
    build_external_planner_sample,
    build_external_segment_rows,
    compute_external_reference_stats,
    extract_hand_joints,
    normalize_external_benchmark_split,
    resolve_external_manifest_base,
    resolve_external_benchmark_name,
)
from sonata.utils import robopianist as robopianist_utils


class _Logger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str, *args: object) -> None:
        self.messages.append(message % args if args else message)

    def warning(self, message: str, *args: object) -> None:
        self.messages.append(message % args if args else message)


def test_index_external_midi_dataset_indexes_recursive_files(tmp_path: Path, monkeypatch) -> None:
    dataset_root = tmp_path / "midi_corpus"
    nested = dataset_root / "nested"
    nested.mkdir(parents=True, exist_ok=True)
    for path in (dataset_root / "alpha.mid", nested / "alpha.mid", dataset_root / "skip.midi"):
        path.write_bytes(b"MThd")

    def _fake_load_note_events(
        note_path: str | Path,
        control_timestep: float,
        chord_tolerance_steps: int,
        song_id: str | None = None,
        episode_id: str | None = None,
    ) -> list[ScoreEvent]:
        del control_timestep, chord_tolerance_steps
        path = Path(note_path)
        if path.stem == "skip":
            raise ValueError("bad midi")
        return [
            ScoreEvent(
                event_id=f"{episode_id}_0",
                song_id=song_id or path.stem,
                episode_id=episode_id or path.stem,
                onset_step=0,
                end_step=12,
                start_time_sec=0.0,
                end_time_sec=0.6,
                key_numbers=(39,),
                chord_size=1,
                key_center=39.0 / 87.0,
                inter_onset_steps=0,
                source="midi",
            )
        ]

    monkeypatch.setattr("sonata.data.indexer.load_note_events", _fake_load_note_events)
    outputs = index_external_midi_dataset(
        config={
            "dataset_root": str(dataset_root),
            "output_root": str(tmp_path / "outputs"),
            "manifest_name": "external_midi_manifest",
            "split_name": "external_midi_splits",
            "summary_name": "external_midi_summary.json",
            "control_timestep": 0.05,
            "recursive": True,
            "force": True,
        },
        logger=_Logger(),
    )

    manifest_df = pd.read_csv(outputs["manifest_base"].with_suffix(".csv"))
    summary = (tmp_path / "outputs" / "external_midi_summary.json").read_text()

    assert manifest_df["backend"].tolist() == ["midi_only", "midi_only"]
    assert manifest_df["split"].tolist() == ["benchmark", "benchmark"]
    assert manifest_df["benchmark_name"].tolist() == ["external_midi_benchmark", "external_midi_benchmark"]
    assert manifest_df["song_id"].tolist() == ["alpha", "alpha__dup01"]
    assert manifest_df["num_steps"].tolist() == [12, 12]
    assert '"skipped_count": 1' in summary


def test_load_episode_record_supports_midi_only_backend(tmp_path: Path) -> None:
    midi_path = tmp_path / "song.mid"
    midi_path.write_bytes(b"MThd")

    episode = load_episode_record(
        {
            "song_id": "song",
            "episode_id": "song__ep00000",
            "split": "test",
            "backend": "midi_only",
            "note_path": str(midi_path),
            "control_timestep": 0.05,
        }
    )

    assert episode.note_path == midi_path.resolve()
    assert episode.actions is None
    assert episode.hand_joints is None


def test_external_helpers_build_seeded_planner_inputs() -> None:
    token_df = pd.DataFrame(
        [
            {
                "split": "train",
                "primitive_id": "primitive_002",
                "primitive_index": 2,
                "primitive_family_index": 1,
                "duration_bucket": 3,
                "dynamics_bucket": 4,
                "motion_energy": 7.5,
                "start_state_norm": 2.0,
                "end_state_norm": 3.5,
            }
        ]
    )
    reference = compute_external_reference_stats(token_df)
    events = [
        ScoreEvent(
            event_id="episode_0",
            song_id="song",
            episode_id="song__ep00000",
            onset_step=0,
            end_step=8,
            start_time_sec=0.0,
            end_time_sec=0.4,
            key_numbers=(39, 51),
            chord_size=2,
            key_center=0.5,
            inter_onset_steps=0,
            source="midi",
        )
    ]
    rows = build_external_segment_rows(
        events=events,
        song_id="song",
        episode_id="song__ep00000",
        control_timestep=0.05,
        split="test",
        reference=reference,
        num_steps=16,
    )

    sample = build_external_planner_sample(
        history_rows=[],
        current_row=rows[0],
        metadata=SimpleNamespace(continuous_param_dim=4),
        reference=reference,
        context_length=8,
    )
    joints = extract_hand_joints(
        {
            "lh_shadow_hand/joints_pos": np.asarray([3.0, 4.0], dtype=np.float32),
            "rh_shadow_hand/joints_pos": np.asarray([1.0, 2.0], dtype=np.float32),
        }
    )

    assert isinstance(reference, ExternalReferenceStats)
    assert rows[0]["segment_source"] == "note_aligned_external"
    assert rows[0]["heuristic_family"] == "stacked"
    assert "goal_histogram" in rows[0]["score_context_json"]
    assert sample["primitive_history"].tolist() == [2]
    assert sample["family_history"].tolist() == [1]
    assert sample["target_params"].shape == (4,)
    assert joints.tolist() == [1.0, 2.0, 3.0, 4.0]


def test_resolve_external_manifest_base_accepts_csv_path(tmp_path: Path) -> None:
    manifest_csv = tmp_path / "external_midi_manifest.csv"
    manifest_csv.write_text("song_id,episode_id\n")

    assert resolve_external_manifest_base(benchmark_manifest=manifest_csv) == tmp_path / "external_midi_manifest"


def test_resolve_external_benchmark_name_prefers_manifest_metadata(tmp_path: Path) -> None:
    manifest_df = pd.DataFrame(
        [
            {
                "song_id": "song",
                "episode_id": "song__ep00000",
                "benchmark_name": "maestro",
            }
        ]
    )

    name = resolve_external_benchmark_name(
        manifest_df=manifest_df,
        manifest_base=tmp_path / "external_midi_manifest",
    )

    assert name == "maestro"


def test_normalize_external_benchmark_split_allows_unfiltered_eval() -> None:
    assert normalize_external_benchmark_split(None) is None
    assert normalize_external_benchmark_split("") is None
    assert normalize_external_benchmark_split("all") is None
    assert normalize_external_benchmark_split("benchmark") == "benchmark"


def test_robopianist_helper_uses_repo_root_package_parent(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "project_root"
    package_dir = repo_root / "robopianist"
    suite_dir = package_dir / "suite"
    suite_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "__init__.py").write_text("")
    (suite_dir / "__init__.py").write_text("")

    monkeypatch.setattr(robopianist_utils, "repo_root", lambda: repo_root)
    monkeypatch.setattr(robopianist_utils.importlib.util, "find_spec", lambda name: None)

    added = robopianist_utils.ensure_local_robopianist_on_path()

    assert added == repo_root
    assert str(repo_root.resolve()) in sys.path
