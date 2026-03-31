from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sonata.evaluation.rollout import _canonicalize_environment_name, _manifest_note_path, _resolve_rollout_source


class _Logger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str, *args: object) -> None:
        self.messages.append(message % args if args else message)


def test_canonicalize_environment_name_strips_rp1m_episode_suffix() -> None:
    assert (
        _canonicalize_environment_name("RoboPianist-repertoire-150-FrenchSuiteNo5Sarabande-v0_0")
        == "RoboPianist-repertoire-150-FrenchSuiteNo5Sarabande-v0"
    )
    assert _canonicalize_environment_name("RoboPianist-debug-TwinkleTwinkleRousseau-v0") == "RoboPianist-debug-TwinkleTwinkleRousseau-v0"


def test_manifest_note_path_requires_existing_file(tmp_path: Path) -> None:
    midi_path = tmp_path / "song.mid"
    midi_path.write_bytes(b"MThd")

    assert _manifest_note_path({"note_path": str(midi_path)}) == midi_path.resolve()
    assert _manifest_note_path({"note_path": str(tmp_path / "missing.mid")}) is None
    assert _manifest_note_path({"note_path": ""}) is None


def test_resolve_rollout_source_prefers_manifest_note_path(tmp_path: Path) -> None:
    midi_path = tmp_path / "song.mid"
    midi_path.write_bytes(b"MThd")
    logger = _Logger()
    song_id = "RoboPianist-repertoire-150-FrenchSuiteNo5Sarabande-v0_0"
    episode_id = f"{song_id}__ep00000"

    source = _resolve_rollout_source(
        song_id=song_id,
        episode_id=episode_id,
        manifest_lookup={(song_id, episode_id): {"note_path": str(midi_path)}},
        logger=logger,
    )

    assert source["environment_name"] == "RoboPianist-repertoire-150-FrenchSuiteNo5Sarabande-v0"
    assert source["midi_file"] == midi_path.resolve()
    assert source["song_id_normalized"] is True
    assert logger.messages == []
