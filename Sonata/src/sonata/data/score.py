from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from sonata.data.schema import ScoreEvent

_PIANO_LOWEST_MIDI = 21
_NUM_PIANO_KEYS = 88


def load_note_events(
    note_path: str | Path,
    control_timestep: float,
    chord_tolerance_steps: int = 1,
    song_id: str | None = None,
    episode_id: str | None = None,
) -> list[ScoreEvent]:
    path = Path(note_path).resolve()
    suffix = path.suffix.lower()
    if suffix == ".proto":
        notes = _load_proto_notes(path)
        source = "proto"
    elif suffix in {".mid", ".midi"}:
        notes = _load_midi_notes(path)
        source = "midi"
    else:
        raise ValueError(f"Unsupported score file type: {path.suffix}")
    return score_events_from_notes(
        notes=notes,
        control_timestep=control_timestep,
        chord_tolerance_steps=chord_tolerance_steps,
        song_id=song_id or path.stem,
        episode_id=episode_id or path.stem,
        source=source,
    )


def score_events_from_notes(
    *,
    notes: list[dict[str, float | int]],
    control_timestep: float,
    chord_tolerance_steps: int,
    song_id: str,
    episode_id: str,
    source: str,
) -> list[ScoreEvent]:
    quantized: list[dict[str, int]] = []
    for note in notes:
        pitch = int(note["pitch"])
        key_number = pitch - _PIANO_LOWEST_MIDI
        if key_number < 0 or key_number >= _NUM_PIANO_KEYS:
            continue
        onset_step = max(int(round(float(note["start_time"]) / control_timestep)), 0)
        end_step = max(int(round(float(note["end_time"]) / control_timestep)), onset_step + 1)
        quantized.append({"key_number": key_number, "onset_step": onset_step, "end_step": end_step})
    return _build_events_from_quantized_notes(
        quantized=quantized,
        control_timestep=control_timestep,
        chord_tolerance_steps=chord_tolerance_steps,
        song_id=song_id,
        episode_id=episode_id,
        source=source,
    )


def infer_events_from_goal_roll(
    roll: np.ndarray | None,
    *,
    song_id: str,
    episode_id: str,
    control_timestep: float,
    chord_tolerance_steps: int = 1,
    source: str = "goals",
) -> list[ScoreEvent]:
    piano_roll = _piano_roll_from_roll(roll)
    if piano_roll.size == 0:
        return []

    quantized: list[dict[str, int]] = []
    prev_active = np.zeros((piano_roll.shape[1],), dtype=bool)
    for step, frame in enumerate(piano_roll):
        active = frame > 0.5
        onset_keys = np.flatnonzero(active & ~prev_active)
        for key_number in onset_keys.tolist():
            release = _find_release_step(piano_roll[:, key_number], onset_step=step)
            quantized.append({"key_number": int(key_number), "onset_step": int(step), "end_step": int(release)})
        prev_active = active

    return _build_events_from_quantized_notes(
        quantized=quantized,
        control_timestep=control_timestep,
        chord_tolerance_steps=chord_tolerance_steps,
        song_id=song_id,
        episode_id=episode_id,
        source=source,
    )


def score_context_from_roll(roll: np.ndarray | None, onset_step: int, future_window_steps: int = 8) -> dict[str, Any]:
    piano_roll = _piano_roll_from_roll(roll)
    if piano_roll.size == 0:
        return {
            "goal_histogram": [0.0] * 12,
            "active_ratio": 0.0,
            "future_density": 0.0,
        }

    index = int(np.clip(onset_step, 0, max(piano_roll.shape[0] - 1, 0)))
    active = piano_roll[index] > 0.5
    histogram = np.zeros((12,), dtype=np.float32)
    active_keys = np.flatnonzero(active)
    if active_keys.size:
        pitch_classes = active_keys % 12
        histogram += np.bincount(pitch_classes, minlength=12).astype(np.float32)
        histogram /= max(float(histogram.sum()), 1.0)

    future_end = min(index + int(future_window_steps), piano_roll.shape[0])
    future_window = piano_roll[index:future_end]
    future_density = float((future_window > 0.5).mean()) if future_window.size else 0.0
    return {
        "goal_histogram": histogram.tolist(),
        "active_ratio": float(active.mean()),
        "future_density": future_density,
    }


def dumps_score_context(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True)


def _load_proto_notes(path: Path) -> list[dict[str, float | int]]:
    try:
        from note_seq.protobuf import music_pb2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "note-seq is required to parse Sonata .proto note files."
        ) from exc

    sequence = music_pb2.NoteSequence()
    sequence.ParseFromString(path.read_bytes())
    return [
        {
            "pitch": int(note.pitch),
            "start_time": float(note.start_time),
            "end_time": float(note.end_time),
            "velocity": int(note.velocity),
        }
        for note in sequence.notes
        if float(note.end_time) > float(note.start_time)
    ]


def _load_midi_notes(path: Path) -> list[dict[str, float | int]]:
    try:
        import note_seq
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "note-seq is required to parse Sonata MIDI note files."
        ) from exc

    sequence = note_seq.midi_file_to_note_sequence(str(path))
    return [
        {
            "pitch": int(note.pitch),
            "start_time": float(note.start_time),
            "end_time": float(note.end_time),
            "velocity": int(note.velocity),
        }
        for note in sequence.notes
        if float(note.end_time) > float(note.start_time)
    ]


def _build_events_from_quantized_notes(
    *,
    quantized: list[dict[str, int]],
    control_timestep: float,
    chord_tolerance_steps: int,
    song_id: str,
    episode_id: str,
    source: str,
) -> list[ScoreEvent]:
    if not quantized:
        return []

    notes = sorted(quantized, key=lambda item: (item["onset_step"], item["key_number"]))
    groups: list[dict[str, Any]] = []
    for note in notes:
        if not groups or note["onset_step"] - groups[-1]["last_onset_step"] > chord_tolerance_steps:
            groups.append(
                {
                    "onset_step": note["onset_step"],
                    "last_onset_step": note["onset_step"],
                    "notes": [note],
                }
            )
        else:
            groups[-1]["notes"].append(note)
            groups[-1]["last_onset_step"] = note["onset_step"]

    events: list[ScoreEvent] = []
    previous_onset = None
    for index, group in enumerate(groups):
        key_numbers = tuple(sorted({int(item["key_number"]) for item in group["notes"]}))
        onset_step = int(group["onset_step"])
        end_step = max(int(item["end_step"]) for item in group["notes"])
        inter_onset_steps = 0 if previous_onset is None else onset_step - previous_onset
        previous_onset = onset_step
        key_center = float(np.mean(key_numbers) / max(_NUM_PIANO_KEYS - 1, 1)) if key_numbers else 0.0
        events.append(
            ScoreEvent(
                event_id=f"{episode_id}_{source}_{index:06d}",
                song_id=song_id,
                episode_id=episode_id,
                onset_step=onset_step,
                end_step=end_step,
                start_time_sec=float(onset_step * control_timestep),
                end_time_sec=float(end_step * control_timestep),
                key_numbers=key_numbers,
                chord_size=len(key_numbers),
                key_center=key_center,
                inter_onset_steps=int(inter_onset_steps),
                source=source,
            )
        )
    return events


def _piano_roll_from_roll(roll: np.ndarray | None) -> np.ndarray:
    if roll is None:
        return np.zeros((0, _NUM_PIANO_KEYS), dtype=np.float32)
    array = np.asarray(roll, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] == 0:
        return np.zeros((0, _NUM_PIANO_KEYS), dtype=np.float32)
    if array.shape[1] >= _NUM_PIANO_KEYS + 1:
        return array[:, :_NUM_PIANO_KEYS]
    return array[:, : min(array.shape[1], _NUM_PIANO_KEYS)]


def _find_release_step(activity: np.ndarray, onset_step: int) -> int:
    active = np.asarray(activity > 0.5, dtype=bool)
    release_step = onset_step + 1
    while release_step < active.shape[0] and active[release_step]:
        release_step += 1
    return release_step
