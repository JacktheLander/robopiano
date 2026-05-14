from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# Aligned with robopianist/music/constants.py — keep local to avoid importing robopianist on import.
MIN_MIDI_PITCH_PIANO = 21
MAX_MIDI_PITCH_PIANO = 108
NUM_PIANO_KEYS = 88  # MAX_KEY_NUMBER - MIN_KEY_NUMBER + 1


def pitch_to_key_index(midi_pitch: int) -> int | None:
    """Map MIDI pitch to piano key index 0..87, or None if out of range."""
    if MIN_MIDI_PITCH_PIANO <= int(midi_pitch) <= MAX_MIDI_PITCH_PIANO:
        return int(midi_pitch) - MIN_MIDI_PITCH_PIANO
    return None


def piece_id_from_path(midi_path: Path, dataset_root: Path) -> str:
    return midi_path.resolve().relative_to(dataset_root.resolve()).as_posix()


def discover_midi_files(dataset_root: Path) -> list[Path]:
    resolved_root = dataset_root.expanduser().resolve()
    if not resolved_root.exists():
        raise FileNotFoundError(f"MAESTRO dataset root does not exist: {resolved_root}")
    midi_files = [path for path in resolved_root.rglob("*") if path.is_file() and path.suffix.lower() in {".mid", ".midi"}]
    midi_files.sort(key=lambda path: piece_id_from_path(path, resolved_root))
    return midi_files


def _overlap_frame_active(note_start: float, note_end: float, t: int, dt: float) -> bool:
    """True if note interval overlaps control frame [t*dt, (t+1)*dt)."""
    frame_start = float(t) * dt
    frame_end = float(t + 1) * dt
    return note_end > frame_start and note_start < frame_end


def quantize_notes_to_target_keys(
    notes: Iterable[tuple[float, float, int]],
    *,
    control_timestep: float,
    max_steps: int | None = None,
    max_duration_s: float | None = None,
    duration_hint_s: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Build a [T, 88] float32 roll from (start_s, end_s, midi_pitch) note events.

    A key is 1.0 on frame t if any in-range note overlaps [t*dt, (t+1)*dt).
    """
    dt = float(control_timestep)
    if dt <= 0:
        raise ValueError(f"control_timestep must be positive, got {dt}")

    note_list: list[tuple[float, float, int]] = []
    max_end = float(duration_hint_s or 0.0)
    for start, end, pitch in notes:
        s, e = float(start), float(end)
        if e <= s:
            continue
        idx = pitch_to_key_index(int(pitch))
        if idx is None:
            continue
        note_list.append((s, e, idx))
        max_end = max(max_end, e)

    if max_duration_s is not None:
        max_end = min(max_end, float(max_duration_s))

    if max_steps is not None:
        t_count = int(max_steps)
    else:
        t_count = int(np.ceil(max_end / dt)) if max_end > 0 else 1

    roll = np.zeros((t_count, NUM_PIANO_KEYS), dtype=np.float32)
    for t in range(t_count):
        for note_start, note_end, key_idx in note_list:
            if _overlap_frame_active(note_start, note_end, t, dt):
                roll[t, key_idx] = 1.0

    meta: dict[str, Any] = {
        "control_timestep": dt,
        "num_steps": int(t_count),
        "num_notes_used": len(note_list),
        "duration_hint_s": float(max_end),
    }
    return roll, meta


def _pretty_midi_notes(midi_path: Path) -> tuple[list[tuple[float, float, int]], float]:
    try:
        import pretty_midi
    except ImportError as exc:  # pragma: no cover - optional at import time
        raise ImportError(
            "Loading MAESTRO .mid files requires `pretty_midi` (bundled with robopianist). "
            "Install robopianist dependencies or `pip install pretty_midi`."
        ) from exc

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    out: list[tuple[float, float, int]] = []
    for inst in pm.instruments:
        if getattr(inst, "is_drum", False):
            continue
        for n in inst.notes:
            out.append((float(n.start), float(n.end), int(n.pitch)))
    return out, float(pm.get_end_time())


def midi_to_target_key_roll(
    midi_path: str | Path,
    *,
    control_timestep: float = 0.05,
    max_steps: int | None = None,
    max_duration_s: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Load a MIDI file and return quantized `target_keys` [T, 88] for Variations inference.
    """
    path = Path(midi_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"MIDI file not found: {path}")

    notes, end_time = _pretty_midi_notes(path)
    roll, meta = quantize_notes_to_target_keys(
        notes,
        control_timestep=control_timestep,
        max_steps=max_steps,
        max_duration_s=max_duration_s,
        duration_hint_s=end_time,
    )
    meta["midi_path"] = str(path)
    return roll, meta


def simulation_slug(text: str, *, max_len: int = 80) -> str:
    """Filesystem-safe slug for run directory names."""
    s = text.strip().replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    s = s.strip("_") or "piece"
    return s[:max_len]
