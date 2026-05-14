"""Variations MAESTRO → RoboPianist simulation helpers (package `simulate` on `Variations/`)."""

from .midi_keysets import (
    discover_midi_files,
    midi_to_target_key_roll,
    piece_id_from_path,
    pitch_to_key_index,
    quantize_notes_to_target_keys,
    simulation_slug,
)

__all__ = [
    "discover_midi_files",
    "midi_to_target_key_roll",
    "piece_id_from_path",
    "pitch_to_key_index",
    "quantize_notes_to_target_keys",
    "simulation_slug",
]
