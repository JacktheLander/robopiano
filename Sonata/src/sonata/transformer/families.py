from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

CANONICAL_FAMILY_ORDER = (
    "single_note",
    "stacked_onset",
    "chordal",
    "sustain",
    "transition",
    "other",
)


@dataclass(frozen=True)
class PrimitiveFamilyMapping:
    family_names: list[str]
    primitive_to_family_name: dict[str, str]
    primitive_to_family_index: dict[str, int]
    primitive_to_reason: dict[str, str]
    duration_threshold: float
    motion_threshold: float
    sustain_motion_threshold: float

    def to_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for primitive_id, family_name in sorted(self.primitive_to_family_name.items()):
            records.append(
                {
                    "primitive_id": primitive_id,
                    "primitive_family": family_name,
                    "primitive_family_index": int(self.primitive_to_family_index[primitive_id]),
                    "mapping_reason": self.primitive_to_reason.get(primitive_id, ""),
                    "duration_threshold": float(self.duration_threshold),
                    "motion_threshold": float(self.motion_threshold),
                    "sustain_motion_threshold": float(self.sustain_motion_threshold),
                }
            )
        return records


def derive_primitive_family_mapping(
    token_df: pd.DataFrame,
    primitive_ids: list[str],
    *,
    library_df: pd.DataFrame | None = None,
    mode: str = "heuristic_stats",
) -> PrimitiveFamilyMapping:
    normalized_mode = str(mode or "heuristic_stats").strip().lower()
    if normalized_mode not in {"heuristic_stats", "heuristic_only"}:
        raise ValueError(
            f"Unsupported family_mapping_mode={mode!r}. Expected one of: heuristic_stats, heuristic_only."
        )
    if "primitive_id" not in token_df.columns:
        raise ValueError("Primitive token table must include a primitive_id column.")

    source = token_df[token_df["split"] == "train"].copy() if "split" in token_df.columns else token_df.copy()
    if source.empty:
        source = token_df.copy()
    if source.empty:
        family_names = ["other"]
        primitive_to_family_name = {primitive_id: "other" for primitive_id in primitive_ids}
        primitive_to_family_index = {primitive_id: 0 for primitive_id in primitive_ids}
        primitive_to_reason = {primitive_id: "empty_source" for primitive_id in primitive_ids}
        return PrimitiveFamilyMapping(
            family_names=family_names,
            primitive_to_family_name=primitive_to_family_name,
            primitive_to_family_index=primitive_to_family_index,
            primitive_to_reason=primitive_to_reason,
            duration_threshold=0.0,
            motion_threshold=0.0,
            sustain_motion_threshold=0.0,
        )

    grouped = source.groupby("primitive_id", sort=True)
    duration_means = grouped["duration_steps"].mean() if "duration_steps" in source.columns else pd.Series(dtype=float)
    motion_means = grouped["motion_energy"].mean() if "motion_energy" in source.columns else pd.Series(dtype=float)
    chord_means = grouped["chord_size"].mean() if "chord_size" in source.columns else pd.Series(dtype=float)
    dominant_heuristics = (
        grouped["heuristic_family"].agg(_dominant_string)
        if "heuristic_family" in source.columns
        else pd.Series(dtype=object)
    )

    if library_df is not None and not library_df.empty:
        library_frame = library_df.copy()
        library_frame["primitive_id"] = library_frame["primitive_id"].astype(str)
        library_frame = library_frame.set_index("primitive_id", drop=False)
        if duration_means.empty and "mean_duration_steps" in library_frame.columns:
            duration_means = library_frame["mean_duration_steps"].astype(float)
        if motion_means.empty and "mean_motion_energy" in library_frame.columns:
            motion_means = library_frame["mean_motion_energy"].astype(float)
        if chord_means.empty and "mean_chord_size" in library_frame.columns:
            chord_means = library_frame["mean_chord_size"].astype(float)
    else:
        library_frame = pd.DataFrame()

    duration_threshold = float(np.quantile(duration_means.to_numpy(dtype=np.float32), 0.70)) if not duration_means.empty else 0.0
    motion_threshold = float(np.quantile(motion_means.to_numpy(dtype=np.float32), 0.70)) if not motion_means.empty else 0.0
    sustain_motion_threshold = float(np.quantile(motion_means.to_numpy(dtype=np.float32), 0.40)) if not motion_means.empty else 0.0

    primitive_to_family_name: dict[str, str] = {}
    primitive_to_reason: dict[str, str] = {}
    for primitive_id in primitive_ids:
        dominant = str(dominant_heuristics.get(primitive_id, "other") or "other")
        mean_duration = float(duration_means.get(primitive_id, 0.0))
        mean_motion = float(motion_means.get(primitive_id, 0.0))
        mean_chord = float(chord_means.get(primitive_id, 0.0))

        if dominant == "chord" or mean_chord >= 3.0:
            family_name = "chordal"
            reason = f"dominant_heuristic={dominant};mean_chord_size={mean_chord:.3f}"
        elif dominant == "stacked":
            family_name = "stacked_onset"
            reason = f"dominant_heuristic={dominant}"
        elif dominant in {"window", "changepoint"}:
            family_name = "transition"
            reason = f"dominant_heuristic={dominant}"
        elif normalized_mode == "heuristic_only":
            family_name = "single_note" if dominant == "single" else "other"
            reason = f"heuristic_only={dominant}"
        elif mean_duration >= duration_threshold and mean_motion <= sustain_motion_threshold:
            family_name = "sustain"
            reason = (
                f"duration={mean_duration:.3f}>=q70({duration_threshold:.3f}) "
                f"and motion={mean_motion:.3f}<=q40({sustain_motion_threshold:.3f})"
            )
        elif mean_motion >= motion_threshold:
            family_name = "transition"
            reason = f"motion={mean_motion:.3f}>=q70({motion_threshold:.3f})"
        elif dominant == "single":
            family_name = "single_note"
            reason = f"dominant_heuristic={dominant}"
        else:
            family_name = "other"
            reason = f"fallback dominant_heuristic={dominant}"

        primitive_to_family_name[primitive_id] = family_name
        primitive_to_reason[primitive_id] = reason

    family_names = [name for name in CANONICAL_FAMILY_ORDER if name in set(primitive_to_family_name.values())]
    if not family_names:
        family_names = ["other"]
    family_to_index = {name: index for index, name in enumerate(family_names)}
    primitive_to_family_index = {
        primitive_id: int(family_to_index[family_name])
        for primitive_id, family_name in primitive_to_family_name.items()
    }
    return PrimitiveFamilyMapping(
        family_names=family_names,
        primitive_to_family_name=primitive_to_family_name,
        primitive_to_family_index=primitive_to_family_index,
        primitive_to_reason=primitive_to_reason,
        duration_threshold=duration_threshold,
        motion_threshold=motion_threshold,
        sustain_motion_threshold=sustain_motion_threshold,
    )


def _dominant_string(values: pd.Series) -> str:
    cleaned = values.astype(str).replace({"nan": "other"}).fillna("other")
    counts = cleaned.value_counts()
    if counts.empty:
        return "other"
    max_count = int(counts.iloc[0])
    winners = sorted(item for item, count in counts.items() if int(count) == max_count)
    return winners[0] if winners else "other"
