from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sonata.primitives.segmenters import CandidateSegment, aggregate_budget_metrics, budget_segment_candidates  # noqa: E402


def _candidate(
    *,
    onset: int = 20,
    signature: str = "10",
    family: str = "single_press",
    chord_size: int = 1,
    contact: bool = True,
    activation: bool = True,
    inactive: bool = True,
    boundary: float = 0.0,
    start: int = 10,
) -> CandidateSegment:
    return CandidateSegment(
        onset_step=start,
        end_step=start + 14,
        segment_source="prepress_causal",
        score_event_id=f"seg_{onset}_{signature}_{start}",
        key_signature=signature,
        heuristic_family="single" if chord_size <= 1 else "chord",
        chord_size=chord_size,
        key_center=0.0,
        coarse_family=family,
        boundary_alignment_score=boundary,
        target_key_count=chord_size,
        target_key_signature=signature,
        target_onset_step=onset,
        causal_segment=True,
        inactive_start=inactive,
        activation_after_start=activation,
        contact_near_onset=contact,
    )


def _budget(**overrides):
    budget = {
        "enabled": True,
        "max_segments_per_episode": 40,
        "max_segments_per_score_onset": 1,
        "max_segments_per_target_signature": 1,
        "preserve_family_balance": True,
        "ranking_metric": "causal_press_score",
        "seed": 7,
    }
    budget.update(overrides)
    return {"prepress_steps": 10, "post_onset_steps": 4, "segment_min_len": 8, "segment_max_len": 20, "segment_budget": budget}


def test_segment_budget_enforces_onset_and_signature_caps() -> None:
    candidates = [_candidate(start=start) for start in range(10, 16)]

    accepted, stats = budget_segment_candidates(candidates, _budget())

    assert len(accepted) == 1
    assert stats["accepted_segments_before_budget"] == 6
    assert stats["accepted_segments_after_budget"] == 1
    assert stats["dropped_by_budget"] == 5


def test_candidate_ranking_prefers_causal_press_quality() -> None:
    weak = _candidate(contact=False, activation=False, inactive=False, boundary=0.0, start=10)
    strong = _candidate(contact=True, activation=True, inactive=True, boundary=1.0, start=11)

    accepted, _ = budget_segment_candidates([weak, strong], _budget())

    assert accepted == [strong]
    assert strong.causal_press_score > weak.causal_press_score


def test_family_balance_keeps_chord_families_when_present() -> None:
    candidates = [
        _candidate(signature=f"{10 + index}", family="single_press", chord_size=1, boundary=10.0, start=index)
        for index in range(8)
    ]
    candidates.extend(
        [
            _candidate(signature="20-24", family="dyad_press", chord_size=2, boundary=0.0, start=20),
            _candidate(signature="30-34-37", family="triad_press", chord_size=3, boundary=0.0, start=30),
        ]
    )

    accepted, _ = budget_segment_candidates(
        candidates,
        _budget(max_segments_per_score_onset=None, max_segments_per_target_signature=None, max_segments_per_episode=3),
    )

    families = {candidate.coarse_family for candidate in accepted}
    assert {"single_press", "dyad_press", "triad_press"}.issubset(families)


def test_budget_manifest_metrics_are_computed() -> None:
    frame = pd.DataFrame(
        [
            {"episode_id": "e0", "target_onset_step": 10, "duration_steps": 14, "chord_size": 1, "coarse_family": "single_press"},
            {"episode_id": "e0", "target_onset_step": 20, "duration_steps": 14, "chord_size": 2, "coarse_family": "dyad_press"},
        ]
    )

    metrics = aggregate_budget_metrics(
        frame,
        _budget(),
        {"proposed_segments": 5, "accepted_segments_before_budget": 4, "dropped_by_budget": 2},
    )

    assert metrics["proposed_segments"] == 5
    assert metrics["accepted_segments_after_budget"] == 2
    assert metrics["dropped_by_budget"] == 2
    assert metrics["budget_enabled"] is True
    assert metrics["chord_size_counts"] == {"1": 1, "2": 1}
