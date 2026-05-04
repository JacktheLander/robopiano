"""Evaluation utilities for Sonata-3."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["evaluate_external_midi_benchmark", "evaluate_primitives_online"]


def __getattr__(name: str) -> Any:
    if name == "evaluate_external_midi_benchmark":
        from sonata.evaluation.external_midi import evaluate_external_midi_benchmark

        return evaluate_external_midi_benchmark
    if name == "evaluate_primitives_online":
        from sonata.evaluation.primitive_online_eval import evaluate_primitives_online

        return evaluate_primitives_online
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:
    from sonata.evaluation.external_midi import evaluate_external_midi_benchmark
    from sonata.evaluation.primitive_online_eval import evaluate_primitives_online
