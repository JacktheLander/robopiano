"""Evaluation utilities for Sonata-3."""

__all__ = ["evaluate_external_midi_benchmark", "evaluate_primitives_online"]


def __getattr__(name: str):
    if name == "evaluate_external_midi_benchmark":
        from sonata.evaluation.external_midi import evaluate_external_midi_benchmark

        return evaluate_external_midi_benchmark
    if name == "evaluate_primitives_online":
        from sonata.evaluation.primitive_online_eval import evaluate_primitives_online

        return evaluate_primitives_online
    raise AttributeError(name)
