from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from etude.utils.import_utils import load_symbol


SUPPORTED_CONTROLLER_FAMILIES: tuple[str, ...] = (
    "pd",
    "pd_residual",
    "key_aware_residual",
    "fingertip_residual",
    "temporal_residual",
    "inverse_dynamics",
    "hierarchical",
)

_FAMILY_ALIASES: dict[str, str] = {
    "pd": "pd",
    "pd_scheduled": "pd",
    "scheduled_pd": "pd",
    "pd_grouped": "pd",
    "pd_phase_scheduled": "pd",
    "pd_residual": "pd_residual",
    "hybrid": "pd_residual",
    "hybrid_pd_residual": "pd_residual",
    "safe_hybrid_pd_residual": "pd_residual",
    "residual_safe_mlp": "pd_residual",
    "residual_safe_gru": "pd_residual",
    "key_aware_residual": "key_aware_residual",
    "key_aware_mlp": "key_aware_residual",
    "fingertip_residual": "fingertip_residual",
    "fingertip_phase_residual": "fingertip_residual",
    "temporal_residual": "temporal_residual",
    "inverse_dynamics": "inverse_dynamics",
    "hierarchical": "hierarchical",
    "hierarchical_contact": "hierarchical",
}

_CONTROLLER_CLASS_PATHS: dict[str, str] = {
    "pd": "etude.controllers.pd_scheduled:ScheduledPDController",
    "pd_scheduled": "etude.controllers.pd_scheduled:ScheduledPDController",
    "scheduled_pd": "etude.controllers.pd_scheduled:ScheduledPDController",
    "pd_grouped": "etude.controllers.pd_scheduled:ScheduledPDController",
    "pd_phase_scheduled": "etude.controllers.pd_scheduled:ScheduledPDController",
    "pd_residual": "etude.controllers.hybrid:HybridPDResidualController",
    "hybrid": "etude.controllers.hybrid:HybridPDResidualController",
    "hybrid_pd_residual": "etude.controllers.hybrid:HybridPDResidualController",
    "safe_hybrid_pd_residual": "etude.controllers.hybrid_safe:SafeHybridPDResidualController",
    "residual_safe_mlp": "etude.controllers.hybrid_safe:SafeHybridPDResidualController",
    "residual_safe_gru": "etude.controllers.hybrid_safe:SafeHybridPDResidualController",
    "key_aware_residual": "etude.controllers.key_aware_residual:KeyAwareResidualController",
    "key_aware_mlp": "etude.controllers.key_aware_residual:KeyAwareResidualController",
    "fingertip_residual": "etude.controllers.fingertip_phase_residual:FingertipPhaseResidualController",
    "fingertip_phase_residual": "etude.controllers.fingertip_phase_residual:FingertipPhaseResidualController",
    "temporal_residual": "etude.controllers.temporal_residual:TemporalResidualController",
    "inverse_dynamics": "etude.controllers.inverse_dynamics:InverseDynamicsController",
    "hierarchical": "etude.controllers.hierarchical:HierarchicalContactController",
    "hierarchical_contact": "etude.controllers.hierarchical:HierarchicalContactController",
}


def normalize_controller_family(family_or_type: str) -> str:
    key = _normalize_key(family_or_type)
    normalized = _FAMILY_ALIASES.get(key)
    if normalized is None:
        supported = ", ".join(SUPPORTED_CONTROLLER_FAMILIES)
        raise ValueError(
            f"Unsupported controller family '{family_or_type}'. Supported families: {supported}"
        )
    return normalized


def resolve_controller_class(family_or_type: str) -> type[Any]:
    key = _normalize_key(family_or_type)
    path = _CONTROLLER_CLASS_PATHS.get(key)
    if path is None:
        normalize_controller_family(family_or_type)
        raise ValueError(f"No controller class registered for '{family_or_type}'")
    resolved = load_symbol(path)
    if not isinstance(resolved, type):
        raise TypeError(f"Resolved controller for '{family_or_type}' is not a class")
    return resolved


def build_controller(
    family_or_type: str,
    /,
    *args: Any,
    **kwargs: Any,
) -> Any:
    controller_cls = resolve_controller_class(family_or_type)
    return controller_cls(*args, **kwargs)


def list_supported_controller_families() -> list[str]:
    return list(SUPPORTED_CONTROLLER_FAMILIES)


def supported_controller_types() -> list[str]:
    return sorted(_unique_keys(_CONTROLLER_CLASS_PATHS))


def _normalize_key(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _unique_keys(values: dict[str, Any]) -> Iterable[str]:
    seen: set[str] = set()
    for key in values:
        if key in seen:
            continue
        seen.add(key)
        yield key
