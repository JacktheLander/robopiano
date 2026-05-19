from __future__ import annotations

import json
from typing import Any

from etude.evaluation.failure_attribution import attribute_failures


def build_failure_report(**kwargs: Any) -> dict[str, Any]:
    attribution = attribute_failures(**kwargs)
    primary = attribution["primary_failure_mode"]
    return {
        "primary_failure_mode": primary,
        "secondary_failure_modes": attribution["secondary_failure_modes"],
        "evidence": attribution["evidence"].get(primary, []),
        "recommended_next_component_to_fix": attribution["recommended_next_component_to_fix"],
    }


def format_failure_report(**kwargs: Any) -> str:
    return json.dumps(build_failure_report(**kwargs), indent=2, sort_keys=True)
