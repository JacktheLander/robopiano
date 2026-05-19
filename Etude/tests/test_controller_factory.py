from __future__ import annotations

import numpy as np

from etude.controllers import (
    FingertipPhaseResidualController,
    HierarchicalContactController,
    HybridPDResidualController,
    InverseDynamicsController,
    KeyAwareResidualController,
    ScheduledPDController,
    SafeHybridPDResidualController,
    TemporalResidualController,
    build_controller,
    list_supported_controller_families,
    normalize_controller_family,
    resolve_controller_class,
)
from etude.controllers.contact_policy import IdentityContactPolicy
from etude.controllers.pd import PDController
from etude.robopianist.state_mapping import StateMapping


def _mapping(action_dim: int = 4) -> StateMapping:
    return StateMapping(
        qpos_indices_46=list(range(46)),
        qvel_indices_46=list(range(46)),
        action_indices=list(range(action_dim)),
        fingertip_indices=None,
        key_state_indices=None,
        action_low=-np.ones(action_dim, dtype=np.float32),
        action_high=np.ones(action_dim, dtype=np.float32),
    )


def test_normalize_controller_family_accepts_legacy_type_names() -> None:
    assert normalize_controller_family("scheduled_pd") == "pd"
    assert normalize_controller_family("safe_hybrid_pd_residual") == "pd_residual"
    assert normalize_controller_family("key_aware_mlp") == "key_aware_residual"
    assert normalize_controller_family("fingertip_phase_residual") == "fingertip_residual"
    assert normalize_controller_family("hierarchical_contact") == "hierarchical"


def test_resolve_controller_class_covers_all_supported_variants() -> None:
    assert resolve_controller_class("pd") is ScheduledPDController
    assert resolve_controller_class("pd_residual") is HybridPDResidualController
    assert resolve_controller_class("safe_hybrid_pd_residual") is SafeHybridPDResidualController
    assert resolve_controller_class("key_aware_residual") is KeyAwareResidualController
    assert resolve_controller_class("fingertip_residual") is FingertipPhaseResidualController
    assert resolve_controller_class("temporal_residual") is TemporalResidualController
    assert resolve_controller_class("inverse_dynamics") is InverseDynamicsController
    assert resolve_controller_class("hierarchical") is HierarchicalContactController


def test_build_controller_instantiates_legacy_alias() -> None:
    controller = build_controller("scheduled_pd", _mapping(), kp=8.0, kd=0.4, lookahead_steps=0)
    assert isinstance(controller, ScheduledPDController)


def test_build_controller_instantiates_hierarchical_alias() -> None:
    controller = build_controller(
        "hierarchical_contact",
        IdentityContactPolicy(),
        PDController(_mapping()),
    )
    assert isinstance(controller, HierarchicalContactController)


def test_normalize_controller_family_lists_supported_values_on_error() -> None:
    try:
        normalize_controller_family("not_a_controller")
    except ValueError as exc:
        for family in list_supported_controller_families():
            assert family in str(exc)
    else:
        raise AssertionError("Expected normalize_controller_family to fail for an unknown family")
