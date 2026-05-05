from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Sequence

import numpy as np

NUM_PIANO_KEYS = 88
RESTORE_MODES = {"unsafe_legacy", "hands_only", "neutral"}


@dataclass(slots=True)
class CausalRolloutConfig:
    enabled: bool = True
    restore_mode: str = "hands_only"
    allow_piano_state_restore: bool = False
    allow_goal_state_restore: bool = False
    allow_reference_midi_scoring: bool = False
    allow_target_state_as_observation: bool = False
    require_neutral_piano_start: bool = True
    fail_if_initial_keys_active: bool = True
    initial_key_activation_threshold: float = 0.1
    run_zero_action_ablation: bool = True
    require_contact_for_keypress: bool = True
    contact_tolerance_frames: int = 2
    key_activation_threshold: float = 0.1
    fingertip_key_distance_threshold_m: float = 0.025
    video_audio_source: str = "none"
    reset_initial_piano_if_active: bool = False

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "CausalRolloutConfig":
        values = dict(payload or {})
        if "restore_mode" in values and values["restore_mode"] not in RESTORE_MODES:
            raise ValueError(f"Unsupported causal restore_mode={values['restore_mode']!r}.")
        config = cls(**{key: values[key] for key in cls.__dataclass_fields__ if key in values})
        if not config.enabled:
            if config.restore_mode != "unsafe_legacy":
                raise ValueError("causal_eval.enabled=false is only allowed with restore_mode=unsafe_legacy.")
            config.allow_piano_state_restore = True
            config.allow_goal_state_restore = True
            config.allow_reference_midi_scoring = True
            config.allow_target_state_as_observation = True
        if config.restore_mode == "unsafe_legacy":
            config.enabled = False
        return config

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class NeutralPianoCheck:
    passed: bool
    initial_active_key_indices: list[int]
    max_activation: float
    failure_reason: str | None = None


@dataclass(slots=True)
class ContactCollection:
    contact_roll: np.ndarray | None
    contact_method: str
    contact_key_indices: list[int]
    notes: list[str]


def default_causal_eval_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    config = CausalRolloutConfig.from_mapping(overrides or {})
    return config.to_dict()


def mark_uncausal_result(
    result: dict[str, Any],
    *,
    reason: str,
    status: str | None = None,
) -> dict[str, Any]:
    output = dict(result)
    output["causal_validated"] = False
    output["causal_failure_reason"] = str(reason)
    if status is not None:
        output["status"] = str(status)
    output["success"] = False
    return output


def forbid_dataset_piano_state_restore(
    *,
    restore_mode: str,
    causal_config: CausalRolloutConfig,
    source: str,
) -> None:
    if restore_mode == "unsafe_legacy" or not causal_config.enabled:
        return
    if source == "piano_state" and not causal_config.allow_piano_state_restore:
        raise RuntimeError("Causal rollout forbids restoring dataset piano state.")
    if source == "goal_state" and not causal_config.allow_goal_state_restore:
        raise RuntimeError("Causal rollout forbids restoring goal or MIDI target state.")


def assert_neutral_piano_start(
    piano_state: np.ndarray | None = None,
    *,
    env: Any | None = None,
    threshold: float = 0.1,
) -> NeutralPianoCheck:
    state = collect_physical_key_activation(env=env) if piano_state is None else piano_state
    if state is None:
        return NeutralPianoCheck(
            passed=False,
            initial_active_key_indices=[],
            max_activation=float("nan"),
            failure_reason="piano_activation_unavailable",
        )
    array = _key_state_array(state)
    active = np.flatnonzero(array >= float(threshold)).astype(int).tolist()
    max_activation = float(np.max(array)) if array.size else 0.0
    return NeutralPianoCheck(
        passed=not active,
        initial_active_key_indices=active,
        max_activation=max_activation,
        failure_reason=None if not active else "initial_keys_active",
    )


def reset_or_validate_neutral_piano(
    env: Any,
    *,
    causal_config: CausalRolloutConfig,
) -> NeutralPianoCheck:
    check = assert_neutral_piano_start(
        env=env,
        threshold=float(causal_config.initial_key_activation_threshold),
    )
    if check.passed or not causal_config.reset_initial_piano_if_active:
        return check
    reset_physical_piano_state(env)
    return assert_neutral_piano_start(
        env=env,
        threshold=float(causal_config.initial_key_activation_threshold),
    )


def reset_physical_piano_state(env: Any) -> bool:
    task = getattr(env, "task", None)
    physics = getattr(env, "physics", None)
    piano = getattr(task, "piano", None)
    if piano is None or physics is None:
        return False
    changed = False
    joints = getattr(piano, "joints", None)
    if joints is not None:
        try:
            binding = physics.bind(joints)
            binding.qpos = np.zeros_like(np.asarray(binding.qpos, dtype=np.float32))
            if hasattr(binding, "qvel"):
                binding.qvel = np.zeros_like(np.asarray(binding.qvel, dtype=np.float32))
            changed = True
        except Exception:
            pass
    for attr_name in ("_sustain_state", "sustain_state"):
        value = getattr(piano, attr_name, None)
        if value is None:
            continue
        try:
            np.asarray(value)[...] = 0.0
            changed = True
        except Exception:
            pass
    if hasattr(physics, "forward"):
        try:
            physics.forward()
        except Exception:
            pass
    update = getattr(piano, "_update_key_state", None)
    if callable(update):
        try:
            update(physics)
        except Exception:
            pass
    return changed


def collect_physical_key_activation(env: Any | None = None, piano_state: np.ndarray | None = None) -> np.ndarray | None:
    if piano_state is not None:
        return _key_state_array(piano_state)
    if env is None:
        return None
    for current in _iter_wrapped_envs(env):
        task = getattr(current, "task", None)
        piano = getattr(task, "piano", None)
        if piano is None:
            continue
        for attr_name in ("activation", "normalized_state", "state"):
            value = getattr(piano, attr_name, None)
            if value is not None:
                try:
                    return _key_state_array(value)
                except Exception:
                    continue
    return None


def collect_fingertip_key_contacts(
    env: Any | None,
    *,
    fingertip_positions: np.ndarray | None = None,
    key_positions: np.ndarray | None = None,
    key_count: int = NUM_PIANO_KEYS,
    distance_threshold_m: float = 0.025,
) -> ContactCollection:
    physics_roll = _collect_physics_contact_roll(env=env, key_count=key_count)
    if physics_roll is not None:
        indices = _active_indices(physics_roll)
        return ContactCollection(
            contact_roll=physics_roll.astype(np.float32),
            contact_method="physics_contact",
            contact_key_indices=indices,
            notes=[],
        )

    tips = _fingertip_positions_from_env(env) if fingertip_positions is None else fingertip_positions
    keys = _key_positions_from_env(env, key_count=key_count) if key_positions is None else key_positions
    if tips is None or keys is None:
        return ContactCollection(
            contact_roll=None,
            contact_method="unavailable",
            contact_key_indices=[],
            notes=["No MuJoCo contact pairs or fingertip/key positions were available."],
        )
    tips_array = np.asarray(tips, dtype=np.float32).reshape(-1, 3)
    key_array = np.asarray(keys, dtype=np.float32).reshape(-1, 3)
    if tips_array.size == 0 or key_array.shape[0] == 0:
        return ContactCollection(contact_roll=None, contact_method="unavailable", contact_key_indices=[], notes=[])
    distances = np.linalg.norm(tips_array[:, None, :] - key_array[None, :, :], axis=2)
    nearest_dist = distances.min(axis=0)
    roll = (nearest_dist < float(distance_threshold_m)).astype(np.float32)[None, :key_count]
    return ContactCollection(
        contact_roll=roll,
        contact_method="distance_proxy",
        contact_key_indices=np.flatnonzero(roll[0] > 0.5).astype(int).tolist(),
        notes=[],
    )


def contact_gated_keypress_metrics(
    *,
    target_key_indices: Sequence[int],
    activation_roll: np.ndarray | None,
    contact_roll: np.ndarray | None,
    config: CausalRolloutConfig,
    contact_method: str = "unavailable",
) -> dict[str, Any]:
    target_keys = sorted({int(key) for key in target_key_indices if 0 <= int(key) < NUM_PIANO_KEYS})
    if activation_roll is None:
        return _empty_causal_metrics(
            target_keys=target_keys,
            status="activation_unavailable",
            causal_validated=False,
            failure_reason="physical_activation_unavailable",
            contact_method=contact_method,
        )
    activation = _binary_roll(activation_roll, threshold=float(config.key_activation_threshold))
    if activation.shape[0] == 0:
        return _empty_causal_metrics(
            target_keys=target_keys,
            status="activation_unavailable",
            causal_validated=False,
            failure_reason="physical_activation_unavailable",
            contact_method=contact_method,
        )
    initial_active = np.flatnonzero(activation[0]).astype(int).tolist()
    activation_events = _activation_events_from_roll(activation)
    activation_key_indices = sorted({int(key) for key, _ in activation_events})

    contact_available = contact_roll is not None and str(contact_method) != "unavailable"
    if config.require_contact_for_keypress and not contact_available:
        return _causal_metrics_from_sets(
            target_keys=target_keys,
            activation_key_indices=activation_key_indices,
            contact_key_indices=[],
            gated_true_positive_keys=[],
            activation_without_contact_key_indices=activation_key_indices,
            contact_without_activation_key_indices=[],
            initial_active_key_indices=initial_active,
            contact_method=contact_method,
            causal_validated=False,
            status="contact_unavailable",
            failure_reason="contact_unavailable",
        )

    contact = _binary_roll(contact_roll, threshold=0.5) if contact_roll is not None else np.zeros_like(activation)
    contact = _align_roll(contact, activation.shape[0])
    contact_key_indices = sorted(np.flatnonzero(np.any(contact, axis=0)).astype(int).tolist())
    gated_true_positive_keys: list[int] = []
    activation_without_contact: list[int] = []
    tolerance = max(int(config.contact_tolerance_frames), 0)
    for key, activation_frame in activation_events:
        if key in initial_active:
            continue
        contact_start = max(int(activation_frame) - tolerance, 0)
        contact_end = min(int(activation_frame) + 1, contact.shape[0])
        has_pre_activation_contact = bool(np.any(contact[contact_start:contact_end, int(key)]))
        if has_pre_activation_contact:
            if int(key) in target_keys:
                gated_true_positive_keys.append(int(key))
        else:
            activation_without_contact.append(int(key))
    contact_without_activation = sorted(set(contact_key_indices) - set(activation_key_indices))
    metrics = _causal_metrics_from_sets(
        target_keys=target_keys,
        activation_key_indices=activation_key_indices,
        contact_key_indices=contact_key_indices,
        gated_true_positive_keys=sorted(set(gated_true_positive_keys)),
        activation_without_contact_key_indices=sorted(set(activation_without_contact)),
        contact_without_activation_key_indices=contact_without_activation,
        initial_active_key_indices=initial_active,
        contact_method=contact_method,
        causal_validated=True,
        status="ok",
        failure_reason=None,
    )
    if config.require_contact_for_keypress and metrics["causal_false_positive_events"] > 0:
        metrics["contact_gated_success"] = False
    return metrics


def run_zero_action_ablation(
    *,
    action_shape: tuple[int, int],
    rollout_fn: Callable[[np.ndarray], dict[str, Any]],
) -> dict[str, Any]:
    zeros = np.zeros(action_shape, dtype=np.float32)
    return rollout_fn(zeros)


def _empty_causal_metrics(
    *,
    target_keys: list[int],
    status: str,
    causal_validated: bool,
    failure_reason: str,
    contact_method: str,
) -> dict[str, Any]:
    return _causal_metrics_from_sets(
        target_keys=target_keys,
        activation_key_indices=[],
        contact_key_indices=[],
        gated_true_positive_keys=[],
        activation_without_contact_key_indices=[],
        contact_without_activation_key_indices=[],
        initial_active_key_indices=[],
        contact_method=contact_method,
        causal_validated=causal_validated,
        status=status,
        failure_reason=failure_reason,
    )


def _causal_metrics_from_sets(
    *,
    target_keys: list[int],
    activation_key_indices: list[int],
    contact_key_indices: list[int],
    gated_true_positive_keys: list[int],
    activation_without_contact_key_indices: list[int],
    contact_without_activation_key_indices: list[int],
    initial_active_key_indices: list[int],
    contact_method: str,
    causal_validated: bool,
    status: str,
    failure_reason: str | None,
) -> dict[str, Any]:
    target_set = set(target_keys)
    tp_set = set(gated_true_positive_keys)
    activation_set = set(activation_key_indices)
    false_positive_set = (activation_set - target_set) | (set(activation_without_contact_key_indices) & target_set)
    missed_set = target_set - tp_set
    true_positives = len(tp_set)
    false_positives = len(false_positive_set)
    missed = len(missed_set)
    precision = _binary_metric(true_positives, true_positives + false_positives)
    recall = _binary_metric(true_positives, true_positives + missed)
    f1 = _f1(precision, recall)
    return {
        "status": status,
        "causal_validated": bool(causal_validated),
        "causal_failure_reason": failure_reason,
        "causal_true_positive_events": int(true_positives),
        "causal_false_positive_events": int(false_positives),
        "causal_missed_events": int(missed),
        "causal_precision": float(precision),
        "causal_recall": float(recall),
        "causal_f1": float(f1),
        "contact_gated_success": bool(causal_validated and missed == 0 and false_positives == 0 and bool(target_keys)),
        "contact_gate_passed": bool(causal_validated and not activation_without_contact_key_indices),
        "contact_method": str(contact_method),
        "contact_key_indices": sorted(set(int(key) for key in contact_key_indices)),
        "activation_key_indices": sorted(set(int(key) for key in activation_key_indices)),
        "initial_active_key_indices": sorted(set(int(key) for key in initial_active_key_indices)),
        "activation_without_contact_key_indices": sorted(set(int(key) for key in activation_without_contact_key_indices)),
        "contact_without_activation_key_indices": sorted(set(int(key) for key in contact_without_activation_key_indices)),
        "target_key_indices": list(target_keys),
    }


def _binary_metric(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 1.0 if numerator == 0 else 0.0
    return float(numerator) / float(denominator)


def _f1(precision: float, recall: float) -> float:
    if precision + recall <= 0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def _key_state_array(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        return np.zeros((NUM_PIANO_KEYS,), dtype=np.float32)
    if array.ndim > 1:
        array = array.reshape(array.shape[0], -1)[0]
    output = np.zeros((NUM_PIANO_KEYS,), dtype=np.float32)
    width = min(int(array.shape[0]), NUM_PIANO_KEYS)
    if width > 0:
        output[:width] = array[:width]
    return output


def _binary_roll(roll: np.ndarray, *, threshold: float) -> np.ndarray:
    array = np.asarray(roll, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2:
        array = array.reshape(array.shape[0], -1)
    output = np.zeros((array.shape[0], NUM_PIANO_KEYS), dtype=bool)
    width = min(int(array.shape[1]), NUM_PIANO_KEYS)
    if width > 0:
        output[:, :width] = array[:, :width] >= float(threshold)
    return output


def _align_roll(roll: np.ndarray, steps: int) -> np.ndarray:
    array = np.asarray(roll, dtype=bool)
    if array.shape[0] == int(steps):
        return array
    if array.shape[0] == 1 and int(steps) > 1:
        return np.repeat(array, int(steps), axis=0)
    output = np.zeros((int(steps), NUM_PIANO_KEYS), dtype=bool)
    usable = min(int(steps), int(array.shape[0]))
    if usable > 0:
        output[:usable, : min(array.shape[1], NUM_PIANO_KEYS)] = array[:usable, :NUM_PIANO_KEYS]
    return output


def _activation_events_from_roll(active_roll: np.ndarray) -> list[tuple[int, int]]:
    if active_roll.size == 0:
        return []
    previous = active_roll[0]
    events: list[tuple[int, int]] = []
    for frame_index in range(1, active_roll.shape[0]):
        frame = active_roll[frame_index]
        for key in np.flatnonzero(frame & ~previous).astype(int).tolist():
            events.append((int(key), int(frame_index)))
        previous = frame
    return events


def _active_indices(roll: np.ndarray) -> list[int]:
    array = np.asarray(roll, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    return sorted(np.flatnonzero(np.any(array[:, :NUM_PIANO_KEYS] > 0.5, axis=0)).astype(int).tolist())


def _iter_wrapped_envs(env: Any):
    current = env
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = getattr(current, "_environment", None) or getattr(current, "environment", None) or getattr(current, "env", None)


def _collect_physics_contact_roll(env: Any | None, key_count: int) -> np.ndarray | None:
    if env is None:
        return None
    for current in _iter_wrapped_envs(env):
        physics = getattr(current, "physics", None)
        if physics is None:
            continue
        data = getattr(physics, "data", None)
        model = getattr(physics, "model", None)
        ncon = int(getattr(data, "ncon", 0) or 0) if data is not None else 0
        contacts = getattr(data, "contact", None) if data is not None else None
        if contacts is None or ncon <= 0:
            continue
        roll = np.zeros((1, key_count), dtype=np.float32)
        for index in range(ncon):
            try:
                contact = contacts[index]
                names = [
                    _geom_name(model, int(getattr(contact, "geom1"))),
                    _geom_name(model, int(getattr(contact, "geom2"))),
                ]
            except Exception:
                continue
            key_index = _key_index_from_names(names)
            if key_index is None:
                continue
            if not any(_looks_like_fingertip(name) for name in names):
                continue
            if 0 <= key_index < key_count:
                roll[0, key_index] = 1.0
        if np.any(roll):
            return roll
    return None


def _geom_name(model: Any, geom_id: int) -> str:
    for method_name in ("id2name", "geom_id2name"):
        method = getattr(model, method_name, None)
        if callable(method):
            try:
                if method_name == "id2name":
                    name = method(int(geom_id), "geom")
                else:
                    name = method(int(geom_id))
                if name:
                    return str(name)
            except Exception:
                pass
    names = getattr(model, "geom_names", None)
    if names is not None:
        try:
            return str(names[int(geom_id)])
        except Exception:
            pass
    return ""


def _key_index_from_names(names: Sequence[str]) -> int | None:
    for raw_name in names:
        name = str(raw_name).lower()
        if "key" not in name and "piano" not in name:
            continue
        digits = "".join(char if char.isdigit() else " " for char in name).split()
        for item in reversed(digits):
            value = int(item)
            if 0 <= value < NUM_PIANO_KEYS:
                return value
    return None


def _looks_like_fingertip(name: str) -> bool:
    lower = str(name).lower()
    return any(token in lower for token in ("tip", "finger", "fingertip", "distal"))


def _fingertip_positions_from_env(env: Any | None) -> np.ndarray | None:
    if env is None:
        return None
    for current in _iter_wrapped_envs(env):
        task = getattr(current, "task", None)
        physics = getattr(current, "physics", None)
        if task is None or physics is None:
            continue
        positions = []
        for hand_name in ("right_hand", "left_hand"):
            hand = getattr(task, hand_name, None)
            sites = getattr(hand, "fingertip_sites", None)
            if sites is None:
                continue
            try:
                positions.append(np.asarray(physics.bind(sites).xpos, dtype=np.float32).reshape(-1, 3))
            except Exception:
                continue
        if positions:
            return np.concatenate(positions, axis=0)
    return None


def _key_positions_from_env(env: Any | None, key_count: int) -> np.ndarray | None:
    if env is None:
        return None
    for current in _iter_wrapped_envs(env):
        task = getattr(current, "task", None)
        physics = getattr(current, "physics", None)
        piano = getattr(task, "piano", None)
        if piano is None or physics is None:
            continue
        for attr_name in ("key_sites", "_key_sites", "sites", "key_geoms", "_key_geoms", "geoms", "keys"):
            value = getattr(piano, attr_name, None)
            if value is None:
                continue
            positions = _bound_positions(physics, value)
            if positions is not None and positions.shape[0] >= min(key_count, NUM_PIANO_KEYS):
                return positions[:key_count]
        joints = getattr(piano, "joints", None)
        positions = _bound_positions(physics, joints)
        if positions is not None and positions.shape[0] >= min(key_count, NUM_PIANO_KEYS):
            return positions[:key_count]
    return None


def _bound_positions(physics: Any, value: Any) -> np.ndarray | None:
    try:
        binding = physics.bind(value)
    except Exception:
        return None
    for attr_name in ("xpos", "xanchor", "pos"):
        attr = getattr(binding, attr_name, None)
        if attr is None:
            continue
        try:
            array = np.asarray(attr, dtype=np.float32).reshape(-1, 3)
            if array.size:
                return array
        except Exception:
            continue
    return None
