from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch


def load_primitive_remap(path: Path | str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = Path(path)
    if not resolved.exists():
        return None
    payload = json.loads(resolved.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Primitive remap payload must be a JSON object: {resolved}")
    enabled = bool(payload.get("enabled", True))
    if enabled and "remap" not in payload:
        raise ValueError(f"Enabled primitive remap must include a remap object: {resolved}")
    remap = payload.get("remap", {})
    if enabled and not isinstance(remap, dict):
        raise ValueError(f"Enabled primitive remap must include a remap object: {resolved}")
    normalized = dict(payload)
    normalized["enabled"] = enabled
    normalized["mode"] = str(normalized.get("mode", "weak_to_strong"))
    normalized["remap"] = {str(key): str(value) for key, value in dict(remap).items()}
    normalized["_path"] = str(resolved)
    return normalized


def resolve_remap_path(primitive_root: Path, config: dict[str, Any] | None) -> Path | None:
    if config is not None and config.get("path") is not None:
        return Path(config["path"]).expanduser()
    return primitive_root / "artifacts" / "primitive_remap.json"


def apply_primitive_remap_to_token_df(
    token_df: pd.DataFrame,
    vocabulary: dict[str, Any],
    remap_payload: dict[str, Any] | None,
    *,
    apply_to_history: bool = False,
    preserve_original_columns: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    primitive_ids = [str(item) for item in vocabulary.get("primitive_ids", [])]
    summary = summarize_remap(
        remap_payload,
        primitive_ids=primitive_ids,
        path=(remap_payload or {}).get("_path"),
        apply_to_history=apply_to_history,
    )
    if remap_payload is None or not bool(remap_payload.get("enabled", False)):
        return token_df, summary
    if "primitive_id" not in token_df.columns or "primitive_index" not in token_df.columns:
        raise ValueError("Primitive remap requires primitive_id and primitive_index columns.")

    id_to_index = {primitive_id: index for index, primitive_id in enumerate(primitive_ids)}
    remap_index: dict[int, int] = {}
    for raw_source, raw_target in remap_payload.get("remap", {}).items():
        source_index = _resolve_primitive_index(raw_source, primitive_ids)
        target_index = _resolve_primitive_index(raw_target, primitive_ids)
        if source_index is None or target_index is None:
            raise ValueError(f"Primitive remap references unknown primitive: {raw_source!r}->{raw_target!r}")
        if source_index != target_index:
            remap_index[int(source_index)] = int(target_index)

    if not remap_index:
        return token_df.copy(), summary

    remapped = token_df.copy()
    if preserve_original_columns:
        if "original_primitive_id" not in remapped.columns:
            remapped["original_primitive_id"] = remapped["primitive_id"].astype(str)
        if "original_primitive_index" not in remapped.columns:
            remapped["original_primitive_index"] = remapped["primitive_index"].astype(int)

    original_indices = remapped["primitive_index"].astype(int)
    new_indices = original_indices.map(lambda index: remap_index.get(int(index), int(index))).astype(int)
    remapped["primitive_index"] = new_indices
    remapped["primitive_id"] = new_indices.map(lambda index: primitive_ids[int(index)]).astype(str)

    unresolved_ids = sorted(set(remapped["primitive_id"].astype(str)) - set(id_to_index))
    if unresolved_ids:
        raise ValueError(f"Remapped primitive_id values are not in the vocabulary: {unresolved_ids[:5]}")

    summary = summarize_remap(
        remap_payload,
        primitive_ids=primitive_ids,
        path=remap_payload.get("_path"),
        apply_to_history=apply_to_history,
    )
    summary["num_remapped_rows"] = int((original_indices.to_numpy() != new_indices.to_numpy()).sum())
    return remapped, summary


def build_remap_tensor(
    num_primitives: int,
    remap_payload: dict[str, Any] | None,
    primitive_ids: list[str],
    device: torch.device | None = None,
) -> torch.Tensor | None:
    if remap_payload is None or not bool(remap_payload.get("enabled", False)):
        return None
    tensor = torch.arange(int(num_primitives), dtype=torch.long, device=device)
    for raw_source, raw_target in remap_payload.get("remap", {}).items():
        source_index = _resolve_primitive_index(raw_source, primitive_ids)
        target_index = _resolve_primitive_index(raw_target, primitive_ids)
        if source_index is None or target_index is None:
            raise ValueError(f"Primitive remap references unknown primitive: {raw_source!r}->{raw_target!r}")
        if 0 <= source_index < num_primitives and 0 <= target_index < num_primitives:
            tensor[int(source_index)] = int(target_index)
    if bool(torch.all(tensor == torch.arange(int(num_primitives), dtype=torch.long, device=device)).item()):
        return None
    return tensor


def summarize_remap(
    remap_payload: dict[str, Any] | None,
    *,
    primitive_ids: list[str],
    path: str | Path | None = None,
    apply_to_history: bool | None = None,
) -> dict[str, Any]:
    if remap_payload is None or not bool(remap_payload.get("enabled", False)):
        return {
            "enabled": False,
            "num_remapped_primitives": 0,
            "remapped_primitives": [],
            "canonical_targets": [],
            "remap": {},
            "path": str(path) if path is not None else None,
            "apply_to_history": bool(apply_to_history) if apply_to_history is not None else False,
        }
    remapped_primitives: list[str] = []
    canonical_targets: list[str] = []
    normalized_remap: dict[str, str] = {}
    for raw_source, raw_target in remap_payload.get("remap", {}).items():
        source_index = _resolve_primitive_index(raw_source, primitive_ids)
        target_index = _resolve_primitive_index(raw_target, primitive_ids)
        if source_index is None or target_index is None or source_index == target_index:
            continue
        source_id = primitive_ids[int(source_index)]
        target_id = primitive_ids[int(target_index)]
        remapped_primitives.append(source_id)
        canonical_targets.append(target_id)
        normalized_remap[source_id] = target_id
    return {
        "enabled": True,
        "mode": str(remap_payload.get("mode", "weak_to_strong")),
        "num_remapped_primitives": len(remapped_primitives),
        "remapped_primitives": remapped_primitives,
        "canonical_targets": sorted(set(canonical_targets)),
        "remap": normalized_remap,
        "path": str(path) if path is not None else remap_payload.get("_path"),
        "apply_to_history": bool(apply_to_history) if apply_to_history is not None else False,
    }


def _resolve_primitive_index(value: Any, primitive_ids: list[str]) -> int | None:
    text = str(value)
    id_to_index = {primitive_id: index for index, primitive_id in enumerate(primitive_ids)}
    if text in id_to_index:
        return int(id_to_index[text])
    try:
        index = int(text)
    except ValueError:
        return None
    if 0 <= index < len(primitive_ids):
        return index
    return None
