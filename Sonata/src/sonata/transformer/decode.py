from __future__ import annotations

from typing import Any

import torch


def scale_logits(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 1.0:
        return logits
    return logits / max(float(temperature), 1e-6)


def mask_logits_to_family(logits: torch.Tensor, family_index: torch.Tensor, family_mask: torch.Tensor) -> torch.Tensor:
    mask = family_mask[family_index]
    fill_value = -1.0e4
    return logits.masked_fill(~mask, fill_value)


def collapse_weak_primitive_logits(
    primitive_logits: torch.Tensor,
    remap_tensor: torch.Tensor | None,
) -> torch.Tensor:
    if remap_tensor is None:
        return primitive_logits
    collapsed = primitive_logits.clone()
    remap = remap_tensor.to(device=collapsed.device, dtype=torch.long)
    for weak_index, strong_index in enumerate(remap.tolist()):
        if int(strong_index) == int(weak_index):
            continue
        collapsed[..., int(strong_index)] = torch.logaddexp(
            collapsed[..., int(strong_index)],
            collapsed[..., int(weak_index)],
        )
        collapsed[..., int(weak_index)] = -1.0e4
    return collapsed


def decode_factored_outputs(
    outputs: dict[str, torch.Tensor],
    *,
    family_mask: torch.Tensor,
    temperature: float = 1.0,
    remap_tensor: torch.Tensor | None = None,
) -> dict[str, Any]:
    scaled_family_logits = scale_logits(torch.nan_to_num(outputs["family_logits"]), temperature)
    scaled_primitive_logits = scale_logits(torch.nan_to_num(outputs["primitive_logits"]), temperature)
    scaled_primitive_logits = collapse_weak_primitive_logits(scaled_primitive_logits, remap_tensor)
    scaled_duration_logits = scale_logits(torch.nan_to_num(outputs["duration_logits"]), temperature)
    scaled_dynamics_logits = scale_logits(torch.nan_to_num(outputs["dynamics_logits"]), temperature)

    predicted_family = scaled_family_logits.argmax(dim=-1)
    raw_predicted_primitive = scaled_primitive_logits.argmax(dim=-1)
    masked_primitive_logits = mask_logits_to_family(
        scaled_primitive_logits,
        predicted_family,
        family_mask,
    )
    predicted_primitive = masked_primitive_logits.argmax(dim=-1)
    predicted_duration = scaled_duration_logits.argmax(dim=-1)
    predicted_dynamics = scaled_dynamics_logits.argmax(dim=-1)

    raw_decode_valid = family_mask[predicted_family, raw_predicted_primitive]

    return {
        "scaled_family_logits": scaled_family_logits,
        "scaled_primitive_logits": scaled_primitive_logits,
        "scaled_duration_logits": scaled_duration_logits,
        "scaled_dynamics_logits": scaled_dynamics_logits,
        "predicted_family": predicted_family,
        "predicted_primitive": predicted_primitive,
        "predicted_duration": predicted_duration,
        "predicted_dynamics": predicted_dynamics,
        "raw_predicted_primitive": raw_predicted_primitive,
        "raw_decode_valid_under_family_mask": raw_decode_valid,
        "masked_primitive_logits": masked_primitive_logits,
    }
