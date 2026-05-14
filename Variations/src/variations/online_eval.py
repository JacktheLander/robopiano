from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any
import csv

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
for _path in (
    REPO_ROOT / "Bagatelle" / "src",
    REPO_ROOT / "Intermezzo" / "src",
    REPO_ROOT / "Variations" / "src",
    REPO_ROOT / "Variations",
    REPO_ROOT / "partita" / "src",
    REPO_ROOT,
):
    _text = str(_path)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from intermezzo.io import atomic_save_json, atomic_save_npz, create_unique_run_dir, filesystem_slug  # noqa: E402
from intermezzo.midi import load_target_keys_from_midi  # noqa: E402
try:
    from intermezzo.online_eval import (  # type: ignore[import-not-found]  # noqa: E402
        DEFAULT_MAESTRO_ROOT,
        RolloutConfig,
        rollout_hand_targets_headless,
        select_maestro_midi,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised on older main branches.
    from bagatelle.evaluation import RolloutConfig, rollout_bagatelle_hand_targets_headless as rollout_hand_targets_headless  # noqa: E402

    DEFAULT_MAESTRO_ROOT = Path("/WAVE/datasets/ccoelho_lab-jlanders/maestro-v3.0.0/maestro-v3.0.0")

    def select_maestro_midi(
        *,
        midi_path: str | Path | None,
        maestro_root: str | Path,
        selection: str = "shortest",
        piece_index: int = 0,
    ) -> tuple[Path, dict[str, Any]]:
        if midi_path:
            path = Path(midi_path).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"MIDI file not found: {path}")
            return path, {"midi_selection": "explicit", "midi_path": str(path)}

        root = Path(maestro_root).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"MAESTRO root not found: {root}")
        metadata = root / "maestro-v3.0.0.csv"
        if selection == "shortest" and metadata.is_file():
            rows: list[dict[str, str]] = []
            with metadata.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    filename = row.get("midi_filename")
                    duration = row.get("duration")
                    if not filename or duration is None:
                        continue
                    path = root / filename
                    if path.is_file():
                        rows.append({**row, "_path": str(path)})
            if rows:
                rows.sort(key=lambda row: float(row["duration"]))
                row = rows[0]
                return Path(row["_path"]).resolve(), {
                    "midi_selection": "shortest_from_maestro_metadata",
                    "maestro_root": str(root),
                    "maestro_duration_s": float(row["duration"]),
                    "canonical_composer": row.get("canonical_composer"),
                    "canonical_title": row.get("canonical_title"),
                    "split": row.get("split"),
                }

        files = [path for path in root.rglob("*") if path.suffix.lower() in {".mid", ".midi"}]
        files.sort(key=lambda path: path.relative_to(root).as_posix())
        if not files:
            raise RuntimeError(f"No MIDI files found under {root}")
        index = int(piece_index)
        if index < 0 or index >= len(files):
            raise IndexError(f"piece_index {index} out of range for {len(files)} MIDI files")
        return files[index].resolve(), {
            "midi_selection": "sorted_piece_index",
            "piece_index": index,
            "maestro_root": str(root),
        }

if TYPE_CHECKING:
    from simulate.model_loader import LoadedSimulationModel


DEFAULT_VARIATIONS_CHECKPOINT_ROOT = Path("/WAVE/datasets/ccoelho_lab-jlanders/Variations")
DEFAULT_OUTPUT_ROOT = Path("/WAVE/datasets/ccoelho_lab-jlanders/Variations/online_evaluation")
MODEL_TYPES = ("mlp_baseline", "latent_mdn", "diffusion")


@dataclass
class VariationsPredictor:
    model_type: str
    checkpoint_path: Path
    loaded_model: "LoadedSimulationModel"

    @property
    def device(self) -> Any:
        return self.loaded_model.device

    @property
    def diffusion_steps(self) -> int | None:
        if self.model_type != "diffusion":
            return None
        return int(self.loaded_model.diffusion_steps)

    def predict(self, target_keys: np.ndarray, *, batch_size: int = 256) -> np.ndarray:
        values = self.loaded_model.predict_hand_states(target_keys, batch_size=int(batch_size))
        out = np.asarray(values, dtype=np.float32)
        if out.ndim != 2:
            raise ValueError(f"{self.model_type} returned non-matrix hand states: {out.shape}")
        if out.shape[0] != target_keys.shape[0]:
            raise ValueError(
                f"{self.model_type} returned {out.shape[0]} rows for {target_keys.shape[0]} target frames"
            )
        return out


def normalize_model_type(model_type: str) -> str:
    value = str(model_type).strip().lower().replace("-", "_")
    aliases = {
        "mlp": "mlp_baseline",
        "mlpbaseline": "mlp_baseline",
        "mdn": "latent_mdn",
        "latentmdn": "latent_mdn",
    }
    value = aliases.get(value, value)
    if value == "fingerpred":
        raise ValueError(
            "fingerpred predicts 30D fingertip positions and cannot be used for online rollout, "
            "which requires 46D hand-joint targets."
        )
    if value not in MODEL_TYPES:
        raise ValueError(f"Unsupported model_type {model_type!r}; expected one of {', '.join(MODEL_TYPES)}.")
    return value


def _checkpoint_globs(model_type: str) -> tuple[str, ...]:
    if model_type == "mlp_baseline":
        return (
            "*/variations/mlp_baseline/*/checkpoints/best.pt",
            "*/variations/mlp_baseline/checkpoints/best.pt",
        )
    if model_type == "latent_mdn":
        return (
            "*/variations/latent_mdn/mdn/checkpoints/best.pt",
            "*/variations/latent_mdn/*/checkpoints/best.pt",
        )
    if model_type == "diffusion":
        return (
            "*/variations/diffusion/checkpoints/best.pt",
            "*/variations/diffusion/*/checkpoints/best.pt",
        )
    raise ValueError(model_type)


def resolve_checkpoint(
    model_type: str,
    requested: str | Path | None,
    *,
    search_root: str | Path = DEFAULT_VARIATIONS_CHECKPOINT_ROOT,
) -> tuple[Path, dict[str, Any]]:
    normalized = normalize_model_type(model_type)
    meta: dict[str, Any] = {
        "model_type": normalized,
        "requested_checkpoint": str(requested) if requested is not None else None,
    }
    if requested:
        checkpoint = Path(requested).expanduser()
        if checkpoint.is_file():
            resolved = checkpoint.resolve()
            meta["resolved_checkpoint"] = str(resolved)
            meta["checkpoint_resolution"] = "requested_path"
            return resolved, meta
        meta["requested_checkpoint_missing"] = True

    root = Path(search_root).expanduser()
    candidates: list[Path] = []
    for pattern in _checkpoint_globs(normalized):
        candidates.extend(path for path in root.glob(pattern) if path.is_file())
    candidates = sorted(set(candidates), key=lambda path: path.stat().st_mtime, reverse=True)
    if candidates:
        resolved = candidates[0].resolve()
        meta["resolved_checkpoint"] = str(resolved)
        meta["checkpoint_resolution"] = f"latest_{normalized}_best_pt"
        meta["checkpoint_search_root"] = str(root)
        return resolved, meta

    detail = f" requested path {requested!s}" if requested else f" under {root}"
    raise FileNotFoundError(f"No {normalized} checkpoint found{detail}.")


def load_variations_predictor(
    model_type: str,
    checkpoint_path: str | Path,
    *,
    device: str = "auto",
    diffusion_steps: int | None = None,
) -> VariationsPredictor:
    from simulate.model_loader import load_simulation_model

    normalized = normalize_model_type(model_type)
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    loaded = load_simulation_model(
        str(checkpoint),
        normalized,
        device=str(device),
        diffusion_steps=diffusion_steps if normalized == "diffusion" else None,
    )
    return VariationsPredictor(model_type=normalized, checkpoint_path=checkpoint, loaded_model=loaded)


def evaluate_variations_models_online(
    *,
    midi_path: str | Path | None = None,
    maestro_root: str | Path = DEFAULT_MAESTRO_ROOT,
    midi_selection: str = "shortest",
    piece_index: int = 0,
    mlp_checkpoint: str | Path | None = None,
    latent_mdn_checkpoint: str | Path | None = None,
    diffusion_checkpoint: str | Path | None = None,
    checkpoint_search_root: str | Path = DEFAULT_VARIATIONS_CHECKPOINT_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    control_timestep: float = 0.05,
    max_steps: int | None = None,
    max_duration_s: float | None = None,
    batch_size: int = 256,
    diffusion_steps: int | None = None,
    device: str = "auto",
    seed: int = 0,
    threshold: float = 0.5,
    timing_tolerance_s: float = 0.15,
) -> dict[str, Any]:
    midi, midi_meta = select_maestro_midi(
        midi_path=midi_path,
        maestro_root=maestro_root,
        selection=midi_selection,
        piece_index=piece_index,
    )
    target_keys, quantization_meta = load_target_keys_from_midi(
        midi,
        control_timestep=float(control_timestep),
        max_steps=max_steps,
        max_duration_s=max_duration_s,
    )

    requested_by_model = {
        "mlp_baseline": mlp_checkpoint,
        "latent_mdn": latent_mdn_checkpoint,
        "diffusion": diffusion_checkpoint,
    }
    checkpoint_meta: dict[str, dict[str, Any]] = {}
    predictors: dict[str, VariationsPredictor] = {}
    for model_type, requested in requested_by_model.items():
        checkpoint, meta = resolve_checkpoint(model_type, requested, search_root=checkpoint_search_root)
        checkpoint_meta[model_type] = meta
        predictors[model_type] = load_variations_predictor(
            model_type,
            checkpoint,
            device=device,
            diffusion_steps=diffusion_steps,
        )

    run_name = (
        f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%SZ')}_"
        f"{filesystem_slug(Path(midi).stem)}_variations_models"
    )
    run_dir = create_unique_run_dir(output_root, run_name=run_name, prefix="variations_online")
    rollout_config = RolloutConfig(
        control_timestep=float(control_timestep),
        threshold=float(threshold),
        timing_tolerance_s=float(timing_tolerance_s),
        seed=int(seed),
    )

    model_targets: dict[str, np.ndarray] = {}
    results: dict[str, dict[str, Any]] = {}
    for model_type, predictor in predictors.items():
        hand_targets = predictor.predict(target_keys, batch_size=int(batch_size))
        model_targets[f"{model_type}_hand_joints"] = hand_targets
        results[model_type] = rollout_hand_targets_headless(
            hand_targets=hand_targets,
            target_keys=target_keys,
            output_dir=run_dir,
            label=model_type,
            config=rollout_config,
        )

    atomic_save_npz(run_dir / "model_hand_targets.npz", target_keys=target_keys, **model_targets)
    summary = {
        "run_dir": str(run_dir),
        "midi_path": str(midi),
        "midi": midi_meta,
        "midi_quantization": quantization_meta,
        "checkpoints": checkpoint_meta,
        "control_timestep": float(control_timestep),
        "max_steps": max_steps,
        "max_duration_s": max_duration_s,
        "batch_size": int(batch_size),
        "device": str(next(iter(predictors.values())).device),
        "diffusion_steps": predictors["diffusion"].diffusion_steps,
        "target_keys_shape": list(target_keys.shape),
        "models": results,
    }
    atomic_save_json(run_dir / "summary.json", summary)
    return summary
