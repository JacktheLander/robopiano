from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    torch = None

    class Dataset:  # type: ignore[no-redef]
        pass

from variations.data.fingerpred import coord_mask_from_tip_mask, infer_active_tip_mask
from variations.data.press_extractor import goal_fingerprint
from variations.utils.io import ensure_dir, load_csv, save_csv, save_json

JOINT_STATE_DIM = 46
EXTRACTED_HAND_STATE_DIM = 76
FINGERTIP_STATE_DIM = EXTRACTED_HAND_STATE_DIM - JOINT_STATE_DIM


def assign_split(song_id: str, seed: int, val_fraction: float) -> str:
    h = int(hashlib.sha256(f"{seed}:{song_id}".encode("utf-8")).hexdigest(), 16)
    return "val" if (h % 10_000) / 10_000.0 < val_fraction else "train"


def _manifest_rows(extraction_root: str | Path) -> list[dict[str, str]]:
    manifest = Path(extraction_root) / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(
            f"Missing manifest.csv at {manifest}\n"
            "Run extraction first, for example:\n"
            "  python Variations/scripts/extract_press_pairs.py --config Variations/configs/extraction/debug.yaml\n"
            "Then point evaluation configs at that folder (e.g. extraction_root: Variations/outputs/extraction/debug), "
            "or pass --extraction-root and/or set VARIATIONS_OUTPUT_ROOT."
        )
    return load_csv(manifest)


def _row_count(row: dict[str, str]) -> int:
    for key in ("rows_accepted", "num_rows"):
        if row.get(key) not in {None, ""}:
            return int(float(row[key]))
    return 0


def build_splits(
    extraction_root: str | Path,
    *,
    val_fraction: float,
    seed: int,
    min_pairs_per_split: int = 1000,
    force: bool = False,
) -> Path:
    root = Path(extraction_root)
    split_dir = ensure_dir(root / "splits")
    split_path = split_dir / "split_index.csv"
    if split_path.exists() and not force:
        return split_path
    rows = _manifest_rows(root)
    split_rows = []
    totals = {"train": 0, "val": 0}
    for row in rows:
        song_id = str(row["song_id"])
        split = assign_split(song_id, seed, val_fraction)
        count = _row_count(row)
        totals[split] += count
        split_rows.append({"song_id": song_id, "safe_song_id": row.get("safe_song_id", ""), "split": split, "rows_accepted": count})
    if totals["train"] < min_pairs_per_split or totals["val"] < min_pairs_per_split:
        raise RuntimeError(
            f"Split has too few pairs: train={totals['train']} val={totals['val']} "
            f"min_pairs_per_split={min_pairs_per_split}. Lower the guard for debug runs."
        )
    save_csv(split_path, split_rows, fieldnames=["song_id", "safe_song_id", "split", "rows_accepted"])
    save_json(split_dir / "summary.json", {"seed": seed, "val_fraction": val_fraction, "pair_totals": totals, "num_songs": len(split_rows)})
    return split_path


def load_split_index(extraction_root: str | Path) -> list[dict[str, str]]:
    return load_csv(Path(extraction_root) / "splits" / "split_index.csv")


def _song_npz_path(extraction_root: Path, row: dict[str, str]) -> Path:
    if row.get("path"):
        path = Path(row["path"])
        return path if path.is_absolute() else extraction_root / path
    safe = row.get("safe_song_id") or row["song_id"]
    return extraction_root / f"song_{safe}.npz"


def normalize_output_mode(output_mode: str | None) -> str:
    value = str(output_mode or "joints_only").strip().lower()
    aliases = {"joint": "joints_only", "joints": "joints_only", "active_fingertips": "fingerpred"}
    value = aliases.get(value, value)
    if value not in {"joints_only", "fingerpred"}:
        raise ValueError(f"Unsupported output_mode {output_mode!r}; expected joints_only or fingerpred.")
    return value


def norm_stats_path_for_mode(extraction_root: str | Path, output_mode: str) -> Path:
    root = Path(extraction_root)
    mode = normalize_output_mode(output_mode)
    name = "norm_stats.npz" if mode == "joints_only" else "norm_stats_fingerpred.npz"
    return root / "splits" / name


def compute_norm_stats(extraction_root: str | Path, *, force: bool = False) -> Path:
    root = Path(extraction_root)
    out = ensure_dir(root / "splits") / "norm_stats.npz"
    if out.exists() and not force:
        mean, _std = load_norm_stats(out)
        if mean.shape[0] == JOINT_STATE_DIM:
            return out
    rows = [row for row in load_split_index(root) if row["split"] == "train"]
    chunks = []
    for row in rows:
        path = _song_npz_path(root, row)
        if not path.exists():
            continue
        data = np.load(path, allow_pickle=False)
        hand = np.asarray(data["hand_state"], dtype=np.float32)[:, :JOINT_STATE_DIM]
        if hand.size:
            chunks.append(hand)
    if not chunks:
        raise RuntimeError(f"No train joint_state rows found under {root}")
    arr = np.concatenate(chunks, axis=0)
    mean = arr.mean(axis=0).astype(np.float32)
    std = arr.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    np.savez_compressed(out, mean=mean, std=std, state_dim=np.asarray(JOINT_STATE_DIM, dtype=np.int32))
    return out


def compute_fingerpred_norm_stats(
    extraction_root: str | Path,
    *,
    force: bool = False,
    key_positions: np.ndarray | None = None,
) -> Path:
    root = Path(extraction_root)
    out = ensure_dir(root / "splits") / "norm_stats_fingerpred.npz"
    if out.exists() and not force:
        mean, _std = load_norm_stats(out, expected_dim=FINGERTIP_STATE_DIM)
        if mean.shape[0] == FINGERTIP_STATE_DIM:
            return out
    rows = [row for row in load_split_index(root) if row["split"] == "train"]
    weighted_sum = np.zeros((FINGERTIP_STATE_DIM,), dtype=np.float64)
    weighted_sq_sum = np.zeros((FINGERTIP_STATE_DIM,), dtype=np.float64)
    counts = np.zeros((FINGERTIP_STATE_DIM,), dtype=np.float64)
    fallback_chunks = []
    for row in rows:
        path = _song_npz_path(root, row)
        if not path.exists():
            continue
        data = np.load(path, allow_pickle=False)
        target = np.asarray(data["target_keys"], dtype=np.float32)
        full_hand = np.asarray(data["hand_state"], dtype=np.float32)
        if full_hand.shape[1] < EXTRACTED_HAND_STATE_DIM or target.shape[0] == 0:
            continue
        fingertips = full_hand[:, JOINT_STATE_DIM:EXTRACTED_HAND_STATE_DIM]
        mask = coord_mask_from_tip_mask(infer_active_tip_mask(target, fingertips, key_positions=key_positions))
        weighted_sum += (fingertips * mask).sum(axis=0, dtype=np.float64)
        weighted_sq_sum += ((fingertips * fingertips) * mask).sum(axis=0, dtype=np.float64)
        counts += mask.sum(axis=0, dtype=np.float64)
        fallback_chunks.append(fingertips)
    if not fallback_chunks:
        raise RuntimeError(f"No train fingertip rows found under {root}")
    fallback = np.concatenate(fallback_chunks, axis=0)
    fallback_mean = fallback.mean(axis=0).astype(np.float64)
    fallback_std = fallback.std(axis=0).astype(np.float64)
    safe_counts = np.maximum(counts, 1.0)
    mean = weighted_sum / safe_counts
    var = np.maximum(weighted_sq_sum / safe_counts - mean * mean, 0.0)
    std = np.sqrt(var)
    mean = np.where(counts > 0, mean, fallback_mean).astype(np.float32)
    std = np.where((counts > 0) & (std >= 1e-6), std, np.where(fallback_std >= 1e-6, fallback_std, 1.0)).astype(np.float32)
    np.savez_compressed(out, mean=mean, std=std, state_dim=np.asarray(FINGERTIP_STATE_DIM, dtype=np.int32), output_mode="fingerpred")
    return out


def load_norm_stats(path: str | Path, *, expected_dim: int | None = JOINT_STATE_DIM) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    mean = np.asarray(data["mean"], dtype=np.float32)
    std = np.asarray(data["std"], dtype=np.float32)
    if expected_dim is not None and mean.shape[0] > int(expected_dim):
        mean = mean[: int(expected_dim)]
        std = std[: int(expected_dim)]
    return mean, std


def _target_bucket(target_keys: np.ndarray) -> tuple[int, int, int, int]:
    keys = np.asarray(target_keys) > 0.5
    return (
        int(keys.sum()),
        int(keys[:29].any()),
        int(keys[29:59].any()),
        int(keys[59:].any()),
    )


def split_target_coverage(train_targets: np.ndarray, val_targets: np.ndarray) -> float:
    if len(val_targets) == 0:
        return 0.0
    train_buckets = {_target_bucket(row) for row in train_targets}
    hits = sum(1 for row in val_targets if _target_bucket(row) in train_buckets)
    return float(hits / max(len(val_targets), 1))


class PressPairsDataset(Dataset):
    def __init__(
        self,
        extraction_root: str | Path,
        *,
        split: str,
        norm_stats_path: str | Path | None = None,
        assert_unique_goals: bool = False,
        output_mode: str = "joints_only",
        key_positions: np.ndarray | None = None,
    ) -> None:
        self.extraction_root = Path(extraction_root)
        self.split = split
        self.output_mode = normalize_output_mode(output_mode)
        split_rows = [row for row in load_split_index(self.extraction_root) if row["split"] == split]
        targets: list[np.ndarray] = []
        hands: list[np.ndarray] = []
        fingertips: list[np.ndarray] = []
        active_masks: list[np.ndarray] = []
        song_ids = []
        for row in split_rows:
            path = _song_npz_path(self.extraction_root, row)
            if not path.exists():
                continue
            data = np.load(path, allow_pickle=False)
            target = np.asarray(data["target_keys"], dtype=np.float32)
            full_hand = np.asarray(data["hand_state"], dtype=np.float32)
            hand = full_hand[:, :JOINT_STATE_DIM]
            if full_hand.shape[1] >= EXTRACTED_HAND_STATE_DIM:
                fingertip = full_hand[:, JOINT_STATE_DIM:EXTRACTED_HAND_STATE_DIM]
            else:
                fingertip = np.zeros((full_hand.shape[0], FINGERTIP_STATE_DIM), dtype=np.float32)
            if target.shape[0] == 0:
                continue
            targets.append(target)
            hands.append(hand)
            fingertips.append(fingertip)
            if self.output_mode == "fingerpred":
                active_masks.append(infer_active_tip_mask(target, fingertip, key_positions=key_positions))
            song_ids.extend([str(row["song_id"])] * target.shape[0])
        if targets:
            self.target_keys = np.concatenate(targets, axis=0).astype(np.float32)
            self.hand_state = np.concatenate(hands, axis=0).astype(np.float32)
            self.fingertip_state = np.concatenate(fingertips, axis=0).astype(np.float32)
        else:
            self.target_keys = np.zeros((0, 88), dtype=np.float32)
            self.hand_state = np.zeros((0, JOINT_STATE_DIM), dtype=np.float32)
            self.fingertip_state = np.zeros((0, FINGERTIP_STATE_DIM), dtype=np.float32)
        self.active_tip_mask = (
            np.concatenate(active_masks, axis=0).astype(np.float32)
            if active_masks
            else np.zeros((self.target_keys.shape[0], 10), dtype=np.float32)
        )
        self.song_ids = np.asarray(song_ids)
        if assert_unique_goals:
            seen = set()
            for target in self.target_keys:
                fp = goal_fingerprint(target)
                if fp in seen:
                    raise AssertionError(f"Duplicate target_keys found in split {split}")
                seen.add(fp)
        if norm_stats_path is None:
            norm_stats_path = norm_stats_path_for_mode(self.extraction_root, self.output_mode)
        expected_dim = JOINT_STATE_DIM if self.output_mode == "joints_only" else FINGERTIP_STATE_DIM
        self.mean, self.std = load_norm_stats(norm_stats_path, expected_dim=expected_dim)
        if self.mean.shape[0] != expected_dim:
            raise ValueError(
                f"Normalizer width {self.mean.shape[0]} does not match {self.output_mode} target width {expected_dim}"
            )
        if self.output_mode == "joints_only":
            self.target_state = self.hand_state
            self.target_coord_mask = np.ones_like(self.hand_state, dtype=np.float32)
            self.hand_state_normalized = ((self.hand_state - self.mean) / self.std).astype(np.float32)
            self.target_state_normalized = self.hand_state_normalized
        else:
            self.target_state = self.fingertip_state
            self.target_coord_mask = coord_mask_from_tip_mask(self.active_tip_mask)
            self.target_state_normalized = ((self.target_state - self.mean) / self.std).astype(np.float32)

    def __len__(self) -> int:
        return int(self.target_keys.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        if torch is None:
            raise RuntimeError("PyTorch is required to materialize PressPairsDataset samples.")
        item = {
            "target_keys": torch.from_numpy(self.target_keys[index]),
            "hand_state": torch.from_numpy(self.hand_state[index]),
            "fingertip_state": torch.from_numpy(self.fingertip_state[index]),
            "joint_state": torch.from_numpy(self.hand_state[index]),
            "hand_fingertips": torch.from_numpy(self.fingertip_state[index]),
            "target_state": torch.from_numpy(self.target_state[index]),
            "target_state_normalized": torch.from_numpy(self.target_state_normalized[index]),
            "active_tip_mask": torch.from_numpy(self.active_tip_mask[index]),
            "target_coord_mask": torch.from_numpy(self.target_coord_mask[index]),
        }
        if self.output_mode == "joints_only":
            item["hand_state_normalized"] = torch.from_numpy(self.hand_state_normalized[index])
            item["joint_state_normalized"] = torch.from_numpy(self.hand_state_normalized[index])
        return item
