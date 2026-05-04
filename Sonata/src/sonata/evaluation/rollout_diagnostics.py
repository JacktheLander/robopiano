"""Diagnostics helpers for DM Control RoboPianist rollouts (no Stage 3)."""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np


def resolve_ffmpeg_executable() -> str | None:
    for key in ("SONATA_FFMPEG", "FFMPEG_PATH"):
        raw = os.environ.get(key)
        if raw:
            p = Path(raw).expanduser()
            if p.is_file():
                return str(p)
    which = shutil.which("ffmpeg")
    if which:
        return which
    try:
        import imageio_ffmpeg  # type: ignore

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).exists():
            return str(exe)
    except Exception:
        pass
    return None


def format_int_list(arr: np.ndarray | list[int]) -> str:
    if isinstance(arr, np.ndarray):
        flat = np.asarray(arr).astype(np.int64).reshape(-1).tolist()
    else:
        flat = list(arr)
    return ";".join(str(int(x)) for x in flat)


def iter_wrapped_env_chain(env: Any) -> list[Any]:
    chain: list[Any] = []
    current: Any = env
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        chain.append(current)
        seen.add(id(current))
        current = getattr(current, "_environment", None)
    return chain


def get_inner_task(env: Any) -> Any:
    for node in iter_wrapped_env_chain(env):
        task = getattr(node, "task", None)
        if task is not None:
            return task
    return None


def get_inner_physics(env: Any) -> Any:
    for node in iter_wrapped_env_chain(env):
        physics = getattr(node, "physics", None)
        if physics is not None:
            return physics
    return None


def get_piano(env: Any) -> Any:
    task = get_inner_task(env)
    return getattr(task, "piano", None) if task is not None else None


def target_key_indices_after_step(env: Any) -> list[int]:
    """Keys expected active for the timestep that just completed (RoboPianist task)."""
    task = get_inner_task(env)
    if task is None:
        return []
    notes_attr = getattr(task, "_notes", None)
    t_idx = int(getattr(task, "_t_idx", 0))
    if notes_attr is None or t_idx < 1:
        return []
    step_notes = notes_attr[t_idx - 1]
    return [int(n.key) for n in step_notes]


def pressed_key_indices(piano: Any, *, threshold: float = 0.1) -> list[int]:
    if piano is None:
        return []
    act = getattr(piano, "activation", None)
    if act is None:
        return []
    arr = np.asarray(act).astype(np.float64).reshape(-1)
    return np.flatnonzero(arr > threshold).astype(np.int64).tolist()


def flatten_fingertip_xyz(env: Any) -> str | None:
    physics = get_inner_physics(env)
    task = get_inner_task(env)
    if physics is None or task is None:
        return None
    parts: list[str] = []
    try:
        for hand_name in ("right_hand", "left_hand"):
            hand = getattr(task, hand_name, None)
            if hand is None:
                continue
            sites = getattr(hand, "fingertip_sites", None) or []
            for site in sites:
                pos = physics.bind(site).xpos.copy()
                parts.extend([f"{float(x):.4f}" for x in pos])
    except Exception:
        return None
    return "|".join(parts) if parts else None


def flatten_key_qpos_sample(piano: Any, *, max_keys: int = 88) -> str | None:
    state = getattr(piano, "state", None)
    if state is None:
        return None
    try:
        arr = np.asarray(state, dtype=np.float64).reshape(-1)[:max_keys]
        return "|".join(f"{float(x):.5f}" for x in arr[: min(len(arr), max_keys)])
    except Exception:
        return None


def micro_precision_recall(overlap: int, num_target: int, num_pressed: int) -> tuple[float, float]:
    false_positive = max(num_pressed - overlap, 0)
    false_negative = max(num_target - overlap, 0)
    precision = float(overlap / max(overlap + false_positive, 1))
    recall = float(overlap / max(overlap + false_negative, 1))
    return precision, recall


def cumulative_prf(cum_tp: int, cum_fp: int, cum_fn: int) -> tuple[float, float, float]:
    p = float(cum_tp / max(cum_tp + cum_fp, 1))
    r = float(cum_tp / max(cum_tp + cum_fn, 1))
    f1 = 0.0 if p + r == 0 else float(2 * p * r / (p + r))
    return p, r, f1


def collect_robot_midi_since_reset(env: Any) -> list[Any]:
    piano = get_piano(env)
    if piano is None:
        return []
    midi_module = getattr(piano, "midi_module", None)
    get_all = getattr(midi_module, "get_all_midi_messages", None)
    if not callable(get_all):
        return []
    try:
        return list(get_all())
    except Exception:
        return []


def save_env_introspection(env: Any, output_path: Path) -> None:
    lines: list[str] = []
    chain = iter_wrapped_env_chain(env)
    lines.append("Wrapper / environment chain (outer → inner):")
    for i, node in enumerate(chain):
        lines.append(f"  [{i}] {type(node).__module__}.{type(node).__name__}")

    inner = chain[-1] if chain else None
    action_spec = getattr(inner, "action_spec", None)
    if callable(action_spec):
        try:
            spec = action_spec()
            lines.append("")
            lines.append(f"action_spec: {spec}")
            lines.append(f"  shape: {getattr(spec, 'shape', None)}")
            lines.append(f"  dtype: {getattr(spec, 'dtype', None)}")
            lines.append(f"  minimum: {getattr(spec, 'minimum', None)}")
            lines.append(f"  maximum: {getattr(spec, 'maximum', None)}")
        except Exception as exc:
            lines.append(f"action_spec() failed: {exc}")

    task = get_inner_task(env)
    lines.append("")
    lines.append(f"task class: {type(task).__module__}.{type(task).__name__}" if task else "task: None")

    piano = get_piano(env)
    lines.append(f"piano class: {type(piano).__module__}.{type(piano).__name__}" if piano else "piano: None")

    def scan_attrs(obj: Any, needles: tuple[str, ...]) -> list[str]:
        found: list[str] = []
        if obj is None:
            return found
        for name in sorted(dir(obj)):
            if name.startswith("_") and name not in ("__class__",):
                continue
            low = name.lower()
            if any(n in low for n in needles):
                found.append(name)
        return found

    if task is not None:
        lines.append("")
        lines.append(
            "task attributes (filtered): "
            + ", ".join(scan_attrs(task, ("midi", "note", "key", "piano", "goal", "control", "action")))
        )
    if piano is not None:
        lines.append(
            "piano attributes (filtered): "
            + ", ".join(scan_attrs(piano, ("midi", "key", "note", "activation", "control", "state", "qpos")))
        )
        mm = getattr(piano, "midi_module", None)
        lines.append("")
        lines.append(f"midi_module class: {type(mm).__module__}.{type(mm).__name__}" if mm else "midi_module: None")
        if mm is not None:
            lines.append(
                "midi_module methods (subset): "
                + ", ".join(
                    m for m in sorted(dir(mm)) if "midi" in m.lower() or m.startswith("get_") or "callback" in m.lower()
                )
            )

    lines.append("")
    lines.append(
        "Reference score MIDI: task._notes / task._midi — target trajectory used for rewards/metrics."
    )
    lines.append(
        "Robot-generated MIDI stream: piano.midi_module tracks key activation *changes* from simulation "
        "(NoteOn/NoteOff from physics), via get_all_midi_messages()."
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def overlay_debug_text_on_frame(
    frame: np.ndarray,
    *,
    lines: list[str],
    margin: int = 8,
) -> np.ndarray:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return frame
    img = Image.fromarray(frame.copy())
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    y = margin
    text_color = (255, 255, 0)
    shadow = (0, 0, 0)
    for line in lines[:14]:
        if font is not None:
            for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
                draw.text((margin + dx, y + dy), line[:120], font=font, fill=shadow)
            draw.text((margin, y), line[:120], font=font, fill=text_color)
            y += 12
        else:
            draw.text((margin, y), line[:120], fill=text_color)
            y += 12
    return np.asarray(img, dtype=np.uint8)


def synthesize_reference_wav(midi_path: Path, wav_path: Path, *, logger: Any = None) -> bool:
    try:
        from robopianist.music import midi_file as mf

        seq = mf.MidiFile.from_file(midi_path)
        waveform_float = seq.synthesize()
        normalizer = float(np.iinfo(np.int16).max)
        waveform = np.asarray(waveform_float * normalizer, dtype=np.int16)
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        import wave

        with wave.open(str(wav_path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(44100)
            handle.writeframes(waveform.tobytes())
        return True
    except Exception as exc:
        if logger is not None:
            logger.warning("Reference MIDI synthesize failed: %s", exc)
        return False


def mux_video_with_wav(
    *,
    video_path: Path,
    wav_path: Path,
    temp_root: Path,
    ffmpeg_exe: str,
    logger: Any,
) -> str | None:
    import shutil as sh
    import subprocess
    import tempfile

    temp_root.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"{video_path.stem}_refaudio_", dir=str(temp_root)))
    try:
        tv = tmp_dir / video_path.name
        sh.copyfile(video_path, tv)
        cmd = [
            ffmpeg_exe,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(tv),
            "-i",
            str(wav_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(video_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return None
    except Exception as exc:
        return f"Reference audio mux failed: {exc}"
    finally:
        sh.rmtree(tmp_dir, ignore_errors=True)


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "episode"
