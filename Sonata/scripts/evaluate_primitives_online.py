from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sonata.config import load_yaml, resolve_path
from sonata.evaluation.primitive_online_eval import evaluate_primitives_online
from sonata.utils.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Stage 1 primitive usefulness online in RoboPianist simulation.",
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "evaluation" / "primitives_online.yaml"))
    parser.add_argument("--profile", default="debug")
    parser.add_argument("--primitive-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--primitive-ids", nargs="*", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--instances-per-primitive", type=int, default=None)
    parser.add_argument("--min-chord-size", type=int, default=None)
    parser.add_argument("--max-chord-size", type=int, default=None)
    parser.add_argument("--min-duration-steps", type=int, default=None)
    parser.add_argument("--max-duration-steps", type=int, default=None)
    parser.add_argument("--use-goals", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-piano-states", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save-debug", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save-plots", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--rollout-device", default=None)
    parser.add_argument("--rollout-source-mode", default=None)
    parser.add_argument("--example-midi-paths", nargs="*", default=None)
    parser.add_argument("--robopianist-root", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logger = configure_logging(args.log_level)
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)

    config["primitive_root"] = str(
        _resolve_user_path(
            args.primitive_root or config.get("primitive_root") or PROJECT_ROOT / "outputs" / "primitives" / args.profile
        )
    )
    config["output_root"] = str(
        _resolve_user_path(
            args.output_root
            or config.get("output_root")
            or PROJECT_ROOT / "outputs" / "evaluation" / "primitives_online" / args.profile
        )
    )
    if args.robopianist_root is not None:
        config["robopianist_root"] = str(_resolve_user_path(args.robopianist_root))
    if args.rollout_device is not None:
        config["rollout_device"] = str(args.rollout_device)
    if args.seed is not None:
        config["seed"] = int(args.seed)
    if args.resume is not None:
        config["resume"] = bool(args.resume)
    if args.save_debug is not None:
        config["save_debug"] = bool(args.save_debug)
    if args.save_plots is not None:
        config["save_plots"] = bool(args.save_plots)
    config["force"] = bool(args.force or config.get("force", False))

    sampling = dict(config.get("sampling", {}))
    events = dict(config.get("events", {}))
    rollout = dict(config.get("rollout", {}))
    if args.max_instances is not None:
        sampling["max_instances"] = int(args.max_instances)
    if args.instances_per_primitive is not None:
        sampling["instances_per_primitive"] = int(args.instances_per_primitive)
    if args.split is not None:
        sampling["split"] = str(args.split)
    if args.min_chord_size is not None:
        sampling["min_chord_size"] = int(args.min_chord_size)
    if args.max_chord_size is not None:
        sampling["max_chord_size"] = int(args.max_chord_size)
    if args.min_duration_steps is not None:
        sampling["min_duration_steps"] = int(args.min_duration_steps)
    if args.max_duration_steps is not None:
        sampling["max_duration_steps"] = int(args.max_duration_steps)
    if args.primitive_ids is not None:
        sampling["primitive_ids"] = _normalize_primitive_ids(args.primitive_ids)
    if args.use_goals is not None:
        events["use_goals"] = bool(args.use_goals)
    if args.use_piano_states is not None:
        events["use_piano_states"] = bool(args.use_piano_states)
    if args.rollout_source_mode is not None:
        rollout["source_mode"] = str(args.rollout_source_mode)
    if args.example_midi_paths is not None:
        rollout["example_midi_paths"] = [str(_resolve_user_path(path)) for path in args.example_midi_paths]
    config["sampling"] = sampling
    config["events"] = events
    config["rollout"] = rollout

    payload = evaluate_primitives_online(config=config, logger=logger)
    if payload.get("status") == "backend_unavailable":
        logger.error("%s", payload.get("error", "RoboPianist runtime was unavailable."))
        raise SystemExit(1)
    logger.info("Primitive online evaluation complete: %s", payload["output_root"])


def _normalize_primitive_ids(values: list[str]) -> list[str]:
    output: list[str] = []
    for value in values:
        parts = [item.strip() for item in str(value).split(",")]
        output.extend(item for item in parts if item)
    return output


def _resolve_user_path(value: str | Path) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return resolve_path(path, PROJECT_ROOT)


if __name__ == "__main__":
    main()
