"""Online rollout: dataset actions vs Stage 1 GMR priors (same harness as Stage 2 rollout)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sonata.evaluation.stage1_rollout import evaluate_stage1_rollout
from sonata.utils.logging import configure_logging
from sonata.utils.wandb import add_wandb_arguments, apply_wandb_cli_overrides
from sonata.utils.wandb_eval import finish_eval_wandb_run, log_prefixed_metrics, safe_init_eval_wandb_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate oracle dataset actions or oracle GMR primitives in DM Control.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--primitive-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--mode",
        choices=["oracle_dataset_actions", "oracle_gmr_primitives", "zero_actions"],
        default="oracle_dataset_actions",
    )
    parser.add_argument("--backend", choices=["offline", "dm_control"], default="offline")
    parser.add_argument("--split", default="val")
    parser.add_argument("--episode-id", default=None, help="Restrict rollout to this episode_id (exact match).")
    parser.add_argument("--limit-episodes", type=int, default=None)
    parser.add_argument("--render-video", action="store_true")
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--video-height", type=int, default=480)
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--max-render-episodes", type=int, default=None)
    parser.add_argument("--environment-name", default="RoboPianist-debug-TwinkleTwinkleLittleStar-v0")
    parser.add_argument("--midi-root", default=None)
    parser.add_argument("--control-timestep", type=float, default=0.05)
    parser.add_argument(
        "--audio-source",
        choices=["none", "reference_midi", "robot_midi"],
        default="none",
        help="MP4 audio track: none | score MIDI file (FluidSynth) | simulated piano MidiModule events.",
    )
    parser.add_argument("--debug-overlay", action="store_true", help="Overlay step/reward/key stats on video frames.")
    parser.add_argument("--prefer-midi-manifest", dest="prefer_midi_manifest", action="store_true", default=True)
    parser.add_argument("--no-prefer-midi-manifest", dest="prefer_midi_manifest", action="store_false")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--log-level", default="INFO")
    add_wandb_arguments(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logger = configure_logging(args.log_level)
    primitive_root = Path(args.primitive_root).resolve()
    output_root = Path(args.output_root).resolve()
    midi_root = Path(args.midi_root).resolve() if args.midi_root else None
    output_root.mkdir(parents=True, exist_ok=True)

    wandb_config = apply_wandb_cli_overrides({"wandb": {"enabled": False}}, args)["wandb"]
    wandb_run = safe_init_eval_wandb_run(
        config=wandb_config,
        run_name=args.wandb_run_name or f"sonata-stage1-rollout-{args.backend}-{args.mode}",
        config_payload={
            "primitive_root": str(primitive_root),
            "output_root": str(output_root),
            "mode": args.mode,
            "backend": args.backend,
            "split": args.split,
            "episode_id": args.episode_id,
            "limit_episodes": args.limit_episodes,
            "render_video": bool(args.render_video),
            "video_fps": int(args.video_fps),
            "video_height": int(args.video_height),
            "video_width": int(args.video_width),
            "max_render_episodes": args.max_render_episodes,
            "environment_name": args.environment_name,
            "midi_root": str(midi_root) if midi_root is not None else None,
            "prefer_midi_manifest": bool(args.prefer_midi_manifest),
            "control_timestep": float(args.control_timestep),
            "video_audio_source": args.audio_source,
            "debug_overlay": bool(args.debug_overlay),
        },
        logger=logger,
        group=args.wandb_group,
        tags=["evaluation", "stage1_rollout", args.mode, args.backend],
    )
    try:
        result = evaluate_stage1_rollout(
            primitive_root=primitive_root,
            output_root=output_root,
            mode=args.mode,
            backend=args.backend,
            split=args.split,
            episode_id=args.episode_id,
            limit_episodes=args.limit_episodes,
            render_video=args.render_video,
            video_fps=args.video_fps,
            video_height=args.video_height,
            video_width=args.video_width,
            max_render_episodes=args.max_render_episodes,
            environment_name=args.environment_name,
            midi_root=midi_root,
            prefer_midi_manifest=bool(args.prefer_midi_manifest),
            control_timestep=float(args.control_timestep),
            video_audio_source=args.audio_source,
            debug_overlay=bool(args.debug_overlay),
            logger=logger,
            wandb_run=wandb_run,
        )
        logger.info("Stage 1 rollout summary: %s", result["summary"])
        log_prefixed_metrics(wandb_run, result["metrics"], prefix="stage1_rollout", summary=True)
    finally:
        finish_eval_wandb_run(wandb_run)


if __name__ == "__main__":
    main()
