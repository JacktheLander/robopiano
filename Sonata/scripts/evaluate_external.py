"""Evaluate Sonata on unseen songs from an external MIDI benchmark manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sonata.evaluation.external_midi import DEFAULT_EXTERNAL_ENVIRONMENT, evaluate_external_midi_benchmark
from sonata.utils.logging import configure_logging
from sonata.utils.wandb import add_wandb_arguments, apply_wandb_cli_overrides
from sonata.utils.wandb_eval import finish_eval_wandb_run, safe_init_eval_wandb_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Sonata rollouts on a benchmark of unseen external MIDI songs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--primitive-root", required=True)
    parser.add_argument("--diffusion-checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--benchmark-manifest", default=None)
    parser.add_argument("--benchmark-root", default=None)
    parser.add_argument("--benchmark-split", default="test")
    parser.add_argument("--variant", default=None)
    parser.add_argument("--limit-episodes", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--environment-name", default=DEFAULT_EXTERNAL_ENVIRONMENT)
    parser.add_argument("--render-video", action="store_true")
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--video-height", type=int, default=480)
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--max-render-episodes", type=int, default=None)
    parser.add_argument("--causal-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--restore-mode", choices=["hands_only", "neutral", "unsafe_legacy"], default="hands_only")
    parser.add_argument("--video-audio-source", choices=["none", "robot_midi"], default="none")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--log-level", default="INFO")
    add_wandb_arguments(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logger = configure_logging(args.log_level)
    primitive_root = Path(args.primitive_root).resolve()
    diffusion_checkpoint = Path(args.diffusion_checkpoint).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    wandb_config = apply_wandb_cli_overrides({"wandb": {"enabled": False}}, args)["wandb"]
    wandb_run = safe_init_eval_wandb_run(
        config=wandb_config,
        run_name=args.wandb_run_name or f"sonata-external-midi-{diffusion_checkpoint.stem}",
        config_payload={
            "primitive_root": str(primitive_root),
            "diffusion_checkpoint": str(diffusion_checkpoint),
            "output_root": str(output_root),
            "benchmark_manifest": args.benchmark_manifest,
            "benchmark_root": args.benchmark_root,
            "benchmark_split": args.benchmark_split,
            "variant": args.variant,
            "device": args.device,
            "environment_name": args.environment_name,
            "render_video": bool(args.render_video),
            "video_fps": int(args.video_fps),
            "video_height": int(args.video_height),
            "video_width": int(args.video_width),
            "max_render_episodes": args.max_render_episodes,
        },
        logger=logger,
        group=args.wandb_group,
        tags=["evaluation", "external_midi"],
    )
    try:
        payload = evaluate_external_midi_benchmark(
            primitive_root=primitive_root,
            diffusion_checkpoint=diffusion_checkpoint,
            output_root=output_root,
            benchmark_manifest=args.benchmark_manifest,
            benchmark_root=args.benchmark_root,
            benchmark_split=args.benchmark_split,
            variant=args.variant,
            limit_episodes=args.limit_episodes,
            device=args.device,
            environment_name=args.environment_name,
            render_video=args.render_video,
            video_fps=args.video_fps,
            video_height=args.video_height,
            video_width=args.video_width,
            max_render_episodes=args.max_render_episodes,
            causal_eval={
                "enabled": bool(args.causal_eval and args.restore_mode != "unsafe_legacy"),
                "restore_mode": str(args.restore_mode),
                "video_audio_source": str(args.video_audio_source),
            },
            wandb_run=wandb_run,
            logger=logger,
        )
        logger.info("External MIDI rollout summary: %s", payload.get("summary", {}))
    finally:
        finish_eval_wandb_run(wandb_run)


if __name__ == "__main__":
    main()
