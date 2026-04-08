"""Evaluate Sonata offline metrics and optional DM Control playback.

Example:
python scripts/evaluate.py \
  --primitive-root /path/to/outputs/primitives/medium \
  --diffusion-checkpoint /path/to/checkpoints/best.pt \
  --output-root /path/to/eval_out \
  --backend dm_control \
  --device cuda \
  --render-video \
  --video-fps 20 \
  --wandb \
  --wandb-project robopianist \
  --wandb-entity <entity> \
  --wandb-run-name sonata3-dmcontrol-playback
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sonata.evaluation.offline import evaluate_offline_pipeline
from sonata.evaluation.rollout import evaluate_dm_control_rollout, evaluate_mjx_physics
from sonata.utils.logging import configure_logging
from sonata.utils.wandb import add_wandb_arguments, apply_wandb_cli_overrides
from sonata.utils.wandb_eval import finish_eval_wandb_run, log_prefixed_metrics, safe_init_eval_wandb_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Sonata-3 offline metrics and optional online playback backends.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--primitive-root", required=True)
    parser.add_argument("--diffusion-checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--backend", choices=["offline", "dm_control", "mjx_physics"], default="offline")
    parser.add_argument("--xml-path", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--render-video", action="store_true", help="Render DM Control rollout videos during evaluation.")
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--video-height", type=int, default=480)
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--max-render-episodes", type=int, default=None)
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
        run_name=args.wandb_run_name or f"sonata-eval-{args.backend}-{diffusion_checkpoint.stem}",
        config_payload={
            "primitive_root": str(primitive_root),
            "diffusion_checkpoint": str(diffusion_checkpoint),
            "output_root": str(output_root),
            "backend": args.backend,
            "variant": args.variant,
            "device": args.device,
            "render_video": bool(args.render_video),
            "video_fps": int(args.video_fps),
            "video_height": int(args.video_height),
            "video_width": int(args.video_width),
            "max_render_episodes": args.max_render_episodes,
        },
        logger=logger,
        group=args.wandb_group,
        tags=["evaluation", args.backend],
    )
    offline = evaluate_offline_pipeline(
        primitive_root=primitive_root,
        diffusion_checkpoint=diffusion_checkpoint,
        output_root=output_root / "offline",
        variant=args.variant,
        device=args.device,
    )
    logger.info("Offline metrics: %s", offline["metrics"])
    log_prefixed_metrics(wandb_run, offline["metrics"], prefix="offline", summary=True)
    try:
        if args.backend == "dm_control":
            rollout = evaluate_dm_control_rollout(
                primitive_root=primitive_root,
                predictions_by_episode=offline["predictions_by_episode"],
                output_root=output_root / "rollout",
                render_video=args.render_video,
                video_fps=args.video_fps,
                video_height=args.video_height,
                video_width=args.video_width,
                max_render_episodes=args.max_render_episodes,
                wandb_run=wandb_run,
                logger=logger,
            )
            logger.info("DM Control rollout: %s", rollout)
        elif args.backend == "mjx_physics":
            if args.xml_path is None:
                raise ValueError("--xml-path is required for mjx_physics backend")
            first_episode = next(iter(offline["predictions_by_episode"].values()))
            import numpy as np

            actions = np.stack([item["predicted"] for item in first_episode], axis=0)
            rollout = evaluate_mjx_physics(
                xml_path=Path(args.xml_path),
                action_sequences=actions,
                output_root=output_root / "rollout",
            )
            logger.info("MJX rollout: %s", rollout)
            log_prefixed_metrics(wandb_run, rollout, prefix="rollout/mjx_physics", summary=True)
        else:
            log_prefixed_metrics(wandb_run, {"backend": args.backend, "status": "offline_only"}, prefix="evaluation", summary=True)
        log_prefixed_metrics(wandb_run, {"backend": args.backend, "status": "completed"}, prefix="evaluation", summary=True)
    finally:
        finish_eval_wandb_run(wandb_run)


if __name__ == "__main__":
    main()
