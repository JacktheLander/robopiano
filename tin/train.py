from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import random
import time

import numpy as np
import tyro
import wandb

try:
    from tin.online_rl import TrainArgs as Args
    from tin.online_rl import get_env, initialize_agent_and_replay, prefix_dict, train_online
except ImportError:  # pragma: no cover
    from online_rl import TrainArgs as Args
    from online_rl import get_env, initialize_agent_and_replay, prefix_dict, train_online


def main(args: Args) -> None:
    if args.name:
        run_name = args.name
    else:
        run_name = f"SAC-{args.environment_name}-{args.seed}-{time.time()}"

    experiment_dir = Path(args.root_dir) / run_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)

    wandb.init(
        project=args.project,
        entity=args.entity or None,
        tags=(args.tags.split(",") if args.tags else []),
        notes=args.notes or None,
        config=asdict(args),
        mode=args.mode,
        name=run_name,
    )

    env = get_env(args)
    eval_env = get_env(args, record_dir=experiment_dir / "eval", enable_midi_metrics=True)
    spec, agent, replay_buffer, device_info = initialize_agent_and_replay(args, env)
    wandb.log(prefix_dict("runtime", device_info), step=0)

    def on_episode_end(step: int, statistics: dict[str, float]) -> None:
        wandb.log(prefix_dict("train", statistics), step=step)

    def on_train_metrics(step: int, metrics: dict[str, float]) -> None:
        wandb.log(prefix_dict("train", metrics), step=step)

    def on_eval(step: int, payload: dict[str, object]) -> None:
        statistics = dict(payload.get("statistics", {}))
        musical = dict(payload.get("music", {}))
        if statistics or musical:
            wandb.log(prefix_dict("eval", statistics) | prefix_dict("eval", musical), step=step)
        latest_filename = payload.get("latest_filename")
        if latest_filename is not None:
            video_path = Path(str(latest_filename))
            if video_path.exists():
                video = wandb.Video(str(video_path), fps=4, format="mp4")
                wandb.log({"video": video, "global_step": step})
                video_path.unlink()

    def on_fps(step: int, fps: int) -> None:
        wandb.log({"train/fps": fps}, step=step)

    _, training_summary = train_online(
        args=args,
        env=env,
        spec=spec,
        agent=agent,
        replay_buffer=replay_buffer,
        eval_env=eval_env,
        on_episode_end=on_episode_end,
        on_train_metrics=on_train_metrics,
        on_eval=on_eval,
        on_fps=on_fps,
    )
    wandb.log(prefix_dict("train_summary", training_summary), step=args.max_steps)


if __name__ == "__main__":
    main(tyro.cli(Args, description=__doc__))
