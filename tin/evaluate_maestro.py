from __future__ import annotations

import argparse
import logging
from pathlib import Path

try:
    from tin.maestro_eval import DEFAULT_DATASET_ROOT, DEFAULT_OUTPUT_ROOT, MaestroEvalConfig, evaluate_maestro_corpus
except ImportError:  # pragma: no cover
    from maestro_eval import DEFAULT_DATASET_ROOT, DEFAULT_OUTPUT_ROOT, MaestroEvalConfig, evaluate_maestro_corpus


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train TIN online from scratch on each MAESTRO MIDI file and build an F1 histogram.",
    )
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--max-steps-per-song", type=int, required=True)
    parser.add_argument("--final-eval-episodes", type=int, default=1)
    parser.add_argument("--limit-songs", type=int, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmstart-steps", type=int, default=5_000)
    parser.add_argument("--log-interval", type=int, default=1_000)
    parser.add_argument("--eval-interval", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--replay-capacity", type=int, default=1_000_000)
    parser.add_argument("--tqdm-bar", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--environment-name", default="RoboPianist-debug-TwinkleTwinkleRousseau-v0")
    parser.add_argument("--n-steps-lookahead", type=int, default=10)
    parser.add_argument("--trim-silence", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gravity-compensation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--reduced-action-space", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--control-timestep", type=float, default=0.05)
    parser.add_argument("--stretch-factor", type=float, default=1.0)
    parser.add_argument("--shift-factor", type=int, default=0)
    parser.add_argument("--wrong-press-termination", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--disable-fingering-reward", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--disable-forearm-reward", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--disable-colorization", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--disable-hand-collisions", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--disable-key-proximity-reward", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--disable-smooth-motion-reward", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--disable-anticipation-reward", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--primitive-fingertip-collisions", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--frame-stack", type=int, default=1)
    parser.add_argument("--clip", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--action-reward-observation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--agent-backend", choices=("auto", "sac", "droq"), default="auto")
    parser.add_argument("--utd-ratio", type=int, default=20)
    parser.add_argument("--n-step-return", type=int, default=3)
    parser.add_argument("--droq-hidden-dim", type=int, default=256)
    parser.add_argument("--droq-dropout", type=float, default=0.01)
    parser.add_argument("--droq-tau", type=float, default=0.005)
    parser.add_argument("--droq-lr", type=float, default=3e-4)
    parser.add_argument("--droq-min-alpha", type=float, default=0.05)
    parser.add_argument("--droq-grad-clip", type=float, default=1.0)
    parser.add_argument("--normalize-observations", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--normalize-rewards", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--normalizer-warmup-steps", type=int, default=50)
    parser.add_argument("--observation-normalizer-clip", type=float, default=5.0)
    parser.add_argument("--reward-normalizer-clip", type=float, default=10.0)
    parser.add_argument("--compile-models", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-intermediate-evals", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = MaestroEvalConfig(
        dataset_root=Path(args.dataset_root),
        output_root=Path(args.output_root),
        max_steps_per_song=args.max_steps_per_song,
        final_eval_episodes=args.final_eval_episodes,
        limit_songs=args.limit_songs,
        resume=args.resume,
        log_level=args.log_level,
        seed=args.seed,
        warmstart_steps=args.warmstart_steps,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        batch_size=args.batch_size,
        discount=args.discount,
        tqdm_bar=args.tqdm_bar,
        replay_capacity=args.replay_capacity,
        environment_name=args.environment_name,
        n_steps_lookahead=args.n_steps_lookahead,
        trim_silence=args.trim_silence,
        gravity_compensation=args.gravity_compensation,
        reduced_action_space=args.reduced_action_space,
        control_timestep=args.control_timestep,
        stretch_factor=args.stretch_factor,
        shift_factor=args.shift_factor,
        wrong_press_termination=args.wrong_press_termination,
        disable_fingering_reward=args.disable_fingering_reward,
        disable_forearm_reward=args.disable_forearm_reward,
        disable_colorization=args.disable_colorization,
        disable_hand_collisions=args.disable_hand_collisions,
        disable_key_proximity_reward=args.disable_key_proximity_reward,
        disable_smooth_motion_reward=args.disable_smooth_motion_reward,
        disable_anticipation_reward=args.disable_anticipation_reward,
        primitive_fingertip_collisions=args.primitive_fingertip_collisions,
        frame_stack=args.frame_stack,
        clip=args.clip,
        action_reward_observation=args.action_reward_observation,
        device=args.device,
        agent_backend=args.agent_backend,
        utd_ratio=args.utd_ratio,
        n_step_return=args.n_step_return,
        droq_hidden_dim=args.droq_hidden_dim,
        droq_dropout=args.droq_dropout,
        droq_tau=args.droq_tau,
        droq_lr=args.droq_lr,
        droq_min_alpha=args.droq_min_alpha,
        droq_grad_clip=args.droq_grad_clip,
        normalize_observations=args.normalize_observations,
        normalize_rewards=args.normalize_rewards,
        normalizer_warmup_steps=args.normalizer_warmup_steps,
        observation_normalizer_clip=args.observation_normalizer_clip,
        reward_normalizer_clip=args.reward_normalizer_clip,
        compile_models=args.compile_models,
        run_intermediate_evals=args.run_intermediate_evals,
    )
    payload = evaluate_maestro_corpus(config)
    logging.getLogger(__name__).info("MAESTRO evaluation complete: %s", payload)


if __name__ == "__main__":
    main()
