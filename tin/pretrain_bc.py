from __future__ import annotations

import argparse
import logging

try:
    from tin.bc import BCPolicy, RP1MIterableDataset, infer_obs_dim, train_bc
except ImportError:  # pragma: no cover
    from bc import BCPolicy, RP1MIterableDataset, infer_obs_dim, train_bc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pretrain a BC actor on RP1M for Tin warmstarts.")
    parser.add_argument("--zarr-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-steps-lookahead", type=int, default=10)
    parser.add_argument("--action-dim", type=int, default=45)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-songs", type=int, default=None)
    parser.add_argument("--max-traj-per-song", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    dataset = RP1MIterableDataset(
        args.zarr_root,
        n_steps_lookahead=args.n_steps_lookahead,
        action_dim=args.action_dim,
        max_songs=args.max_songs,
        max_traj_per_song=args.max_traj_per_song,
    )
    model = BCPolicy(
        obs_dim=infer_obs_dim(args.n_steps_lookahead),
        act_dim=args.action_dim,
        hidden_dim=args.hidden_dim,
    )
    history = train_bc(
        model=model,
        dataset=dataset,
        save_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_amp=not args.disable_amp,
        device=args.device,
    )
    logging.getLogger(__name__).info("BC pretraining complete: %s", history)


if __name__ == "__main__":
    main()
