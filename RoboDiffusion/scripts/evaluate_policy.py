from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from torch.utils.data import DataLoader

from robodiffusion.config import load_stage_config, resolve_path
from robodiffusion.data.windows import CachedWindowDataset, collate_window_batch
from robodiffusion.evaluation.rollout import evaluate_policy_rollout, load_rollout_sources
from robodiffusion.model.policy import RoboDiffusionPolicy
from robodiffusion.training.trainer import diffusion_epoch
from robodiffusion.utils.checkpointing import find_latest_checkpoint
from robodiffusion.utils.io import write_json
from robodiffusion.utils.wandb import WandbRun


def main() -> None:
    profile = sys.argv[1] if len(sys.argv) > 1 else "debug"
    config = load_stage_config("evaluation", profile=profile)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger("robodiffusion.evaluate")

    checkpoint_path = resolve_path(config.get("checkpoint_path"), base_dir=ROOT)
    if checkpoint_path is None:
        raise ValueError("evaluation.checkpoint_path is required")
    if checkpoint_path.is_dir():
        latest = find_latest_checkpoint(checkpoint_path / "checkpoints")
        if latest is None:
            latest = find_latest_checkpoint(checkpoint_path)
        if latest is None:
            raise FileNotFoundError(f"No checkpoint found in {checkpoint_path}")
        checkpoint_path = latest

    output_root = resolve_path(config["output_root"], base_dir=ROOT)
    if output_root is None:
        raise ValueError("evaluation.output_root is required")
    output_root.mkdir(parents=True, exist_ok=True)

    wandb_run = WandbRun(
        config.get("wandb"),
        run_name=f"RoboDiffusion-eval-{output_root.name}",
        config_payload=config,
        logger=logger,
        job_type="evaluation",
        tags=["robodiffusion", "evaluation"],
    )
    try:
        payload: dict[str, object] = {"checkpoint_path": str(checkpoint_path)}
        wandb_run.summary({"checkpoint_path": str(checkpoint_path), "output_root": str(output_root)})
        if bool(config.get("run_offline", True)):
            dataset_root = resolve_path(config.get("dataset_root"), base_dir=ROOT)
            if dataset_root is None:
                raise ValueError("evaluation.dataset_root is required when run_offline is true")
            split = str(config.get("split", "val"))
            policy = RoboDiffusionPolicy.from_checkpoint(checkpoint_path)
            dataset = CachedWindowDataset(dataset_root, split=split)
            loader = DataLoader(dataset, batch_size=int(config.get("batch_size", 8)), shuffle=False, collate_fn=collate_window_batch)
            metrics, _ = diffusion_epoch(policy.model, policy.scheduler, loader, policy.device, train=False, optimizer=None, config=config)
            payload["offline_metrics"] = metrics
            logger.info("Offline metrics: %s", metrics)
            wandb_run.log({f"eval/offline/{key}": value for key, value in metrics.items()})
            wandb_run.summary({f"eval/offline/{key}": value for key, value in metrics.items()})
        if bool(config.get("run_rollout", False)):
            if config.get("data_manifest_path"):
                config["data_manifest_path"] = str(resolve_path(config["data_manifest_path"], base_dir=ROOT))
            sources = load_rollout_sources(config)
            rollout = evaluate_policy_rollout(
                checkpoint_path=checkpoint_path,
                output_root=output_root / "rollout",
                sources=sources,
                max_steps=int(config.get("max_steps", 512)),
                logger=logger,
            )
            payload["rollout"] = rollout
            summary = rollout.get("summary", {})
            wandb_run.log({f"eval/rollout/{key}": value for key, value in summary.items()})
            wandb_run.summary({f"eval/rollout/{key}": value for key, value in summary.items()})
        summary_path = write_json(payload, output_root / "evaluation_summary.json")
        wandb_run.log_artifact_bundle(
            artifact_name=f"{output_root.name}-evaluation",
            artifact_type="evaluation-output",
            entries={"evaluation": output_root},
            aliases=["latest"],
            metadata={"checkpoint_path": str(checkpoint_path)},
        )
        print(summary_path)
    finally:
        wandb_run.finish()


if __name__ == "__main__":
    main()
