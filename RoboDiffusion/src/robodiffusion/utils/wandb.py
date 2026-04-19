from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

DEFAULT_WANDB_ENTITY = "tnguyen31-santa-clara-university"
DEFAULT_WANDB_PROJECT = "robopianist"


def _json_ready(payload: Any) -> Any:
    return json.loads(json.dumps(payload, default=str))


def _artifact_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-") or "artifact"


class WandbRun:
    def __init__(
        self,
        config: dict[str, Any] | None,
        *,
        run_name: str,
        config_payload: dict[str, Any],
        logger: logging.Logger | None = None,
        job_type: str | None = None,
        tags: list[str] | None = None,
    ) -> None:
        raw = dict(config or {})
        raw_tags = raw.get("tags", [])
        if isinstance(raw_tags, str):
            raw_tags = [tag.strip() for tag in raw_tags.split(",") if tag.strip()]
        raw_entity = raw.get("entity", DEFAULT_WANDB_ENTITY)
        raw_group = raw.get("group")
        raw_notes = raw.get("notes")
        raw_mode = raw.get("mode", "online")
        mode = str(raw_mode).strip() if raw_mode is not None else "online"
        if not mode:
            mode = "online"
        self.config = {
            "enabled": bool(raw.get("enabled", True)),
            "project": str(raw.get("project", DEFAULT_WANDB_PROJECT)),
            "entity": str(raw_entity).strip() if raw_entity is not None else None,
            "mode": mode,
            "group": str(raw_group).strip() if raw_group is not None and str(raw_group).strip() else None,
            "notes": str(raw_notes).strip() if raw_notes is not None and str(raw_notes).strip() else None,
            "tags": [str(tag).strip() for tag in raw_tags if str(tag).strip()],
            "dir": raw.get("dir"),
        }
        if tags:
            merged_tags = self.config["tags"] + [tag for tag in tags if tag not in self.config["tags"]]
            self.config["tags"] = merged_tags
        self.enabled = bool(self.config["enabled"]) and self.config["mode"] != "disabled"
        self.run = None
        self._wandb = None
        if not self.enabled:
            return
        try:
            import wandb
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "W&B logging is enabled for RoboDiffusion, but the `wandb` package is not installed. "
                "Install it with `pip install wandb`."
            ) from exc
        init_kwargs = {
            "project": self.config["project"],
            "entity": self.config["entity"],
            "group": self.config["group"],
            "tags": self.config["tags"],
            "notes": self.config["notes"],
            "mode": self.config["mode"],
            "name": run_name,
            "config": _json_ready(config_payload),
            "job_type": job_type,
            "reinit": True,
        }
        if self.config["dir"]:
            init_kwargs["dir"] = str(Path(self.config["dir"]).resolve())
        try:
            self.run = wandb.init(**init_kwargs)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Failed to initialize W&B for RoboDiffusion. Run `wandb login` or set `WANDB_API_KEY`, "
                "or disable logging with `wandb.enabled: false`."
            ) from exc
        self._wandb = wandb
        if logger is not None and getattr(self.run, "url", None):
            logger.info("W&B run: %s", self.run.url)

    def log(self, payload: dict[str, Any], step: int | None = None) -> None:
        if self.run is None:
            return
        self.run.log(_json_ready(payload), step=step)

    def summary(self, payload: dict[str, Any]) -> None:
        if self.run is None:
            return
        for key, value in _json_ready(payload).items():
            self.run.summary[key] = value

    def log_artifact_bundle(
        self,
        *,
        artifact_name: str,
        artifact_type: str,
        entries: dict[str, str | Path],
        aliases: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self.run is None or self._wandb is None:
            return
        artifact = self._wandb.Artifact(
            name=_artifact_name(artifact_name),
            type=artifact_type,
            metadata=_json_ready(metadata or {}),
        )
        added = False
        for name, path_like in entries.items():
            path = Path(path_like).resolve()
            if not path.exists():
                continue
            if path.is_dir():
                artifact.add_dir(local_path=str(path), name=name)
            else:
                artifact.add_file(local_path=str(path), name=name)
            added = True
        if added:
            self.run.log_artifact(artifact, aliases=list(aliases or []))

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
