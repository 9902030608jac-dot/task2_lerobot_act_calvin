"""Small optional WandB integration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.paths import resolve_path


def should_use_wandb(config: dict[str, Any]) -> bool:
    """Return whether WandB logging is enabled for this run."""
    return bool(config.get("use_wandb", False))


def init_wandb(config: dict[str, Any], *, job_type: str) -> Any | None:
    """Initialize WandB when enabled, otherwise return None.

    WandB remains optional: if `use_wandb` is false nothing happens; if it is
    true but the package is missing, the caller receives a clear RuntimeError.
    """
    if not should_use_wandb(config):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "use_wandb=true but wandb is not installed. Install it with `pip install wandb`, "
            "or run with `--override use_wandb=false`."
        ) from exc

    try:
        return wandb.init(
            project=config["project_name"],
            name=config["run_name"],
            config=config,
            job_type=job_type,
            dir=str(resolve_path("wandb")),
            reinit=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "WandB initialization failed. Run `wandb login`, set WANDB_MODE=offline, "
            "or disable it with `--override use_wandb=false`."
        ) from exc


def wandb_log(run: Any | None, metrics: dict[str, Any], *, step: int | None = None) -> None:
    """Log metrics to WandB if a run exists."""
    if run is not None:
        run.log(metrics, step=step)


def wandb_log_artifact(run: Any | None, path: str | Path, *, name: str, artifact_type: str) -> None:
    """Log a file as a WandB artifact if a run exists."""
    if run is None:
        return
    import wandb

    resolved = resolve_path(path)
    if not resolved.exists():
        return
    artifact = wandb.Artifact(name=name, type=artifact_type)
    artifact.add_file(str(resolved))
    run.log_artifact(artifact)


def wandb_finish(run: Any | None) -> None:
    """Finish a WandB run if one exists."""
    if run is not None:
        run.finish()
