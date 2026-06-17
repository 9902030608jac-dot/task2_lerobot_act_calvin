"""Plotting utilities for training and evaluation figures."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from src.paths import PROJECT_ROOT

_MPLCONFIGDIR = PROJECT_ROOT / "logs" / "matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib.pyplot as plt

from src.metrics import DEFAULT_RUNS, NA, RunSpec, read_eval_metrics, read_train_metrics, warn
from src.paths import ensure_dir, resolve_path
from src.config import load_config
from src.wandb_utils import init_wandb, wandb_finish, wandb_log_artifact


def save_current_figure(path: str | Path) -> Path:
    """Save the current Matplotlib figure as PNG."""
    resolved = resolve_path(path)
    ensure_dir(resolved.parent)
    plt.tight_layout()
    plt.savefig(resolved, dpi=200)
    plt.close()
    return resolved


def plot_train_action_l1_curve(
    runs: list[RunSpec] | None = None,
    output_path: str | Path = "outputs/figures/train_action_l1_curve.png",
) -> Path | None:
    """Plot train Action L1 loss curves for A-only and ABC."""
    runs = runs or DEFAULT_RUNS
    warnings: list[str] = []
    plotted = False
    plt.figure(figsize=(7, 4.5))
    for run in runs:
        frame = read_train_metrics(run.train_metrics_path, warnings)
        if frame is None:
            continue
        plt.plot(frame["step"], frame["train_action_l1_loss"], label=run.model_name)
        plotted = True
    if not plotted:
        plt.close()
        print("WARN: No training curves were plotted because train_metrics.csv files are missing.")
        return None
    plt.xlabel("Step")
    plt.ylabel("Train Action L1 Loss")
    plt.title("Training Action L1 Loss")
    plt.legend()
    plt.grid(alpha=0.25)
    return save_current_figure(output_path)


def plot_val_action_l1_curve(
    runs: list[RunSpec] | None = None,
    output_path: str | Path = "outputs/figures/val_action_l1_curve.png",
) -> Path | None:
    """Plot validation Action L1 loss curves for A-only and ABC."""
    runs = runs or DEFAULT_RUNS
    warnings: list[str] = []
    plotted = False
    plt.figure(figsize=(7, 4.5))
    for run in runs:
        frame = read_train_metrics(run.train_metrics_path, warnings)
        if frame is None or "val_action_l1_loss" not in frame.columns:
            continue
        frame = frame.copy()
        frame["val_action_l1_loss"] = pd.to_numeric(frame["val_action_l1_loss"], errors="coerce")
        frame = frame.dropna(subset=["val_action_l1_loss"])
        if frame.empty:
            continue
        plt.plot(frame["step"], frame["val_action_l1_loss"], marker="o", label=run.model_name)
        plotted = True
    if not plotted:
        plt.close()
        print("WARN: No validation curves were plotted because val_action_l1_loss is missing.")
        return None
    plt.xlabel("Step")
    plt.ylabel("Validation Action L1 Loss")
    plt.title("Validation Action L1 Loss")
    plt.legend()
    plt.grid(alpha=0.25)
    return save_current_figure(output_path)


def load_eval_frames(runs: list[RunSpec] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Load eval metrics into a dataframe."""
    runs = runs or DEFAULT_RUNS
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []
    for run in runs:
        metrics = read_eval_metrics(run.eval_metrics_path, warnings)
        rows.append(
            {
                "model_name": run.model_name,
                "action_l1_loss": metrics.get("action_l1_loss") if metrics else None,
                "action_error_mean": metrics.get("action_error_mean") if metrics else None,
                "action_error_by_dim": metrics.get("action_error_by_dim") if metrics else None,
                "success_rate": metrics.get("success_rate") if metrics else None,
            }
        )
    return pd.DataFrame(rows), warnings


def numeric_or_none(value: Any) -> float | None:
    """Convert a value to float when possible."""
    if value is None or value == NA:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def plot_eval_bar(
    column: str,
    ylabel: str,
    title: str,
    output_path: str | Path,
    runs: list[RunSpec] | None = None,
) -> Path | None:
    """Plot a two-model bar chart for one eval metric."""
    frame, _ = load_eval_frames(runs)
    values = [numeric_or_none(value) for value in frame[column].tolist()]
    if all(value is None for value in values):
        print(f"WARN: {column} is missing for all models; skipped {output_path}.")
        return None
    plot_values = [value if value is not None else 0.0 for value in values]
    labels = frame["model_name"].tolist()
    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, plot_values, color=["#4C78A8", "#F58518"])
    for bar, value in zip(bars, values):
        label = "NA" if value is None else f"{value:.4f}"
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label, ha="center", va="bottom")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.25)
    return save_current_figure(output_path)


def plot_action_error_by_dim(
    runs: list[RunSpec] | None = None,
    output_path: str | Path = "outputs/figures/action_error_by_dim_on_D.png",
) -> Path | None:
    """Plot action error by dimension for both models."""
    frame, _ = load_eval_frames(runs)
    rows = []
    for _, row in frame.iterrows():
        values = row["action_error_by_dim"]
        if not isinstance(values, list):
            continue
        for dim, value in enumerate(values):
            rows.append({"model_name": row["model_name"], "dim": dim, "error": float(value)})
    if not rows:
        print(f"WARN: action_error_by_dim is missing; skipped {output_path}.")
        return None

    data = pd.DataFrame(rows)
    pivot = data.pivot(index="dim", columns="model_name", values="error")
    ax = pivot.plot(kind="bar", figsize=(8, 4.5), width=0.8)
    ax.set_xlabel("Action Dimension")
    ax.set_ylabel("Mean Absolute Error")
    ax.set_title("Action Error by Dimension on D")
    ax.grid(axis="y", alpha=0.25)
    return save_current_figure(output_path)


def plot_success_rate_if_available(
    runs: list[RunSpec] | None = None,
    output_path: str | Path = "outputs/figures/success_rate_on_D.png",
) -> Path | None:
    """Plot success rate only if at least one model has a non-null value."""
    frame, _ = load_eval_frames(runs)
    values = [numeric_or_none(value) for value in frame["success_rate"].tolist()]
    if all(value is None for value in values):
        print("WARN: success_rate is Not Available for all models; skipped success_rate_on_D.png.")
        return None
    return plot_eval_bar(
        "success_rate",
        "Success Rate",
        "Zero-shot Success Rate on D",
        output_path,
        runs,
    )


def plot_all_report_figures() -> dict[str, str | None]:
    """Generate all requested report figures."""
    outputs = {
        "train_action_l1_curve": plot_train_action_l1_curve(),
        "val_action_l1_curve": plot_val_action_l1_curve(),
        "eval_action_l1_on_D": plot_eval_bar(
            "action_l1_loss",
            "Action L1 Loss",
            "Offline Action L1 Loss on D",
            "outputs/figures/eval_action_l1_on_D.png",
        ),
        "action_error_mean_on_D": plot_eval_bar(
            "action_error_mean",
            "Mean Action Error",
            "Mean Action Error on D",
            "outputs/figures/action_error_mean_on_D.png",
        ),
        "action_error_by_dim_on_D": plot_action_error_by_dim(),
        "success_rate_on_D": plot_success_rate_if_available(),
    }
    wandb_run = None
    wandb_url = None
    try:
        config = load_config("configs/train_A_only.yaml")
        config["run_name"] = "plot_report_figures"
        wandb_run = init_wandb(config, job_type="report")
        for key, value in outputs.items():
            if value is not None:
                wandb_log_artifact(
                    wandb_run,
                    value,
                    name=key,
                    artifact_type="report_figure",
                )
        wandb_url = getattr(wandb_run, "url", None) if wandb_run is not None else None
    except RuntimeError as exc:
        print(f"WARN: WandB figure upload skipped: {exc}")
    finally:
        wandb_finish(wandb_run)

    result = {key: str(value) if value is not None else None for key, value in outputs.items()}
    result["wandb_run_url"] = wandb_url
    return result
