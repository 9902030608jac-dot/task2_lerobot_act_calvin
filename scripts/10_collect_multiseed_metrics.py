"""Collect multi-seed train/eval metrics into mean/std report tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def final_train_loss(train_dir: Path) -> float | None:
    path = train_dir / "train_metrics.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if frame.empty or "train_action_l1_loss" not in frame:
        return None
    return float(frame["train_action_l1_loss"].dropna().iloc[-1])


def eval_l1(eval_dir: Path) -> float | None:
    path = eval_dir / "metrics.json"
    if not path.exists():
        return None
    metrics = json.loads(path.read_text(encoding="utf-8"))
    value = metrics.get("action_l1_loss")
    return float(value) if value is not None else None


def collect_one(model: str, seeds: list[int]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in seeds:
        rows.append(
            {
                "model": model,
                "seed": seed,
                "train_final_action_l1_loss": final_train_loss(Path(f"outputs/train/{model}_seed{seed}")),
                "d_action_l1_loss": eval_l1(Path(f"outputs/eval/{model}_seed{seed}_on_D")),
            }
        )
    return rows


def read_metric_csv(eval_dir: Path, filename: str, model: str, seed: int) -> pd.DataFrame | None:
    path = eval_dir / filename
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    frame["model"] = model
    frame["seed"] = seed
    return frame


def collect_eval_breakdowns(models: list[str], seeds: list[int]) -> dict[str, pd.DataFrame]:
    """Collect per-seed task, episode, and chunk-horizon metrics when available."""
    task_frames: list[pd.DataFrame] = []
    episode_frames: list[pd.DataFrame] = []
    chunk_frames: list[pd.DataFrame] = []
    for model in models:
        for seed in seeds:
            eval_dir = Path(f"outputs/eval/{model}_seed{seed}_on_D")
            task = read_metric_csv(eval_dir, "task_metrics.csv", model, seed)
            episode = read_metric_csv(eval_dir, "episode_metrics.csv", model, seed)
            chunk = read_metric_csv(eval_dir, "chunk_horizon_metrics.csv", model, seed)
            if task is not None:
                task_frames.append(task)
            if episode is not None:
                episode_frames.append(episode)
            if chunk is not None:
                chunk_frames.append(chunk)
    return {
        "task": pd.concat(task_frames, ignore_index=True) if task_frames else pd.DataFrame(),
        "episode": pd.concat(episode_frames, ignore_index=True) if episode_frames else pd.DataFrame(),
        "chunk": pd.concat(chunk_frames, ignore_index=True) if chunk_frames else pd.DataFrame(),
    }


def summarize_task_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    return (
        frame.groupby(["model", "task_index", "task"], dropna=False)
        .agg(
            action_l1_mean=("action_l1_loss", "mean"),
            action_l1_std=("action_l1_loss", "std"),
            completed_seeds=("action_l1_loss", "count"),
            mean_num_samples=("num_samples", "mean"),
        )
        .reset_index()
    )


def summarize_chunk_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    return (
        frame.groupby(["model", "chunk_step"], dropna=False)
        .agg(
            mean_abs_error_mean=("mean_abs_error", "mean"),
            mean_abs_error_std=("mean_abs_error", "std"),
            completed_seeds=("mean_abs_error", "count"),
        )
        .reset_index()
    )


def summarize_episode_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    per_seed = (
        frame.groupby(["model", "seed"], dropna=False)["action_l1_loss"]
        .agg(
            episode_l1_mean="mean",
            episode_l1_std="std",
            episode_l1_median="median",
            episode_l1_p90=lambda value: value.quantile(0.90),
            num_episodes="count",
        )
        .reset_index()
    )
    summary = (
        per_seed.groupby("model", dropna=False)
        .agg(
            episode_l1_mean_across_seeds=("episode_l1_mean", "mean"),
            episode_l1_mean_std_across_seeds=("episode_l1_mean", "std"),
            episode_l1_median_across_seeds=("episode_l1_median", "mean"),
            episode_l1_p90_across_seeds=("episode_l1_p90", "mean"),
            completed_seeds=("episode_l1_mean", "count"),
        )
        .reset_index()
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45])
    parser.add_argument("--output-dir", type=Path, default=Path("report_assets/result_tables"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    models = ["act_A_only", "act_ABC"]
    rows = collect_one("act_A_only", args.seeds) + collect_one("act_ABC", args.seeds)
    per_seed = pd.DataFrame(rows)
    summary = (
        per_seed.groupby("model", dropna=False)
        .agg(
            train_final_action_l1_mean=("train_final_action_l1_loss", "mean"),
            train_final_action_l1_std=("train_final_action_l1_loss", "std"),
            d_action_l1_mean=("d_action_l1_loss", "mean"),
            d_action_l1_std=("d_action_l1_loss", "std"),
            completed_seeds=("d_action_l1_loss", "count"),
        )
        .reset_index()
    )
    per_seed.to_csv(args.output_dir / "multiseed_per_seed_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "multiseed_summary_metrics.csv", index=False)
    (args.output_dir / "multiseed_summary_metrics.md").write_text(
        "# Multi-seed Summary Metrics\n\n" + summary.to_markdown(index=False) + "\n",
        encoding="utf-8",
    )
    breakdowns = collect_eval_breakdowns(models, args.seeds)
    task_summary = summarize_task_metrics(breakdowns["task"])
    chunk_summary = summarize_chunk_metrics(breakdowns["chunk"])
    episode_summary = summarize_episode_metrics(breakdowns["episode"])
    task_summary.to_csv(args.output_dir / "multiseed_task_metrics.csv", index=False)
    chunk_summary.to_csv(args.output_dir / "multiseed_chunk_horizon_metrics.csv", index=False)
    episode_summary.to_csv(args.output_dir / "multiseed_episode_distribution_metrics.csv", index=False)
    if not task_summary.empty:
        (args.output_dir / "multiseed_task_metrics.md").write_text(
            "# Multi-seed Task-wise D Action L1\n\n" + task_summary.to_markdown(index=False) + "\n",
            encoding="utf-8",
        )
    if not episode_summary.empty:
        (args.output_dir / "multiseed_episode_distribution_metrics.md").write_text(
            "# Multi-seed Episode-wise D Action L1 Distribution\n\n"
            + episode_summary.to_markdown(index=False)
            + "\n",
            encoding="utf-8",
        )
    print(
        {
            "per_seed": str(args.output_dir / "multiseed_per_seed_metrics.csv"),
            "summary": str(args.output_dir / "multiseed_summary_metrics.csv"),
            "summary_md": str(args.output_dir / "multiseed_summary_metrics.md"),
            "task_summary": str(args.output_dir / "multiseed_task_metrics.csv"),
            "chunk_horizon_summary": str(args.output_dir / "multiseed_chunk_horizon_metrics.csv"),
            "episode_distribution_summary": str(args.output_dir / "multiseed_episode_distribution_metrics.csv"),
        }
    )


if __name__ == "__main__":
    main()
