"""Plot task-wise, episode-wise, and chunk-horizon D diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def read_csv(path: Path, model_name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["model_name"] = model_name
    return frame


def save_task_delta(a_dir: Path, abc_dir: Path, output_dir: Path, top_k: int) -> Path:
    a = read_csv(a_dir / "task_metrics.csv", "ACT-A-only")
    abc = read_csv(abc_dir / "task_metrics.csv", "ACT-ABC")
    merged = a.merge(
        abc,
        on=["task_index", "task"],
        suffixes=("_a_only", "_abc"),
    )
    merged["delta_a_minus_abc"] = merged["action_l1_loss_a_only"] - merged["action_l1_loss_abc"]
    merged.to_csv(output_dir / "task_wise_delta.csv", index=False)

    plot_frame = pd.concat(
        [
            merged.nlargest(top_k, "delta_a_minus_abc").assign(group="ABC lower error"),
            merged.nsmallest(top_k, "delta_a_minus_abc").assign(group="ABC higher error"),
        ],
        ignore_index=True,
    )
    labels = [
        f"{row.task_index}: {str(row.task).split(':', 1)[0]}"
        for row in plot_frame.itertuples(index=False)
    ]
    colors = ["#2C7BB6" if value >= 0 else "#D7191C" for value in plot_frame["delta_a_minus_abc"]]
    plt.figure(figsize=(10, max(5, 0.32 * len(plot_frame))), dpi=200)
    plt.barh(labels, plot_frame["delta_a_minus_abc"], color=colors)
    plt.axvline(0.0, color="black", linewidth=1)
    plt.xlabel("Action L1 Delta (A-only minus ABC)")
    plt.title("Task-wise D Action Error Difference")
    plt.tight_layout()
    path = output_dir / "task_wise_action_l1_delta_on_D.png"
    plt.savefig(path)
    plt.close()
    return path


def save_episode_boxplot(a_dir: Path, abc_dir: Path, output_dir: Path) -> Path:
    frame = pd.concat(
        [
            read_csv(a_dir / "episode_metrics.csv", "ACT-A-only"),
            read_csv(abc_dir / "episode_metrics.csv", "ACT-ABC"),
        ],
        ignore_index=True,
    )
    data = [
        frame.loc[frame["model_name"] == "ACT-A-only", "action_l1_loss"].to_numpy(),
        frame.loc[frame["model_name"] == "ACT-ABC", "action_l1_loss"].to_numpy(),
    ]
    plt.figure(figsize=(6, 4.5), dpi=200)
    plt.boxplot(data, labels=["ACT-A-only", "ACT-ABC"], showfliers=False)
    plt.ylabel("Episode Mean Action L1")
    plt.title("Episode-wise D Action Error Distribution")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    path = output_dir / "episode_wise_action_l1_boxplot_on_D.png"
    plt.savefig(path)
    plt.close()
    return path


def save_chunk_horizon(a_dir: Path, abc_dir: Path, output_dir: Path) -> Path:
    frame = pd.concat(
        [
            read_csv(a_dir / "chunk_horizon_metrics.csv", "ACT-A-only"),
            read_csv(abc_dir / "chunk_horizon_metrics.csv", "ACT-ABC"),
        ],
        ignore_index=True,
    )
    plt.figure(figsize=(9, 4.8), dpi=200)
    for model_name, group in frame.groupby("model_name"):
        plt.plot(group["chunk_step"], group["mean_abs_error"], label=model_name, linewidth=2)
    plt.xlabel("Action Chunk Step")
    plt.ylabel("Mean Absolute Error")
    plt.title("D Action Error Across ACT Chunk Horizon")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    path = output_dir / "action_error_by_chunk_step_on_D.png"
    plt.savefig(path)
    plt.close()
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-dir", type=Path, default=Path("outputs/eval/A_only_on_D"))
    parser.add_argument("--abc-dir", type=Path, default=Path("outputs/eval/ABC_on_D"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures"))
    parser.add_argument("--top-k", type=int, default=15)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "task_wise_delta": save_task_delta(args.a_dir, args.abc_dir, args.output_dir, args.top_k),
        "episode_boxplot": save_episode_boxplot(args.a_dir, args.abc_dir, args.output_dir),
        "chunk_horizon": save_chunk_horizon(args.a_dir, args.abc_dir, args.output_dir),
    }
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
