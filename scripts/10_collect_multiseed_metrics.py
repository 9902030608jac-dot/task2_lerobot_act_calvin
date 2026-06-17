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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45])
    parser.add_argument("--output-dir", type=Path, default=Path("report_assets/result_tables"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
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
    print(
        {
            "per_seed": str(args.output_dir / "multiseed_per_seed_metrics.csv"),
            "summary": str(args.output_dir / "multiseed_summary_metrics.csv"),
            "summary_md": str(args.output_dir / "multiseed_summary_metrics.md"),
        }
    )


if __name__ == "__main__":
    main()
