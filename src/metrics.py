"""Metric collection helpers for report-ready tables and notes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.paths import ensure_dir, resolve_path
from src.config import load_config
from src.wandb_utils import init_wandb, wandb_finish, wandb_log_artifact


NA = "Not Available"


@dataclass(frozen=True)
class RunSpec:
    """Input paths and labels for one experiment."""

    model_name: str
    train_metrics_path: Path
    eval_metrics_path: Path


DEFAULT_RUNS = [
    RunSpec(
        model_name="ACT-A-only",
        train_metrics_path=resolve_path("outputs/train/act_A_only/train_metrics.csv"),
        eval_metrics_path=resolve_path("outputs/eval/A_only_on_D/metrics.json"),
    ),
    RunSpec(
        model_name="ACT-ABC",
        train_metrics_path=resolve_path("outputs/train/act_ABC/train_metrics.csv"),
        eval_metrics_path=resolve_path("outputs/eval/ABC_on_D/metrics.json"),
    ),
]


def warn(message: str, warnings: list[str]) -> None:
    """Record and print a warning without failing the run."""
    warnings.append(message)
    print(f"WARN: {message}")


def read_train_metrics(path: str | Path, warnings: list[str]) -> pd.DataFrame | None:
    """Read a train_metrics.csv file if it exists."""
    resolved = resolve_path(path)
    if not resolved.exists():
        warn(f"Missing train metrics: {resolved}", warnings)
        return None
    frame = pd.read_csv(resolved)
    if "step" not in frame.columns or "train_action_l1_loss" not in frame.columns:
        warn(f"Train metrics missing required columns step/train_action_l1_loss: {resolved}", warnings)
        return None
    return frame


def read_eval_metrics(path: str | Path, warnings: list[str]) -> dict[str, Any] | None:
    """Read an eval metrics.json file if it exists."""
    resolved = resolve_path(path)
    if not resolved.exists():
        warn(f"Missing eval metrics: {resolved}", warnings)
        return None
    with resolved.open("r", encoding="utf-8") as file:
        return json.load(file)


def maybe_number(value: Any) -> Any:
    """Return a compact numeric value or Not Available."""
    if value is None:
        return NA
    if isinstance(value, float):
        return round(value, 6)
    return value


def collect_summary_metrics(runs: list[RunSpec] | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Collect train/eval metrics into one summary table."""
    runs = runs or DEFAULT_RUNS
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []
    raw: dict[str, Any] = {"warnings": warnings, "train": {}, "eval": {}}

    for run in runs:
        train = read_train_metrics(run.train_metrics_path, warnings)
        eval_metrics = read_eval_metrics(run.eval_metrics_path, warnings)
        raw["train"][run.model_name] = train
        raw["eval"][run.model_name] = eval_metrics

        if train is not None and len(train) > 0:
            final_train_l1 = train["train_action_l1_loss"].iloc[-1]
            best_train_l1 = train["train_action_l1_loss"].min()
            final_step = int(train["step"].iloc[-1])
            val_series = (
                pd.to_numeric(train.get("val_action_l1_loss"), errors="coerce")
                if "val_action_l1_loss" in train.columns
                else pd.Series(dtype=float)
            )
            valid_val = val_series.dropna()
            final_val_l1 = valid_val.iloc[-1] if len(valid_val) else None
            best_val_l1 = valid_val.min() if len(valid_val) else None
        else:
            final_train_l1 = best_train_l1 = final_step = None
            final_val_l1 = best_val_l1 = None

        rows.append(
            {
                "model_name": run.model_name,
                "train_final_step": maybe_number(final_step),
                "train_final_action_l1_loss": maybe_number(final_train_l1),
                "train_best_action_l1_loss": maybe_number(best_train_l1),
                "val_final_action_l1_loss": maybe_number(final_val_l1),
                "val_best_action_l1_loss": maybe_number(best_val_l1),
                "test_env": maybe_number(eval_metrics.get("test_env") if eval_metrics else None),
                "eval_mode": maybe_number(eval_metrics.get("mode") if eval_metrics else None),
                "eval_num_samples": maybe_number(eval_metrics.get("num_samples") if eval_metrics else None),
                "eval_num_episodes": maybe_number(eval_metrics.get("num_episodes") if eval_metrics else None),
                "d_action_l1_loss": maybe_number(eval_metrics.get("action_l1_loss") if eval_metrics else None),
                "d_action_error_mean": maybe_number(eval_metrics.get("action_error_mean") if eval_metrics else None),
                "success_rate": maybe_number(eval_metrics.get("success_rate") if eval_metrics else None),
                "avg_episode_length": maybe_number(eval_metrics.get("avg_episode_length") if eval_metrics else None),
                "checkpoint_path": maybe_number(eval_metrics.get("checkpoint_path") if eval_metrics else None),
            }
        )

    return pd.DataFrame(rows), raw


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Render a dataframe as a simple Markdown table."""
    if frame.empty:
        return "| Metric | Value |\n| --- | --- |\n| No data | Not Available |"
    headers = list(frame.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in frame.iterrows():
        values = [str(row[column]).replace("|", "\\|") for column in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_summary_tables(
    summary: pd.DataFrame,
    warnings: list[str],
    csv_path: str | Path = "report_assets/result_tables/summary_metrics.csv",
    md_path: str | Path = "report_assets/result_tables/summary_metrics.md",
) -> tuple[Path, Path]:
    """Write summary CSV and Markdown files."""
    csv_resolved = resolve_path(csv_path)
    md_resolved = resolve_path(md_path)
    ensure_dir(csv_resolved.parent)
    ensure_dir(md_resolved.parent)
    summary.to_csv(csv_resolved, index=False)
    md_content = "\n".join(
        [
            "# Summary Metrics",
            "",
            dataframe_to_markdown(summary),
            "",
            "## Warnings",
            "",
            *(f"- {message}" for message in warnings),
            "" if warnings else "- None",
            "",
        ]
    )
    md_resolved.write_text(md_content, encoding="utf-8")
    return csv_resolved, md_resolved


def compare_metric(summary: pd.DataFrame, column: str) -> str:
    """Return a small comparison sentence for two model rows."""
    if summary.empty or len(summary) < 2:
        return "Not enough data is available for comparison."
    values = {}
    for _, row in summary.iterrows():
        value = row[column]
        if value == NA:
            return f"{column} is Not Available for at least one model."
        try:
            values[row["model_name"]] = float(value)
        except (TypeError, ValueError):
            return f"{column} is not numeric for at least one model."
    if values["ACT-ABC"] < values["ACT-A-only"]:
        return f"ACT-ABC is lower on {column}, which is favorable for error/loss metrics."
    if values["ACT-ABC"] > values["ACT-A-only"]:
        return f"ACT-A-only is lower on {column}, which is favorable for error/loss metrics."
    return f"Both models have the same {column}."


def write_analysis_notes(
    summary: pd.DataFrame,
    warnings: list[str],
    path: str | Path = "report_assets/result_analysis_notes.md",
) -> Path:
    """Generate an analysis outline for the final report."""
    resolved = resolve_path(path)
    ensure_dir(resolved.parent)
    success_available = (
        "success_rate" in summary.columns
        and any(value != NA for value in summary["success_rate"].tolist())
    )
    content = f"""# Result Analysis Notes

## 1. A-only 与 ABC 训练收敛对比

- 对比 `train_action_l1_curve.png` 中两条曲线的下降速度、最终 loss 和震荡情况。
- 重点观察 A-only 是否更快拟合单一环境 A，以及 ABC 是否因为数据更多样而收敛稍慢但更稳定。
- 当前训练数据可用性：{"有训练日志" if any(value != NA for value in summary.get("train_final_action_l1_loss", [])) else "训练日志 Not Available"}。
- 当前验证数据可用性：{"有验证曲线" if any(value != NA for value in summary.get("val_final_action_l1_loss", [])) else "验证指标 Not Available"}。

## 2. D 环境 Action L1 Loss 对比

- {compare_metric(summary, "d_action_l1_loss")}
- offline Action L1 Loss 只能说明模型在 D 离线轨迹上的动作预测误差，不能直接等价于闭环任务成功率。

## 3. Success Rate 是否可用

- Success Rate 状态：{"Available" if success_available else "Not Available"}。
- 如果 `success_rate` 为 Not Available 或 null，报告中应使用 Action Error / Action L1 Loss 作为主要 D 环境指标，并明确说明没有闭环 rollout 结果。

## 4. ABC 是否更好

- 主要依据 D 环境的 `action_l1_loss`、`action_error_mean`，以及可用时的 `success_rate`。
- {compare_metric(summary, "d_action_error_mean")}

## 5. 如果 ABC 更好，可能解释

- ABC 训练包含 A/B/C 多环境视觉和状态分布，可能学到较少依赖单一背景或布局的表征。
- 多环境数据可能降低对环境 A 的过拟合，从而在未见 D 上获得更低动作误差。

## 6. 如果 ABC 没更好，可能解释

- A/B/C 与 D 的视觉差距仍然过大，多环境训练没有覆盖关键视觉变化。
- 混合数据可能引入更高优化难度，模型容量或训练步数不足。
- 数据量、任务分布或 episode 长度在 A/B/C 之间可能不均衡。

## 7. Action Chunking 的正反两方面分析

- 正面：ACT 的 action chunking 一次预测一段动作，有助于提升动作连续性，减少逐步预测造成的抖动。
- 负面：action chunking 不能直接解决视觉分布偏移。如果 D 中图像表征已经偏离训练分布，模型可能连续执行一段错误动作。
- 因此，chunking 更像是时序平滑和短期一致性机制，而不是视觉泛化机制本身。

## 8. 不应过度声称的结论

- 不应声称 offline Action L1 Loss 等同于真实 Success Rate。
- 不应声称 ABC 在一个 D split 上更好就代表对所有未见环境都泛化。
- 不应声称 Action Chunking 解决了 Visual Distribution Shift。
- 不应忽略 seed、训练步数、数据规模和评估方式对结果的影响。

## Warnings

{chr(10).join(f"- {message}" for message in warnings) if warnings else "- None"}
"""
    resolved.write_text(content, encoding="utf-8")
    return resolved


def collect_and_write_report_assets() -> dict[str, Any]:
    """Collect metrics and write summary tables plus analysis notes."""
    summary, raw = collect_summary_metrics()
    csv_path, md_path = write_summary_tables(summary, raw["warnings"])
    notes_path = write_analysis_notes(summary, raw["warnings"])
    wandb_run = None
    wandb_url = None
    try:
        config = load_config("configs/train_A_only.yaml")
        config["run_name"] = "collect_report_metrics"
        wandb_run = init_wandb(config, job_type="report")
        wandb_log_artifact(wandb_run, csv_path, name="summary_metrics_csv", artifact_type="report_table")
        wandb_log_artifact(wandb_run, md_path, name="summary_metrics_md", artifact_type="report_table")
        wandb_log_artifact(wandb_run, notes_path, name="result_analysis_notes", artifact_type="report_notes")
        wandb_url = getattr(wandb_run, "url", None) if wandb_run is not None else None
    except RuntimeError as exc:
        warn(f"WandB report upload skipped: {exc}", raw["warnings"])
    finally:
        wandb_finish(wandb_run)
    return {
        "summary": summary,
        "warnings": raw["warnings"],
        "csv_path": csv_path,
        "md_path": md_path,
        "notes_path": notes_path,
        "wandb_run_url": wandb_url,
        "raw": raw,
    }
