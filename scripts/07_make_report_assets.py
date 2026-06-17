#!/usr/bin/env python
"""Generate a final report draft from available experiment assets."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import NA, collect_and_write_report_assets
from src.paths import ensure_dir, resolve_path


REPORT_PATH = resolve_path("report_assets/final_report_draft.md")


def read_text_or_na(path: str | Path) -> str:
    """Read text file if available, otherwise return Not Available."""
    resolved = resolve_path(path)
    if not resolved.exists():
        return NA
    return resolved.read_text(encoding="utf-8")


def file_link(path: str | Path) -> str:
    """Return a Markdown path reference or Not Available."""
    resolved = resolve_path(path)
    return f"`{path}`" if resolved.exists() else NA


def load_json_or_none(path: str | Path) -> dict[str, Any] | None:
    """Load JSON if available."""
    resolved = resolve_path(path)
    if not resolved.exists():
        return None
    with resolved.open("r", encoding="utf-8") as file:
        return json.load(file)


def metric_value(metrics: dict[str, Any] | None, key: str) -> Any:
    """Return metric value or Not Available."""
    if not metrics or metrics.get(key) is None:
        return NA
    return metrics[key]


def available_figures() -> list[str]:
    """Return Markdown bullet lines for available figure assets."""
    figure_paths = [
        "outputs/figures/train_action_l1_curve.png",
        "outputs/figures/val_action_l1_curve.png",
        "outputs/figures/eval_action_l1_on_D.png",
        "outputs/figures/action_error_mean_on_D.png",
        "outputs/figures/action_error_by_dim_on_D.png",
        "outputs/figures/success_rate_on_D.png",
    ]
    lines = []
    for path in figure_paths:
        status = "Available" if resolve_path(path).exists() else NA
        lines.append(f"- {path}: {status}")
    return lines


def make_report() -> Path:
    """Create final_report_draft.md without fabricating unavailable results."""
    assets = collect_and_write_report_assets()
    summary_md = read_text_or_na("report_assets/result_tables/summary_metrics.md")
    notes_md = read_text_or_na("report_assets/result_analysis_notes.md")
    a_metrics = load_json_or_none("outputs/eval/A_only_on_D/metrics.json")
    abc_metrics = load_json_or_none("outputs/eval/ABC_on_D/metrics.json")

    success_available = (
        metric_value(a_metrics, "success_rate") != NA
        or metric_value(abc_metrics, "success_rate") != NA
    )
    eval_mode = metric_value(a_metrics, "mode")
    if eval_mode == NA:
        eval_mode = metric_value(abc_metrics, "mode")

    content = f"""# Final Report Draft: LeRobot ACT CALVIN Cross-environment Generalization

## 1. 任务背景

本实验面向具身智能中的动作策略学习与跨环境泛化问题。课程任务要求使用 LeRobot 框架中的 ACT policy，在 CALVIN 数据集上比较单环境训练与多环境联合训练在未见环境 D 上的 zero-shot 表现。

## 2. 实验目的

本项目不是提出新算法，而是进行受控对比实验。核心问题是：在 ACT 架构和训练超参数保持一致时，使用 A/B/C 多环境数据训练的 ACT-ABC 是否比仅使用 A 数据训练的 ACT-A-only 更能泛化到环境 D。

## 3. 数据集与环境划分

- ACT-A-only 训练数据：CALVIN A / `splitA`
- ACT-ABC 训练数据：CALVIN A/B/C / `splitA`, `splitB`, `splitC`
- Zero-shot 测试数据：CALVIN D / `splitD`
- D 环境只用于最终评估，不参与训练、验证或调参。

Processed dataset evidence:

- A-only processed dataset: {file_link("data/processed/calvin_A_lerobot")}
- ABC processed dataset: {file_link("data/processed/calvin_ABC_lerobot")}
- D processed dataset: {file_link("data/processed/calvin_D_lerobot")}

## 4. 方法：LeRobot + ACT

实验使用 LeRobot 中的 ACT policy。ACT 通过 action chunking 一次预测一段未来动作，而不是只预测单步动作。该机制有助于动作序列连续性和短时执行稳定性，但不直接解决视觉分布偏移。

## 5. 实验设置与公平对比

公平性原则：

- 相同 ACT 网络架构；
- 相同 chunk size；
- 相同 batch size；
- 相同 learning rate；
- 相同 optimizer；
- 相同训练步数；
- 相同 image/state/action key；
- 唯一区别主要是训练数据来源、run name 和输出路径。

Fair comparison evidence:

- Fair comparison check: {file_link("report_assets/fair_comparison_check.md")}
- A-only config snapshot: {file_link("outputs/train/act_A_only/config_snapshot.yaml")}
- ABC config snapshot: {file_link("outputs/train/act_ABC/config_snapshot.yaml")}

## 6. 训练过程对比

训练指标主要使用 Action L1 Loss。WandB 中应导出以下曲线作为报告主图：

- `train/train_action_l1_loss`
- `val/action_l1_loss`

本地备份图状态：

{chr(10).join(available_figures())}

Summary metrics:

{summary_md}

## 7. D 环境 zero-shot 评估

评估模式：{eval_mode}

ACT-A-only on D:

- Action L1 Loss: {metric_value(a_metrics, "action_l1_loss")}
- Action Error Mean: {metric_value(a_metrics, "action_error_mean")}
- Success Rate: {metric_value(a_metrics, "success_rate")}

ACT-ABC on D:

- Action L1 Loss: {metric_value(abc_metrics, "action_l1_loss")}
- Action Error Mean: {metric_value(abc_metrics, "action_error_mean")}
- Success Rate: {metric_value(abc_metrics, "success_rate")}

Success Rate availability: {"Available" if success_available else "Not Available"}.

如果当前只有 offline evaluation，则这些结果只能说明模型在 D 离线轨迹上的动作预测误差，不能等同于真实闭环任务成功率。

## 8. Action Chunking 在 Visual Distribution Shift 下的分析

Action chunking 的优势是提升动作序列的短期连续性，减少逐步预测带来的抖动。在视觉输入分布与训练环境接近时，连续动作块可能提升执行稳定性。

但在 D 环境存在视觉分布偏移时，action chunking 并不能保证感知表征正确。如果模型对 D 中图像、背景、物体或相机差异理解错误，它可能连续执行一段平滑但错误的动作。因此，报告中应将 action chunking 描述为时序动作建模机制，而不是视觉泛化问题的根本解决方案。

## 9. 失败案例与局限性

- Failure cases file, A-only on D: {file_link("outputs/eval/A_only_on_D/failure_cases.json")}
- Failure cases file, ABC on D: {file_link("outputs/eval/ABC_on_D/failure_cases.json")}
- 如果无 rollout simulator，则无法得到真实 Success Rate、平均 episode 长度和视频失败案例。
- Offline action error 不能完全反映闭环控制中的状态分布偏移和误差累积。
- 当前结论仅适用于本实验 split、模型设置和计算预算。

## 10. 结论

本节应在真实训练和评估完成后填写。若关键指标为 Not Available，不应写出“ABC 一定更好”或“A-only 一定更差”等结论。应基于 `summary_metrics.md` 和 WandB 导出曲线进行保守表述。

## 11. 复现说明

推荐运行顺序：

```bash
python scripts/00_check_env.py
python scripts/02_prepare_dataset.py --split A_only
python scripts/02_prepare_dataset.py --split ABC
python scripts/02_prepare_dataset.py --split D
python scripts/03_train_act.py --config configs/train_A_only.yaml --device auto
python scripts/03_train_act.py --config configs/train_ABC.yaml --device auto
python scripts/check_fair_comparison.py
python scripts/04_eval_on_D.py --config configs/eval_A_on_D.yaml --mode offline --device auto
python scripts/04_eval_on_D.py --config configs/eval_ABC_on_D.yaml --mode offline --device auto
python scripts/05_collect_metrics.py
python scripts/06_plot_results.py
python scripts/07_make_report_assets.py
python scripts/08_final_audit.py
```

WandB 导出说明见 `report_assets/wandb_export_guide.md`。

## Analysis Notes

{notes_md}
"""
    ensure_dir(REPORT_PATH.parent)
    REPORT_PATH.write_text(content, encoding="utf-8")
    return REPORT_PATH


def main() -> None:
    """Generate final report draft."""
    path = make_report()
    print(json.dumps({"final_report_draft": str(path)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
