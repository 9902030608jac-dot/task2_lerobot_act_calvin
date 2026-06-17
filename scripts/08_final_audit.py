#!/usr/bin/env python
"""Final project audit for course Task 2 requirements."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import ensure_dir, resolve_path


AUDIT_PATH = resolve_path("report_assets/final_audit.md")


def exists(path: str | Path) -> bool:
    """Return whether path exists."""
    return resolve_path(path).exists()


def load_yaml_if_exists(path: str | Path) -> dict[str, Any]:
    """Load YAML file if it exists."""
    resolved = resolve_path(path)
    if not resolved.exists():
        return {}
    with resolved.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def load_json_if_exists(path: str | Path) -> dict[str, Any]:
    """Load JSON file if it exists."""
    resolved = resolve_path(path)
    if not resolved.exists():
        return {}
    with resolved.open("r", encoding="utf-8") as file:
        return json.load(file)


def status_line(status: str, item: str, evidence: str, note: str = "") -> dict[str, str]:
    """Create one audit row."""
    return {"status": status, "item": item, "evidence": evidence, "note": note}


def path_check(item: str, path: str, required: bool = True) -> dict[str, str]:
    """Audit one required/optional path."""
    ok = exists(path)
    if ok:
        status = "PASS"
    else:
        status = "FAIL" if required else "WARN"
    return status_line(status, item, path, "Exists." if ok else "Missing.")


def table(rows: list[dict[str, str]]) -> str:
    """Render rows as Markdown table."""
    headers = ["status", "item", "evidence", "note"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[h].replace("|", "\\|") for h in headers) + " |")
    return "\n".join(lines)


def train_envs_include_d() -> bool:
    """Return whether any train config/snapshot includes D."""
    paths = [
        "configs/train_A_only.yaml",
        "configs/train_ABC.yaml",
        "outputs/train/act_A_only/config_snapshot.yaml",
        "outputs/train/act_ABC/config_snapshot.yaml",
    ]
    for path in paths:
        data = load_yaml_if_exists(path)
        if "D" in data.get("train_envs", []):
            return True
    return False


def metric_has_error(metrics: dict[str, Any]) -> bool:
    """Return whether metrics contain an action-error fallback."""
    return metrics.get("action_l1_loss") is not None or metrics.get("action_error_mean") is not None


def run_fairness_check() -> tuple[str, str]:
    """Run fairness audit if possible."""
    result = subprocess.run(
        [sys.executable, "scripts/check_fair_comparison.py"],
        cwd=resolve_path("."),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode == 0:
        return "PASS", "Fair comparison audit passed."
    if result.returncode == 2:
        return "WARN", "Fair comparison audit returned WARN."
    return "FAIL", (result.stderr or result.stdout).strip()


def make_audit() -> Path:
    """Create final_audit.md."""
    rows: list[dict[str, str]] = []
    rows.extend(
        [
            path_check("train_A_only.yaml", "configs/train_A_only.yaml"),
            path_check("train_ABC.yaml", "configs/train_ABC.yaml"),
            path_check("eval_A_on_D.yaml", "configs/eval_A_on_D.yaml"),
            path_check("eval_ABC_on_D.yaml", "configs/eval_ABC_on_D.yaml"),
            path_check("A-only processed dataset", "data/processed/calvin_A_lerobot"),
            path_check("ABC processed dataset", "data/processed/calvin_ABC_lerobot"),
            path_check("D processed dataset", "data/processed/calvin_D_lerobot"),
            path_check("A-only checkpoint", "outputs/train/act_A_only/checkpoints/final/checkpoint.pt"),
            path_check("ABC checkpoint", "outputs/train/act_ABC/checkpoints/final/checkpoint.pt"),
            path_check("A-only train_metrics.csv", "outputs/train/act_A_only/train_metrics.csv"),
            path_check("ABC train_metrics.csv", "outputs/train/act_ABC/train_metrics.csv"),
            path_check("A-only on D metrics.json", "outputs/eval/A_only_on_D/metrics.json"),
            path_check("ABC on D metrics.json", "outputs/eval/ABC_on_D/metrics.json"),
            path_check("Training loss curve", "outputs/figures/train_action_l1_curve.png"),
            path_check("Validation metric curve", "outputs/figures/val_action_l1_curve.png"),
            path_check("D eval Action L1 figure", "outputs/figures/eval_action_l1_on_D.png"),
            path_check("D eval action error figure", "outputs/figures/action_error_mean_on_D.png"),
            path_check("final_report_draft.md", "report_assets/final_report_draft.md"),
            path_check("WandB export guide", "report_assets/wandb_export_guide.md"),
        ]
    )

    fairness_status, fairness_note = run_fairness_check()
    rows.append(status_line(fairness_status, "Fair comparison audit", "scripts/check_fair_comparison.py", fairness_note))

    d_used = train_envs_include_d()
    rows.append(
        status_line(
            "FAIL" if d_used else "PASS",
            "D not used for training",
            "train_envs",
            "D found in train_envs." if d_used else "No D found in train_envs.",
        )
    )

    a_metrics = load_json_if_exists("outputs/eval/A_only_on_D/metrics.json")
    abc_metrics = load_json_if_exists("outputs/eval/ABC_on_D/metrics.json")
    has_success = a_metrics.get("success_rate") is not None or abc_metrics.get("success_rate") is not None
    has_error = metric_has_error(a_metrics) or metric_has_error(abc_metrics)
    rows.append(
        status_line(
            "PASS" if has_success else "WARN",
            "Success Rate available",
            "outputs/eval/*/metrics.json",
            "Available." if has_success else "Not available; report must use action error/offline metrics.",
        )
    )
    rows.append(
        status_line(
            "PASS" if has_success or has_error else "FAIL",
            "Fallback action error available if no Success Rate",
            "action_l1_loss/action_error_mean",
            "Action error available." if has_error else "No success_rate or action error found.",
        )
    )

    failed = [row for row in rows if row["status"] == "FAIL"]
    warnings = [row for row in rows if row["status"] == "WARN"]
    overall = "FAIL" if failed else "PASS"

    requirement_evidence = [
        status_line("PASS" if exists("configs/train_A_only.yaml") else "FAIL", "A-only ACT config", "configs/train_A_only.yaml"),
        status_line("PASS" if exists("configs/train_ABC.yaml") else "FAIL", "ABC ACT config", "configs/train_ABC.yaml"),
        status_line("PASS" if not d_used else "FAIL", "Zero-shot D isolation", "train_envs excludes D"),
        status_line("PASS" if exists("report_assets/wandb_export_guide.md") else "FAIL", "WandB visualization plan", "report_assets/wandb_export_guide.md"),
        status_line("PASS" if exists("report_assets/result_tables/summary_metrics.md") else "FAIL", "Summary metrics table", "report_assets/result_tables/summary_metrics.md"),
    ]

    content = "\n".join(
        [
            "# Final Project Audit",
            "",
            f"Overall status: **{overall}**",
            "",
            "## Necessary File Check",
            "",
            table(rows),
            "",
            "## Course Requirement Evidence",
            "",
            table(requirement_evidence),
            "",
            "## Missing Items",
            "",
            *(f"- {row['item']}: {row['evidence']} ({row['note']})" for row in failed),
            "- None" if not failed else "",
            "",
            "## Risk Items",
            "",
            *(f"- {row['item']}: {row['note']}" for row in warnings),
            "- None" if not warnings else "",
            "",
            "## Final Suggestions Before Submission",
            "",
            "- Run both ACT training jobs and verify checkpoints exist.",
            "- Run D offline or rollout evaluation for both checkpoints.",
            "- Run `python scripts/check_fair_comparison.py` and resolve FAIL items.",
            "- Use WandB-exported training and validation curves in the report.",
            "- If Success Rate is unavailable, explicitly report Action L1 Loss / Action Error and state that offline metrics are not closed-loop success.",
            "- Regenerate report assets with `python scripts/05_collect_metrics.py`, `python scripts/06_plot_results.py`, and `python scripts/07_make_report_assets.py` before final submission.",
            "",
        ]
    )
    ensure_dir(AUDIT_PATH.parent)
    AUDIT_PATH.write_text(content, encoding="utf-8")
    return AUDIT_PATH


def main() -> None:
    """Run final audit."""
    path = make_audit()
    print(json.dumps({"final_audit": str(path)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
