#!/usr/bin/env python
"""Audit whether ACT-A-only and ACT-ABC form a fair controlled comparison."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import ensure_dir, resolve_path


DEFAULT_A_SNAPSHOT = "outputs/train/act_A_only/config_snapshot.yaml"
DEFAULT_ABC_SNAPSHOT = "outputs/train/act_ABC/config_snapshot.yaml"
DEFAULT_REPORT_PATH = "report_assets/fair_comparison_check.md"

ALLOWED_DIFFERENCES = {
    "dataset_path",
    "run_name",
    "output_dir",
    "train_envs",
    "hf_dataset_subdirs",
}

CORE_CHECK_KEYS = [
    "policy_type",
    "network_architecture",
    "chunk_size",
    "batch_size",
    "learning_rate",
    "num_train_steps",
    "optimizer",
    "weight_decay",
    "loss_function",
    "gradient_clip_norm",
    "image_keys",
    "state_key",
    "action_key",
    "task_key",
    "temporal_ensemble",
]


def load_snapshot(path: str | Path) -> dict[str, Any]:
    """Load a config snapshot, failing clearly if it is missing."""
    resolved = resolve_path(path)
    if not resolved.exists():
        raise FileNotFoundError(
            f"Missing config snapshot: {resolved}. Run both training commands first, "
            "or run smoke tests that create config_snapshot.yaml."
        )
    with resolved.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def markdown_value(value: Any) -> str:
    """Format a Python value for terminal/Markdown tables."""
    text = repr(value)
    return text.replace("\n", " ")


def add_check(rows: list[dict[str, str]], status: str, item: str, a_value: Any, abc_value: Any, note: str) -> None:
    """Append one audit row."""
    rows.append(
        {
            "status": status,
            "item": item,
            "A-only": markdown_value(a_value),
            "ABC": markdown_value(abc_value),
            "note": note,
        }
    )


def readme_declares_equal_epoch(readme_path: str | Path = "README.md") -> bool:
    """Return whether README explicitly frames training fairness as equal epoch."""
    path = resolve_path(readme_path)
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8").lower()
    return "equal epoch" in text or "相同 epoch" in text or "相同 epoch 数" in text


def audit_configs(a_config: dict[str, Any], abc_config: dict[str, Any]) -> list[dict[str, str]]:
    """Run fairness checks and return report rows."""
    rows: list[dict[str, str]] = []

    for key in CORE_CHECK_KEYS:
        a_value = a_config.get(key)
        abc_value = abc_config.get(key)
        if key == "num_train_steps" and a_value != abc_value and readme_declares_equal_epoch():
            add_check(
                rows,
                "WARN",
                key,
                a_value,
                abc_value,
                "README appears to describe equal-epoch comparison; explain this choice in the report.",
            )
        elif a_value == abc_value:
            add_check(rows, "PASS", key, a_value, abc_value, "Matched.")
        else:
            add_check(rows, "FAIL", key, a_value, abc_value, "Core hyperparameter differs.")

    if a_config.get("seed") == abc_config.get("seed"):
        add_check(rows, "PASS", "seed", a_config.get("seed"), abc_config.get("seed"), "Matched.")
    else:
        add_check(
            rows,
            "WARN",
            "seed",
            a_config.get("seed"),
            abc_config.get("seed"),
            "Different seeds increase stochastic variation; report this if intentional.",
        )

    for name, config in [("A-only", a_config), ("ABC", abc_config)]:
        train_envs = config.get("train_envs", [])
        status = "FAIL" if "D" in train_envs else "PASS"
        note = "D appears in a training split." if status == "FAIL" else "D is not used for training."
        add_check(rows, status, f"{name} train_envs excludes D", train_envs, train_envs, note)

    expected_differences = []
    unexpected_differences = []
    all_keys = sorted(set(a_config) | set(abc_config))
    for key in all_keys:
        if a_config.get(key) == abc_config.get(key):
            continue
        if key in ALLOWED_DIFFERENCES:
            expected_differences.append(key)
        else:
            unexpected_differences.append(key)

    add_check(
        rows,
        "PASS" if expected_differences else "WARN",
        "allowed differences",
        expected_differences,
        expected_differences,
        "Expected differences should mainly describe data source and run identity.",
    )
    add_check(
        rows,
        "PASS" if not unexpected_differences else "FAIL",
        "unexpected differences",
        unexpected_differences,
        unexpected_differences,
        "Only dataset_path, run_name, output_dir, train_envs, and hf_dataset_subdirs should differ.",
    )
    return rows


def overall_status(rows: list[dict[str, str]]) -> str:
    """Return FAIL, WARN, or PASS for the whole audit."""
    statuses = {row["status"] for row in rows}
    if "FAIL" in statuses:
        return "FAIL"
    if "WARN" in statuses:
        return "WARN"
    return "PASS"


def table_text(rows: list[dict[str, str]]) -> str:
    """Render rows as a Markdown-compatible table."""
    headers = ["status", "item", "A-only", "ABC", "note"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[header].replace("|", "\\|") for header in headers) + " |")
    return "\n".join(lines)


def write_report(
    rows: list[dict[str, str]],
    report_path: str | Path,
    a_snapshot: str | Path,
    abc_snapshot: str | Path,
) -> Path:
    """Write the Markdown audit report."""
    resolved = resolve_path(report_path)
    ensure_dir(resolved.parent)
    status = overall_status(rows)
    content = "\n".join(
        [
            "# Fair Comparison Check",
            "",
            f"Overall status: **{status}**",
            "",
            "## Inputs",
            "",
            f"- A-only snapshot: `{a_snapshot}`",
            f"- ABC snapshot: `{abc_snapshot}`",
            "",
            "## Audit Table",
            "",
            table_text(rows),
            "",
            "## Interpretation",
            "",
            "- PASS means the checked item satisfies the controlled-comparison requirement.",
            "- WARN means the experiment may still be usable, but the report must explain the choice.",
            "- FAIL means the comparison violates the assignment requirement and should be fixed before reporting.",
            "",
        ]
    )
    resolved.write_text(content, encoding="utf-8")
    return resolved


def main() -> None:
    """Run the fairness audit and write a report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-snapshot", default=DEFAULT_A_SNAPSHOT)
    parser.add_argument("--abc-snapshot", default=DEFAULT_ABC_SNAPSHOT)
    parser.add_argument("--report", default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()

    try:
        a_config = load_snapshot(args.a_snapshot)
        abc_config = load_snapshot(args.abc_snapshot)
        rows = audit_configs(a_config, abc_config)
        report_path = write_report(rows, args.report, args.a_snapshot, args.abc_snapshot)
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1) from None

    status = overall_status(rows)
    print(table_text(rows))
    print(f"\nOverall status: {status}")
    print(f"Report written: {report_path}")
    if status == "FAIL":
        raise SystemExit(1)
    if status == "WARN":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
